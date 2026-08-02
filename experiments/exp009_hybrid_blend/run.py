"""
experiments/exp009_hybrid_blend/run.py
------------------------------------------
Suite d'exp008 : le tuning Optuna n'a apporté un gain net que sur CatBoost
(+0.0008 vs manuel), pas sur LightGBM (-0.0009) ni XGBoost (-0.0008). Cette
expérience teste le blend "meilleur des deux mondes" : LightGBM manuel +
XGBoost manuel + CatBoost TUNÉ, avec poids optimisés sur les OOF.

4 modèles entraînés sur EXACTEMENT les mêmes folds en un seul passage de CV
(lgbm_manual, xgb_manual, cat_manual, cat_tuned) -- cat_manual est inclus
uniquement pour reproduire le blend all-manual d'exp006 sur les mêmes folds
que ce script (comparaison honnête, sans ré-entraîner ailleurs).

Usage :
    python -m experiments.exp009_hybrid_blend.run
"""
import numpy as np
import pandas as pd

from src.preprocessing import load_train, load_test
from src.augmentation import build_augmented_replicas
from src.features import featurize_many, featurize
from src.train import cross_validate_blend, default_fit_fn, weighted_score
from src.inference import predict_proba_ensemble, make_submission, save_models
from src.utils import SEED, MODELS_DIR

# Réutilise les factories/fit_fn déjà validés (aucune duplication)
from experiments.exp005_model_comparison.run import (
    lightgbm_factory, xgboost_factory, catboost_factory,
    catboost_fit_fn, xgboost_fit_fn,
)
from experiments.exp008_optuna_lgbm_catboost.run import (
    make_catboost_factory, optimize_blend_weights,
)

N_AUG = 10
N_SPLITS = 5
WEIGHT_GRID_STEP = 0.05

BASELINE_ALL_MANUAL = 0.9756       # exp006, référence à battre
BASELINE_ALL_TUNED_WEIGHTED = 0.9755   # exp008


def load_catboost_tuned_factory():
    """Charge les hyperparamètres CatBoost tunés (exp008, ligne model=='catboost')."""
    path = MODELS_DIR / "exp008_optuna_best_params.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable -- lance exp008 d'abord (il écrit les "
            "hyperparamètres tunés de LightGBM et CatBoost)."
        )
    df = pd.read_csv(path)
    row = df[df["model"] == "catboost"].iloc[0]
    params = {
        "iterations": int(row["iterations"]),
        "learning_rate": row["learning_rate"],
        "depth": int(row["depth"]),
        "l2_leaf_reg": row["l2_leaf_reg"],
        "bagging_temperature": row["bagging_temperature"],
        "random_strength": row["random_strength"],
    }
    return make_catboost_factory(params)


def run_cv(seed=SEED, verbose=True):
    """
    Entraîne les 4 modèles (lgbm_manual, xgb_manual, cat_manual, cat_tuned)
    sur les mêmes folds pour un `seed` donné. Retourne le dict oof_per_model
    (utilisé aussi bien par run.py que par compare_seeds.py).
    """
    cat_tuned_factory = load_catboost_tuned_factory()

    MODEL_SPECS = [
        ("lightgbm_manual", lightgbm_factory, default_fit_fn),
        ("xgboost_manual", xgboost_factory, xgboost_fit_fn),
        ("catboost_manual", catboost_factory, catboost_fit_fn),
        ("catboost_tuned", cat_tuned_factory, catboost_fit_fn),
    ]

    tr_ids, y, tr_cube = load_train()
    cubes, ys, groups = build_augmented_replicas(tr_cube, y, n_aug=N_AUG, seed=seed)
    X = featurize_many(cubes)
    Y = np.concatenate(ys)
    G = np.concatenate(groups)

    result = cross_validate_blend(MODEL_SPECS, X, Y, G, n_splits=N_SPLITS,
                                   threshold=0.5, seed=seed, verbose=verbose)
    return result, X, Y, G


def compute_blends(oof_per_model, Y, threshold=0.5, step=WEIGHT_GRID_STEP):
    """
    À partir des OOF des 4 modèles, calcule les 3 blends comparés :
    all_manual (= exp006 sur ces folds), hybrid_equal, hybrid_weighted.
    Retourne un dict de métriques + les poids optimaux du blend hybride.
    """
    from sklearn.metrics import f1_score, roc_auc_score

    def score_of(proba):
        auc = roc_auc_score(Y, proba)
        f1 = f1_score(Y, (proba > threshold).astype(int))
        return {"auc": auc, "f1": f1, "score": weighted_score(f1, auc)}

    all_manual = (oof_per_model["lightgbm_manual"] + oof_per_model["xgboost_manual"]
                  + oof_per_model["catboost_manual"]) / 3
    hybrid_equal = (oof_per_model["lightgbm_manual"] + oof_per_model["xgboost_manual"]
                    + oof_per_model["catboost_tuned"]) / 3

    hybrid_oof = {
        "lightgbm_manual": oof_per_model["lightgbm_manual"],
        "xgboost_manual": oof_per_model["xgboost_manual"],
        "catboost_tuned": oof_per_model["catboost_tuned"],
    }
    best_weights, _ = optimize_blend_weights(hybrid_oof, Y, threshold=threshold, step=step)
    total_w = sum(best_weights.values())
    hybrid_weighted = sum(best_weights[n] * hybrid_oof[n] for n in hybrid_oof) / total_w

    metrics = {
        "all_manual": score_of(all_manual),
        "hybrid_equal": score_of(hybrid_equal),
        "hybrid_weighted": score_of(hybrid_weighted),
    }
    return metrics, best_weights


def main():
    print(f"{'=' * 70}\nexp009 -- blend hybride (lgbm/xgb manuels + catboost tuné), "
          f"seed={SEED}\n{'=' * 70}")
    result, X, Y, G = run_cv(seed=SEED, verbose=True)

    print("\n--- Modèles individuels (mêmes folds) ---")
    rows = [{"model": name, **m} for name, m in result["metrics"].items() if name != "blend"]
    print(pd.DataFrame(rows).sort_values("score", ascending=False).to_string(index=False))

    metrics, best_weights = compute_blends(result["oof_per_model"], Y, threshold=0.5,
                                            step=WEIGHT_GRID_STEP)

    print("\n--- Comparaison des blends ---")
    df = pd.DataFrame([{"blend": name, **m} for name, m in metrics.items()]) \
        .sort_values("score", ascending=False)
    print(df.to_string(index=False))
    print(f"\nPoids optimaux du blend hybride : {best_weights}")

    delta_vs_exp006 = metrics["hybrid_weighted"]["score"] - BASELINE_ALL_MANUAL
    delta_vs_all_manual_here = metrics["hybrid_weighted"]["score"] - metrics["all_manual"]["score"]
    print(f"\nhybrid_weighted vs exp006 (référence 0.9756) : {delta_vs_exp006:+.4f}")
    print(f"hybrid_weighted vs all_manual (même seed, ici) : {delta_vs_all_manual_here:+.4f}")
    print(f"all_manual (ce run, seed={SEED}) vs exp006 (0.9756) -- doit être proche de 0 "
          f"si reproductible : {metrics['all_manual']['score'] - BASELINE_ALL_MANUAL:+.4f}")

    if abs(delta_vs_all_manual_here) < 0.001:
        print("\n!! Delta < 0.001 sur seed unique -- lance compare_seeds.py avant de "
              "conclure (même règle que H3/H5/exp002/exp003).")

    # ------------------------------------------------------------ soumission
    te_ids, _, te_cube = load_test()
    Xte = featurize(te_cube)[X.columns]

    proba_test_per_model = {
        name: predict_proba_ensemble(result["models_per_model"][name], Xte)
        for name in result["models_per_model"]
    }
    hybrid_test = {
        "lightgbm_manual": proba_test_per_model["lightgbm_manual"],
        "xgboost_manual": proba_test_per_model["xgboost_manual"],
        "catboost_tuned": proba_test_per_model["catboost_tuned"],
    }
    total_w = sum(best_weights.values())
    proba_test_weighted = sum(best_weights[n] * hybrid_test[n] for n in hybrid_test) / total_w

    make_submission(te_ids, proba_test_weighted, threshold=0.5,
                     filename="exp009_hybrid_blend_submission.csv")

    for name, models in result["models_per_model"].items():
        save_models(models, name=f"exp009_{name}")

    df.to_csv(MODELS_DIR / "exp009_blend_comparison.csv", index=False)
    pd.DataFrame([best_weights]).to_csv(MODELS_DIR / "exp009_best_weights.csv", index=False)
    print(f"\nRésultats écrits : {MODELS_DIR / 'exp009_blend_comparison.csv'}")
    print(f"Poids écrits : {MODELS_DIR / 'exp009_best_weights.csv'}")
    print("\nRappel : confirme sur compare_seeds.py avant d'adopter définitivement.")


if __name__ == "__main__":
    main()

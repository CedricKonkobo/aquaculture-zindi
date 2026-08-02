"""
experiments/exp008_optuna_lgbm_catboost/run.py
--------------------------------------------------
Étape 6 (2/2) : complète l'optimisation d'hyperparamètres commencée à
exp007 (XGBoost) en tunant LightGBM et CatBoost avec le même protocole en
2 phases (recherche proxy -> confirmation config complète).

Puis, étape 7 : reconstruit le blend d'exp006 (0.9756, poids égaux) avec
les 3 modèles tunés (XGBoost tuné d'exp007 + LightGBM/CatBoost tunés ici)
sur EXACTEMENT les mêmes folds, et cherche des poids de blend optimaux par
grid search sur le simplexe plutôt que la moyenne simple.

Prérequis : exp007 doit avoir tourné et avoir écrit
outputs/models/exp007_optuna_best_params.csv (sinon on retombe sur le
XGBoost manuel d'exp005 avec un avertissement).

Usage :
    python -m experiments.exp008_optuna_lgbm_catboost.run [n_trials_lgbm] [n_trials_cat]
    (n_trials par défaut : 40 chacun)
"""
import sys
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from catboost import CatBoostClassifier

from src.preprocessing import load_train, load_test
from src.augmentation import build_augmented_replicas
from src.features import featurize_many, featurize
from src.train import cross_validate, cross_validate_blend, default_fit_fn
from src.inference import predict_proba_ensemble, make_submission, save_models
from src.utils import SEED, MODELS_DIR

# Réutilise les fit_fn déjà validés à exp005 (aucune duplication)
from experiments.exp005_model_comparison.run import (
    xgboost_factory, catboost_fit_fn, xgboost_fit_fn,
)

SPW = 1.6                      # décision H2, non retunée ici (comme exp007)
N_TRIALS_LGBM = int(sys.argv[1]) if len(sys.argv) > 1 else 40
N_TRIALS_CAT = int(sys.argv[2]) if len(sys.argv) > 2 else 40
SEARCH_N_AUG = 5
SEARCH_N_SPLITS = 3
FINAL_N_AUG = 10
FINAL_N_SPLITS = 5
WEIGHT_GRID_STEP = 0.05

BASELINE_LGBM = 0.9750         # exp005, lightgbm manuel
BASELINE_CATBOOST = 0.9741     # exp005, catboost manuel
BASELINE_BLEND = 0.9756        # exp006, blend équipoids (3 modèles manuels)


# --------------------------------------------------------------- dataset
def build_dataset(n_aug, seed=SEED):
    tr_ids, y, tr_cube = load_train()
    cubes, ys, groups = build_augmented_replicas(tr_cube, y, n_aug=n_aug, seed=seed)
    X = featurize_many(cubes)
    Y = np.concatenate(ys)
    G = np.concatenate(groups)
    return X, Y, G


# ------------------------------------------------------- LightGBM (Optuna)
def sample_params_lgbm(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 400, 2000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 20.0, log=True),
    }


def make_lgbm_factory(params):
    def factory():
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            min_child_samples=params["min_child_samples"],
            feature_fraction=params["feature_fraction"],
            bagging_fraction=params["bagging_fraction"],
            bagging_freq=1,
            lambda_l1=params["lambda_l1"],
            lambda_l2=params["lambda_l2"],
            scale_pos_weight=SPW,
            verbose=-1,
            seed=SEED,
        )
    return factory


# ------------------------------------------------------- CatBoost (Optuna)
def sample_params_cat(trial):
    return {
        "iterations": trial.suggest_int("iterations", 400, 2000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
    }


def make_catboost_factory(params):
    def factory():
        return CatBoostClassifier(
            iterations=params["iterations"],
            learning_rate=params["learning_rate"],
            depth=params["depth"],
            l2_leaf_reg=params["l2_leaf_reg"],
            bagging_temperature=params["bagging_temperature"],
            random_strength=params["random_strength"],
            eval_metric="AUC",
            random_seed=SEED,
            scale_pos_weight=SPW,
            verbose=False,
            allow_writing_files=False,
        )
    return factory


# --------------------------------------------------------- tuning générique
def tune_model(name, sample_params_fn, make_factory_fn, fit_fn, n_trials,
                baseline_score, trials_csv_name):
    print(f"\n{'=' * 70}\nTuning {name} -- phase 1 (recherche, {n_trials} essais, "
          f"proxy n_aug={SEARCH_N_AUG}, {SEARCH_N_SPLITS} folds)\n{'=' * 70}")
    X_search, Y_search, G_search = build_dataset(SEARCH_N_AUG, seed=SEED)

    def objective(trial):
        params = sample_params_fn(trial)
        factory = make_factory_fn(params)
        result = cross_validate(factory, X_search, Y_search, G_search,
                                 n_splits=SEARCH_N_SPLITS, threshold=0.5, seed=SEED,
                                 verbose=False, fit_fn=fit_fn)
        return result["score"]

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nMeilleur score (recherche, proxy) : {study.best_value:.4f}")
    print(f"Meilleurs hyperparamètres {name} :")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    trials_df = study.trials_dataframe().sort_values("value", ascending=False)
    trials_df.to_csv(MODELS_DIR / trials_csv_name, index=False)

    print(f"\nPhase 2 -- confirmation en config complète "
          f"(n_aug={FINAL_N_AUG}, {FINAL_N_SPLITS} folds)")
    X, Y, G = build_dataset(FINAL_N_AUG, seed=SEED)
    best_factory = make_factory_fn(study.best_params)
    result = cross_validate(best_factory, X, Y, G, n_splits=FINAL_N_SPLITS, threshold=0.5,
                             seed=SEED, verbose=True, fit_fn=fit_fn)

    print(f"\n{name} Optuna (config complète) : AUC={result['auc']:.4f}  "
          f"F1={result['f1']:.4f}  score={result['score']:.4f}")
    print(f"Référence exp005 ({name} manuel) : score={baseline_score:.4f}")
    print(f"Delta : {result['score'] - baseline_score:+.4f}")

    return study.best_params, result["score"]


# ------------------------------------------------------- poids de blend
def optimize_blend_weights(oof_per_model, Y, threshold=0.5, step=WEIGHT_GRID_STEP):
    """
    Grid search sur le simplexe des poids (somme=1, pas=`step`) pour
    maximiser le score composite -- exhaustif mais bon marché (3 modèles).
    Retourne (best_weights: dict, best_score: float).
    """
    from src.train import weighted_score
    from sklearn.metrics import f1_score, roc_auc_score

    names = list(oof_per_model.keys())
    n_steps = int(round(1.0 / step))
    best_score = -1.0
    best_weights = None

    # grille de triplets (w0, w1, w2) sur les pas de `step` qui somment à 1
    grid_vals = [round(i * step, 4) for i in range(n_steps + 1)]
    for w0, w1 in itertools.product(grid_vals, repeat=2):
        w2 = round(1.0 - w0 - w1, 4)
        if w2 < -1e-9 or w2 > 1.0 + 1e-9:
            continue
        w2 = max(w2, 0.0)
        weights = {names[0]: w0, names[1]: w1, names[2]: w2}
        blend = sum(weights[n] * oof_per_model[n] for n in names)
        auc = roc_auc_score(Y, blend)
        f1 = f1_score(Y, (blend > threshold).astype(int))
        score = weighted_score(f1, auc)
        if score > best_score:
            best_score = score
            best_weights = weights

    return best_weights, best_score


def main():
    # ---------------------------------------------------- 1) tuning LightGBM
    lgbm_params, lgbm_score = tune_model(
        "lightgbm", sample_params_lgbm, make_lgbm_factory, default_fit_fn,
        N_TRIALS_LGBM, BASELINE_LGBM, "exp008_optuna_lgbm_trials.csv",
    )

    # ---------------------------------------------------- 2) tuning CatBoost
    cat_params, cat_score = tune_model(
        "catboost", sample_params_cat, make_catboost_factory, catboost_fit_fn,
        N_TRIALS_CAT, BASELINE_CATBOOST, "exp008_optuna_catboost_trials.csv",
    )

    best_params_df = pd.DataFrame([
        {"model": "lightgbm", **lgbm_params},
        {"model": "catboost", **cat_params},
    ])
    best_params_df.to_csv(MODELS_DIR / "exp008_optuna_best_params.csv", index=False)

    # ---------------------------------------------------- 3) blend à 3 modèles tunés
    print(f"\n{'=' * 70}\nPhase 3 -- blend à 3 modèles tunés (mêmes folds)\n{'=' * 70}")

    xgb_params_path = MODELS_DIR / "exp007_optuna_best_params.csv"
    if xgb_params_path.exists():
        xgb_best = pd.read_csv(xgb_params_path).iloc[0].to_dict()
        from xgboost import XGBClassifier

        def xgboost_tuned_factory():
            return XGBClassifier(
                n_estimators=int(xgb_best["n_estimators"]),
                learning_rate=xgb_best["learning_rate"],
                max_depth=int(xgb_best["max_depth"]),
                subsample=xgb_best["subsample"],
                colsample_bytree=xgb_best["colsample_bytree"],
                min_child_weight=int(xgb_best["min_child_weight"]),
                reg_alpha=xgb_best["reg_alpha"],
                reg_lambda=xgb_best["reg_lambda"],
                gamma=xgb_best["gamma"],
                scale_pos_weight=SPW,
                eval_metric="auc",
                random_state=SEED,
                early_stopping_rounds=100,
            )
        print("XGBoost tuné (exp007) chargé depuis outputs/models/exp007_optuna_best_params.csv")
    else:
        print("!! exp007_optuna_best_params.csv introuvable -- repli sur le XGBoost "
              "manuel d'exp005 (lance exp007 d'abord pour un blend pleinement tuné).")
        xgboost_tuned_factory = xgboost_factory

    lgbm_tuned_factory = make_lgbm_factory(lgbm_params)
    cat_tuned_factory = make_catboost_factory(cat_params)

    MODEL_SPECS = [
        ("lightgbm_tuned", lgbm_tuned_factory, default_fit_fn),
        ("xgboost_tuned", xgboost_tuned_factory, xgboost_fit_fn),
        ("catboost_tuned", cat_tuned_factory, catboost_fit_fn),
    ]

    tr_ids, y, tr_cube = load_train()
    te_ids, _, te_cube = load_test()
    cubes, ys, groups = build_augmented_replicas(tr_cube, y, n_aug=FINAL_N_AUG, seed=SEED)
    X = featurize_many(cubes)
    Y = np.concatenate(ys)
    G = np.concatenate(groups)
    Xte = featurize(te_cube)[X.columns]

    result = cross_validate_blend(MODEL_SPECS, X, Y, G, n_splits=FINAL_N_SPLITS,
                                   threshold=0.5, seed=SEED, verbose=True)

    print("\n--- Blend équipoids (3 modèles tunés) ---")
    rows = [{"model": name, **m} for name, m in result["metrics"].items()]
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    print(df.to_string(index=False))

    equal_blend_score = result["metrics"]["blend"]["score"]

    # --------------------------------------- recherche des poids optimaux (OOF)
    best_weights, best_weighted_score = optimize_blend_weights(
        result["oof_per_model"], Y, threshold=0.5, step=WEIGHT_GRID_STEP,
    )
    print(f"\nMeilleurs poids (grid search, pas={WEIGHT_GRID_STEP}) : {best_weights}")
    print(f"Score blend pondéré (OOF) : {best_weighted_score:.4f}")
    print(f"Score blend équipoids (OOF, 3 modèles tunés) : {equal_blend_score:.4f}")
    print(f"Référence exp006 (blend équipoids, 3 modèles MANUELS) : {BASELINE_BLEND:.4f}")
    print(f"Delta pondéré vs exp006 : {best_weighted_score - BASELINE_BLEND:+.4f}")
    print(f"Delta pondéré vs équipoids (3 modèles tunés) : "
          f"{best_weighted_score - equal_blend_score:+.4f}")

    # ------------------------------------------------------------ soumission
    proba_test_per_model = {
        name: predict_proba_ensemble(result["models_per_model"][name], Xte)
        for name in result["models_per_model"]
    }
    total_w = sum(best_weights.values())
    proba_test_weighted = sum(
        best_weights[name] * proba_test_per_model[name] for name in proba_test_per_model
    ) / total_w

    make_submission(te_ids, proba_test_weighted, threshold=0.5,
                     filename="exp008_optuna_lgbm_catboost_submission.csv")

    for name, models in result["models_per_model"].items():
        save_models(models, name=f"exp008_{name}")

    df.to_csv(MODELS_DIR / "exp008_blend_comparison.csv", index=False)
    pd.DataFrame([best_weights]).to_csv(MODELS_DIR / "exp008_best_weights.csv", index=False)
    print(f"\nRésultats écrits : {MODELS_DIR / 'exp008_blend_comparison.csv'}")
    print(f"Poids écrits : {MODELS_DIR / 'exp008_best_weights.csv'}")


if __name__ == "__main__":
    main()

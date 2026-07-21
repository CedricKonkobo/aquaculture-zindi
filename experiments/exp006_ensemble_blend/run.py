"""
experiments/exp006_ensemble_blend/run.py
--------------------------------------------
Suite de exp005 : les 3 boostings (XGBoost, LightGBM, CatBoost) étaient très
proches (score OOF 0.9741-0.9754, écart dans la marge de bruit de CV). Plutôt
que de choisir un seul modèle, on teste un blend (moyenne simple des probas)
sur EXACTEMENT les mêmes folds (cross_validate_blend), pour voir s'il dépasse
le meilleur modèle individuel.

RandomForest/ExtraTrees exclus (nettement en retrait à exp005, cf. results.md).

Usage :
    python -m experiments.exp006_ensemble_blend.run
"""
import numpy as np
import pandas as pd

from src.preprocessing import load_train, load_test
from src.augmentation import build_augmented_replicas
from src.features import featurize_many, featurize
from src.train import cross_validate_blend, default_fit_fn
from src.inference import predict_proba_ensemble, make_submission, save_models
from src.utils import SEED, MODELS_DIR

# Réutilise les factories/fit_fn déjà validés à exp005 -- aucune duplication
from experiments.exp005_model_comparison.run import (
    lightgbm_factory, catboost_factory, xgboost_factory,
    catboost_fit_fn, xgboost_fit_fn,
)

N_AUG = 10

MODEL_SPECS = [
    ("lightgbm", lightgbm_factory, default_fit_fn),
    ("xgboost", xgboost_factory, xgboost_fit_fn),
    ("catboost", catboost_factory, catboost_fit_fn),
]


def main():
    tr_ids, y, tr_cube = load_train()
    te_ids, _, te_cube = load_test()

    cubes, ys, groups = build_augmented_replicas(tr_cube, y, n_aug=N_AUG, seed=SEED)
    X = featurize_many(cubes)
    Y = np.concatenate(ys)
    G = np.concatenate(groups)
    Xte = featurize(te_cube)[X.columns]
    print(f"train augmenté : {X.shape} | test : {Xte.shape}\n")

    result = cross_validate_blend(MODEL_SPECS, X, Y, G, n_splits=5, threshold=0.5,
                                   seed=SEED, verbose=True)

    print("\n--- Comparaison finale (mêmes folds pour tous) ---")
    rows = [{"model": name, **m} for name, m in result["metrics"].items()]
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    print(df.to_string(index=False))

    best_single = max(
        (m["score"] for name, m in result["metrics"].items() if name != "blend"),
    )
    blend_score = result["metrics"]["blend"]["score"]
    print(f"\nBlend vs meilleur modèle seul : {blend_score:.4f} vs {best_single:.4f} "
          f"(delta={blend_score - best_single:+.4f})")

    # --- prédiction test : blend = moyenne des 3 ensembles (chacun moyenné sur ses 5 folds)
    proba_test_per_model = {
        name: predict_proba_ensemble(result["models_per_model"][name], Xte)
        for name in result["models_per_model"]
    }
    proba_test_blend = sum(proba_test_per_model.values()) / len(proba_test_per_model)

    make_submission(te_ids, proba_test_blend, threshold=0.5,
                     filename="exp006_ensemble_blend_submission.csv")

    out = MODELS_DIR / "exp006_blend_comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nRésultats écrits : {out}")


if __name__ == "__main__":
    main()

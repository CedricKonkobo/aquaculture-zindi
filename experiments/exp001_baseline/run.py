"""
experiments/exp001_baseline/run.py
-----------------------------------
Reproduction exacte de la baseline d'origine (script `baseline_aquaculture.py`),
mais construite sur les modules de src/ au lieu de tout réécrire en un bloc.

C'est le point de référence (~100e place Zindi) à partir duquel toute nouvelle
expérience doit être comparée. Voir config.yaml pour les hyperparamètres et
results.md pour le score obtenu et les observations.

Usage :
    python -m experiments.exp001_baseline.run
"""
import numpy as np
import lightgbm as lgb

from src.preprocessing import load_train, load_test
from src.augmentation import build_augmented_replicas
from src.features import featurize_many, featurize
from src.train import cross_validate, feature_importance
from src.inference import predict_proba_ensemble, make_submission, save_models
from src.utils import SEED

N_AUG = 10
LGBM_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=40,
    feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
    lambda_l1=0.5, lambda_l2=5.0, n_estimators=1200, verbose=-1, seed=SEED,
)


def model_factory():
    return lgb.LGBMClassifier(
        **LGBM_PARAMS,
        # early stopping géré via callback dans train.py normalement ; ici on garde
        # la même mécanique que la baseline d'origine en le passant au fit via callbacks
    )


def main():
    tr_ids, y, tr_cube = load_train()
    te_ids, _, te_cube = load_test()

    cubes, ys, groups = build_augmented_replicas(tr_cube, y, n_aug=N_AUG, seed=SEED)
    X = featurize_many(cubes)
    Y = np.concatenate(ys)
    G = np.concatenate(groups)

    Xte = featurize(te_cube)[X.columns]
    print(f"train augmenté : {X.shape} | test : {Xte.shape}")

    result = cross_validate(model_factory, X, Y, G, n_splits=5, threshold=0.5, seed=SEED)

    print("\nTop 15 features :")
    print(feature_importance(result["models"][-1], X.columns, top_n=15).to_string())

    proba_test = predict_proba_ensemble(result["models"], Xte)
    make_submission(te_ids, proba_test, threshold=0.5, filename="exp001_baseline_submission.csv")
    save_models(result["models"], name="exp001_baseline")


if __name__ == "__main__":
    main()

"""
experiments/exp002_feature_reduction/run.py
--------------------------------------------
Test de l'hypothèse H5 (EDA, étape 7) : réduire la redondance entre bandes
brutes/indices très corrélés (src/features_v2.py) et comparer au score OOF
de exp001_baseline.

Identique à exp001_baseline/run.py en tout point (mêmes hyperparamètres,
même seed, même n_aug, même CV) SAUF l'import de featurize/featurize_many,
qui vient de src.features_v2 au lieu de src.features. C'est la seule
variable changée -> toute différence de score est imputable à H5.

Usage :
    python -m experiments.exp002_feature_reduction.run
"""
import numpy as np
import lightgbm as lgb

from src.preprocessing import load_train, load_test
from src.augmentation import build_augmented_replicas
from src.features_v2 import featurize_many, featurize   # <-- seule différence avec exp001
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
    return lgb.LGBMClassifier(**LGBM_PARAMS)


def main():
    tr_ids, y, tr_cube = load_train()
    te_ids, _, te_cube = load_test()

    cubes, ys, groups = build_augmented_replicas(tr_cube, y, n_aug=N_AUG, seed=SEED)
    X = featurize_many(cubes)
    Y = np.concatenate(ys)
    G = np.concatenate(groups)

    Xte = featurize(te_cube)[X.columns]
    print(f"train augmenté : {X.shape} | test : {Xte.shape}")
    print(f"(baseline exp001 avait 245 features -> vérifier la baisse ici)")

    result = cross_validate(model_factory, X, Y, G, n_splits=5, threshold=0.5, seed=SEED)

    print("\nTop 15 features :")
    print(feature_importance(result["models"][-1], X.columns, top_n=15).to_string())

    proba_test = predict_proba_ensemble(result["models"], Xte)
    make_submission(te_ids, proba_test, threshold=0.5, filename="exp002_feature_reduction_submission.csv")
    save_models(result["models"], name="exp002_feature_reduction")


if __name__ == "__main__":
    main()

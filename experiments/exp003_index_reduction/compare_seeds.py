"""
experiments/exp003_index_reduction/compare_seeds.py
------------------------------------------------------
Confirme (ou infirme) H3 sur plusieurs seeds : relance exp001_baseline (245
features, référence -- H5/exp002 a été rejetée) et exp003 (features_v3, garde
uniquement MNDWI + NDRE parmi les 5 indices eau/végétation corrélés).

Usage :
    python -m experiments.exp003_index_reduction.compare_seeds
"""
import numpy as np
import lightgbm as lgb
import pandas as pd

from src.preprocessing import load_train, load_test
from src.augmentation import build_augmented_replicas
from src.features import featurize_many as featurize_many_v1, featurize as featurize_v1
from src.features_v3 import featurize_many as featurize_many_v3, featurize as featurize_v3
from src.train import cross_validate

SEEDS = [42, 1, 7]          # mêmes seeds que exp002, pour comparabilité directe
N_AUG = 10
LGBM_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=40,
    feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
    lambda_l1=0.5, lambda_l2=5.0, n_estimators=1200, verbose=-1,
)


def run_once(seed, featurize_many, featurize):
    tr_ids, y, tr_cube = load_train()
    cubes, ys, groups = build_augmented_replicas(tr_cube, y, n_aug=N_AUG, seed=seed)
    X = featurize_many(cubes)
    Y = np.concatenate(ys)
    G = np.concatenate(groups)

    def model_factory():
        return lgb.LGBMClassifier(**LGBM_PARAMS, seed=seed)

    result = cross_validate(model_factory, X, Y, G, n_splits=5, threshold=0.5,
                             seed=seed, verbose=False)
    return X.shape[1], result["auc"], result["f1"], result["score"]


def main():
    rows = []
    for seed in SEEDS:
        n1, auc1, f1_1, s1 = run_once(seed, featurize_many_v1, featurize_v1)
        n3, auc3, f1_3, s3 = run_once(seed, featurize_many_v3, featurize_v3)
        rows.append({
            "seed": seed,
            "n_feat_v1": n1, "n_feat_v3": n3,
            "auc_v1": round(auc1, 4), "auc_v3": round(auc3, 4),
            "f1_v1": round(f1_1, 4), "f1_v3": round(f1_3, 4),
            "score_v1": round(s1, 4), "score_v3": round(s3, 4),
            "delta_score": round(s3 - s1, 4),
        })
        print(f"seed {seed}  v1 score={s1:.4f}  v3 score={s3:.4f}  delta={s3 - s1:+.4f}")

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    print(f"\nDelta moyen (v3 - v1) : {df['delta_score'].mean():+.4f}")
    print(f"Delta std             : {df['delta_score'].std():.4f}")
    print("\nRègle de décision : si |delta moyen| < delta std -> le delta n'est pas")
    print("significatif -> garder v3 pour la légèreté (perte non distinguable du bruit).")
    print("Si delta moyen < 0 ET |delta| > std sur plusieurs seeds -> rejeter H3, revenir à v1.")
    print("Si delta moyen >= 0 -> gain net, fusionner features_v3.py dans features.py.")

    out = "outputs/models/exp003_multiseed_comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nRésultats écrits : {out}")


if __name__ == "__main__":
    main()

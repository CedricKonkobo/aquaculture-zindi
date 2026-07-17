"""
experiments/exp002_feature_reduction/compare_seeds.py
-------------------------------------------------------
Confirme (ou infirme) H5 sur plusieurs seeds : relance exp001_baseline (245
features) et exp002 (features_v2, 185 features) avec les mêmes seeds pour
chacun, et affiche le delta de score pondéré seed par seed + la moyenne.

Usage :
    python -m experiments.exp002_feature_reduction.compare_seeds
"""
import numpy as np
import lightgbm as lgb
import pandas as pd

from src.preprocessing import load_train, load_test
from src.augmentation import build_augmented_replicas
from src.features import featurize_many as featurize_many_v1, featurize as featurize_v1
from src.features_v2 import featurize_many as featurize_many_v2, featurize as featurize_v2
from src.train import cross_validate

SEEDS = [42, 1, 7]          # ajouter/retirer des seeds ici si besoin
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
        n2, auc2, f1_2, s2 = run_once(seed, featurize_many_v2, featurize_v2)
        rows.append({
            "seed": seed,
            "n_feat_v1": n1, "n_feat_v2": n2,
            "auc_v1": round(auc1, 4), "auc_v2": round(auc2, 4),
            "f1_v1": round(f1_1, 4), "f1_v2": round(f1_2, 4),
            "score_v1": round(s1, 4), "score_v2": round(s2, 4),
            "delta_score": round(s2 - s1, 4),
        })
        print(f"seed {seed}  v1 score={s1:.4f}  v2 score={s2:.4f}  delta={s2 - s1:+.4f}")

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    print(f"\nDelta moyen (v2 - v1) : {df['delta_score'].mean():+.4f}")
    print(f"Delta std             : {df['delta_score'].std():.4f}")
    print("\nRègle de décision : si |delta moyen| < delta std -> le delta n'est pas")
    print("significatif -> garder v2 pour la légèreté (perte non distinguable du bruit).")
    print("Si delta moyen < 0 ET |delta| > std sur plusieurs seeds -> rejeter H5, revenir à v1.")

    out = "outputs/models/exp002_multiseed_comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nRésultats écrits : {out}")


if __name__ == "__main__":
    main()

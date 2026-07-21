"""
experiments/exp006_ensemble_blend/compare_seeds.py
------------------------------------------------------
Confirme (ou infirme) le gain du blend sur plusieurs seeds, vu que le delta
observé sur seed 42 (+0.0002) est plus petit que le bruit de CV typique
(~0.0002-0.0003, cf. H3/H5).

Usage :
    python -m experiments.exp006_ensemble_blend.compare_seeds
"""
import numpy as np
import pandas as pd

from src.preprocessing import load_train
from src.augmentation import build_augmented_replicas
from src.features import featurize_many
from src.train import cross_validate_blend, default_fit_fn
from experiments.exp005_model_comparison.run import (
    lightgbm_factory, catboost_factory, xgboost_factory,
    catboost_fit_fn, xgboost_fit_fn,
)

SEEDS = [42, 1, 7]   # mêmes seeds que H3/H5 pour comparabilité
N_AUG = 10

MODEL_SPECS = [
    ("lightgbm", lightgbm_factory, default_fit_fn),
    ("xgboost", xgboost_factory, xgboost_fit_fn),
    ("catboost", catboost_factory, catboost_fit_fn),
]


def run_once(seed):
    tr_ids, y, tr_cube = load_train()
    cubes, ys, groups = build_augmented_replicas(tr_cube, y, n_aug=N_AUG, seed=seed)
    X = featurize_many(cubes)
    Y = np.concatenate(ys)
    G = np.concatenate(groups)

    result = cross_validate_blend(MODEL_SPECS, X, Y, G, n_splits=5, threshold=0.5,
                                   seed=seed, verbose=False)
    scores = {name: m["score"] for name, m in result["metrics"].items()}
    best_single = max(v for k, v in scores.items() if k != "blend")
    return scores, best_single


def main():
    rows = []
    for seed in SEEDS:
        scores, best_single = run_once(seed)
        delta = scores["blend"] - best_single
        rows.append({"seed": seed, **scores, "best_single": round(best_single, 4),
                      "delta_blend_vs_best": round(delta, 4)})
        print(f"seed {seed}  blend={scores['blend']:.4f}  best_single={best_single:.4f}  "
              f"delta={delta:+.4f}")

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    print(f"\nDelta moyen (blend - best_single) : {df['delta_blend_vs_best'].mean():+.4f}")
    print(f"Delta std                          : {df['delta_blend_vs_best'].std():.4f}")
    print("\nMême règle que H3/H5 : si |delta moyen| < delta std -> pas significatif,")
    print("garder le blend quand même pour la robustesse structurelle (erreurs moins corrélées).")
    print("Si delta moyen < 0 et cohérent -> revenir au meilleur modèle seul (xgboost).")


if __name__ == "__main__":
    main()

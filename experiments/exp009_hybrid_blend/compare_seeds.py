"""
experiments/exp009_hybrid_blend/compare_seeds.py
------------------------------------------------------
Confirme (ou infirme) le gain du blend hybride (lgbm/xgb manuels + catboost
tuné, poids optimisés) sur plusieurs seeds, même règle de décision que
H3/H5 (exp002/exp003) : |delta moyen| < delta std -> bruit, ne pas adopter.

Pour chaque seed : entraîne les 4 modèles (lgbm_manual, xgb_manual,
cat_manual, cat_tuned) sur les mêmes folds, calcule all_manual (= exp006
recalculé sur ce seed) et hybrid_weighted, et le delta entre les deux.

Usage :
    python -m experiments.exp009_hybrid_blend.compare_seeds
"""
import pandas as pd

from src.utils import MODELS_DIR
from experiments.exp009_hybrid_blend.run import run_cv, compute_blends

SEEDS = [42, 1, 7]          # mêmes seeds que H3/H5/exp002/exp003
WEIGHT_GRID_STEP = 0.05


def main():
    rows = []
    for seed in SEEDS:
        print(f"\n{'=' * 70}\nseed={seed}\n{'=' * 70}")
        result, X, Y, G = run_cv(seed=seed, verbose=False)
        metrics, best_weights = compute_blends(result["oof_per_model"], Y, threshold=0.5,
                                                step=WEIGHT_GRID_STEP)

        all_manual_score = metrics["all_manual"]["score"]
        hybrid_score = metrics["hybrid_weighted"]["score"]
        delta = hybrid_score - all_manual_score

        rows.append({
            "seed": seed,
            "all_manual_score": round(all_manual_score, 4),
            "hybrid_weighted_score": round(hybrid_score, 4),
            "delta_score": round(delta, 4),
            "weights": best_weights,
        })
        print(f"seed {seed}  all_manual={all_manual_score:.4f}  "
              f"hybrid_weighted={hybrid_score:.4f}  delta={delta:+.4f}  "
              f"poids={best_weights}")

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    delta_mean = df["delta_score"].mean()
    delta_std = df["delta_score"].std()
    print(f"\nDelta moyen (hybrid_weighted - all_manual) : {delta_mean:+.4f}")
    print(f"Delta std                                   : {delta_std:.4f}")

    print("\nRègle de décision : si |delta moyen| < delta std -> bruit -> garder exp006")
    print("(blend all-manual, plus simple). Si delta moyen > 0 ET cohérent (même signe)")
    print("sur les 3 seeds -> adopter exp009 (hybrid_weighted) comme référence finale.")

    if abs(delta_mean) < delta_std:
        print("\n>> Condition de bruit vérifiée : delta non significatif.")
    else:
        sign_consistent = all((d > 0) == (delta_mean > 0) for d in df["delta_score"])
        verdict = "signal réel" if sign_consistent else "signe incohérent -> rester prudent"
        print(f"\n>> Condition de bruit NON vérifiée : {verdict}.")

    out = MODELS_DIR / "exp009_multiseed_comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nRésultats écrits : {out}")


if __name__ == "__main__":
    main()

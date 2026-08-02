"""
experiments/exp010_gap_diagnostic/run.py
--------------------------------------------
Diagnostic (pas de modèle) : vérifie que S2_GAP_P et la fenêtre [4, 6] mois
codées en dur dans src/utils.py / src/augmentation.py correspondent bien à
ce qu'on observe réellement dans data/Test.csv.

Toute l'augmentation du train (utilisée par exp001 à exp009) repose sur
cette hypothèse -- jamais vérifiée empiriquement jusqu'ici. C'est le levier
anti-décalage train/test le plus important du pipeline.

Usage :
    python -m experiments.exp010_gap_diagnostic.run
"""
import numpy as np
import pandas as pd

from src.preprocessing import load_test
from src.utils import BANDS, MONTHS, S1_BANDS, S2_BANDS, S2_GAP_P, MODELS_DIR, FIGURES_DIR

S1_IDX = [BANDS.index(b) for b in S1_BANDS]
S2_IDX = [BANDS.index(b) for b in S2_BANDS]

ASSUMED_MIN_LEN, ASSUMED_MAX_LEN = 4, 6


def analyze_sample(sample):
    """
    sample : array (12 mois, n_bands).
    Retourne un dict avec la longueur de fenêtre observée, si elle est
    consécutive, et le nombre de trous S2 à l'intérieur de la fenêtre.
    """
    s1_present = ~np.isnan(sample[:, S1_IDX]).all(axis=1)   # (12,) bool
    in_window_months = np.where(s1_present)[0]

    if len(in_window_months) == 0:
        return {"window_len": 0, "is_consecutive": True, "n_gaps": 0,
                "start_month": None, "end_month": None}

    start, end = in_window_months.min(), in_window_months.max()
    expected_span = np.arange(start, end + 1)
    is_consecutive = np.array_equal(in_window_months, expected_span)

    s2_all_nan = np.isnan(sample[:, S2_IDX]).all(axis=1)     # (12,) bool
    n_gaps = int(s2_all_nan[in_window_months].sum())

    return {
        "window_len": len(in_window_months),
        "is_consecutive": bool(is_consecutive),
        "n_gaps": n_gaps,
        "start_month": int(start),
        "end_month": int(end),
    }


def main():
    te_ids, _, te_cube = load_test()
    print(f"Test : {te_cube.shape[0]} échantillons\n")

    rows = [analyze_sample(te_cube[i]) for i in range(te_cube.shape[0])]
    df = pd.DataFrame(rows)
    df.insert(0, "ID", te_ids)

    # ------------------------------------------------------- longueur de fenêtre
    print("=== Longueur de fenêtre observée (mois avec S1 présent) ===")
    len_counts = df["window_len"].value_counts().sort_index()
    print(len_counts)
    print(f"\nHypothèse codée : uniforme sur [{ASSUMED_MIN_LEN}, {ASSUMED_MAX_LEN}]")
    out_of_range = df[(df["window_len"] < ASSUMED_MIN_LEN) | (df["window_len"] > ASSUMED_MAX_LEN)]
    print(f"Échantillons hors de [{ASSUMED_MIN_LEN}, {ASSUMED_MAX_LEN}] : "
          f"{len(out_of_range)} ({len(out_of_range) / len(df):.1%})")

    # ------------------------------------------------------- consécutivité
    pct_consecutive = df["is_consecutive"].mean()
    print(f"\n% de fenêtres réellement consécutives : {pct_consecutive:.1%}")
    if pct_consecutive < 0.95:
        print("!! Une part notable des fenêtres test ne sont PAS consécutives -- "
              "l'hypothèse de mask_window (fenêtre continue) est optimiste.")

    # ------------------------------------------------------- trous S2
    print("\n=== Nombre de trous S2 dans la fenêtre ===")
    gap_counts = df["n_gaps"].value_counts().sort_index()
    gap_p_empirical = (gap_counts / len(df)).reindex(range(len(S2_GAP_P)), fill_value=0.0)
    print(pd.DataFrame({
        "n_gaps": gap_p_empirical.index,
        "p_empirique": gap_p_empirical.values.round(4),
        "p_code_S2_GAP_P": list(S2_GAP_P) + [0.0] * max(0, len(gap_p_empirical) - len(S2_GAP_P)),
    }).to_string(index=False))

    max_gap_observed = df["n_gaps"].max()
    if max_gap_observed >= len(S2_GAP_P):
        print(f"\n!! Trous observés jusqu'à {max_gap_observed} mois, au-delà de "
              f"S2_GAP_P (défini seulement jusqu'à {len(S2_GAP_P) - 1}) -- "
              "mask_window ne peut PAS générer ces cas de figure sur le train.")

    diff = np.abs(gap_p_empirical.values[:len(S2_GAP_P)] - np.array(S2_GAP_P))
    print(f"\nÉcart absolu max (p_empirique vs S2_GAP_P) : {diff.max():.4f}")

    # ------------------------------------------------------------- verdict
    print("\n" + "=" * 70)
    len_ok = len(out_of_range) / len(df) < 0.05
    gap_ok = diff.max() < 0.05 and max_gap_observed < len(S2_GAP_P)
    consec_ok = pct_consecutive > 0.95
    if len_ok and gap_ok and consec_ok:
        print("VERDICT : hypothèses de augmentation.py globalement cohérentes "
              "avec le test réel. Pas de changement nécessaire.")
    else:
        print("VERDICT : écart notable détecté -- voir results.md pour la mise à jour "
              "recommandée de MIN_LEN/MAX_LEN et/ou S2_GAP_P dans src/utils.py, "
              "puis relancer exp006 (blend de référence) avec les valeurs corrigées.")
    print("=" * 70)

    out = MODELS_DIR / "exp010_gap_diagnostic_report.csv"
    df.to_csv(out, index=False)
    print(f"\nRapport détaillé écrit : {out}")

    # ------------------------------------------------------------- figures (best-effort)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        len_counts.plot(kind="bar", ax=ax)
        ax.set_title("Longueur de fenêtre observée (test)")
        ax.set_xlabel("nombre de mois")
        fig.savefig(FIGURES_DIR / "exp010_window_length_hist.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots()
        gap_p_empirical.plot(kind="bar", ax=ax)
        ax.set_title("Nombre de trous S2 observé (test)")
        ax.set_xlabel("n_gaps")
        fig.savefig(FIGURES_DIR / "exp010_n_gaps_hist.png", dpi=120)
        plt.close(fig)
        print(f"Figures écrites dans {FIGURES_DIR}")
    except ImportError:
        print("matplotlib indisponible -- figures non générées (rapport CSV suffit).")


if __name__ == "__main__":
    main()

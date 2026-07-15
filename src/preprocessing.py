"""
preprocessing.py
-----------------
Chargement brut des CSV Zindi et conversion en "cube" numpy (n_echantillons, 12 mois, 12 bandes).

C'est la SEULE fonction qui doit lire les CSV depuis disque : toutes les étapes suivantes
(augmentation, features, EDA) travaillent sur le cube ou sur des DataFrames dérivés, jamais
directement sur les colonnes brutes "{bande}_{mois}". Ça évite de dupliquer le parsing partout.
"""
import numpy as np
import pandas as pd

from src.utils import BANDS, MONTHS, NA_VALUE, TRAIN_PATH, TEST_PATH


def load_raw(path):
    """Charge un CSV Zindi tel quel (sans transformation)."""
    return pd.read_csv(path)


def to_cube(df):
    """
    Transforme un DataFrame large (colonnes "{bande}_{mois}") en cube numpy
    de forme (n, 12, 12) = (échantillons, mois, bandes), avec -9999 -> NaN.
    """
    cube = np.stack(
        [np.stack([df[f"{b}_{m}"].values for b in BANDS], axis=1) for m in MONTHS],
        axis=1,
    ).astype(np.float32)
    cube[cube == NA_VALUE] = np.nan
    return cube


def load_dataset(path, has_label=True):
    """
    Charge un CSV et retourne (ids, y, cube).
    y est None si has_label=False (cas du test).
    """
    df = load_raw(path)
    ids = df["ID"].values
    y = df["label"].values if has_label else None
    cube = to_cube(df)
    return ids, y, cube


def load_train():
    return load_dataset(TRAIN_PATH, has_label=True)


def load_test():
    return load_dataset(TEST_PATH, has_label=False)


def missingness_report(df):
    """
    Petit utilitaire pour l'EDA : pourcentage de -9999 par colonne bande x mois,
    ainsi qu'un résumé agrégé par bande et par mois. Retourne un dict de DataFrames.
    """
    band_month_cols = [f"{b}_{m}" for b in BANDS for m in MONTHS if f"{b}_{m}" in df.columns]
    is_na = (df[band_month_cols] == NA_VALUE)

    pct_by_col = is_na.mean().rename("pct_missing").sort_values(ascending=False)

    records = []
    for b in BANDS:
        for m in MONTHS:
            col = f"{b}_{m}"
            if col in df.columns:
                records.append({"band": b, "month": m, "pct_missing": is_na[col].mean()})
    long_df = pd.DataFrame(records)

    by_band = long_df.groupby("band")["pct_missing"].mean().sort_values(ascending=False)
    by_month = long_df.groupby("month")["pct_missing"].mean()

    return {
        "by_column": pct_by_col,
        "long": long_df,
        "by_band": by_band,
        "by_month": by_month,
        "overall_pct": is_na.values.mean(),
    }

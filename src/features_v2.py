"""
features_v2.py
---------------
Variante de src/features.py testant l'hypothèse H5 de l'EDA (étape 7) :

    "re2/re3/nir/nira (r>0.99), swir1/swir2 (r=0.977), blue/green (r=0.964) et
    VH_VV_ratio/RVI (r=1.000, doublon strict) sont quasi-redondants -> en garder
    un seul par groupe allège le modèle sans perte de signal."

Ne modifie QUE la liste des séries agrégées dans featurize(). build_series()
reste identique à src/features.py (on ne supprime rien du calcul des indices,
on filtre seulement ce qui est ensuite agrégé en features tabulaires) — ça
évite de dupliquer les formules et garde une seule source de vérité pour les
indices spectraux.

Protocole : comparer le score OOF (cross_validate) de exp002 à celui de
exp001_baseline, même seed / même n_splits / mêmes hyperparamètres. Garder
cette version seulement si le score est égal ou meilleur.
"""
import numpy as np
import pandas as pd

from src.features import build_series, agg  # réutilise les formules, aucune duplication

# --- Groupes redondants identifiés à l'étape 7 (Pearson |r| > 0.95 sur les moyennes)
# On garde un seul représentant par groupe (celui qui apparaît le plus souvent
# en tête des classements d'importance à l'étape 8).
DROP_SERIES = {
    "re2", "re3", "nira",   # redondants avec nir (r > 0.99) -> on garde nir
    "swir2",                 # redondant avec swir1 (r=0.977) -> on garde swir1
    "green",                  # redondant avec blue (r=0.964) -> on garde blue
    "VH_VV_ratio",             # doublon strict de RVI (r=1.000) -> on garde RVI
}


def featurize(cube):
    """Comme src.features.featurize(), mais sans les séries de DROP_SERIES."""
    s = build_series(cube)
    feats = {}
    for name, x in s.items():
        if name in DROP_SERIES:
            continue
        if np.all(np.isnan(x)):
            continue
        for k, v in agg(x).items():
            feats[f"{name}_{k}"] = v
        feats[f"{name}_range"] = feats[f"{name}_max"] - feats[f"{name}_min"]
        feats[f"{name}_iqr"] = feats[f"{name}_p75"] - feats[f"{name}_p25"]
        feats[f"{name}_cv"] = feats[f"{name}_std"] / (np.abs(feats[f"{name}_mean"]) + 1e-6)

    # --- contexte d'observation : identique à features.py (indépendant des bandes retirées)
    feats["n_s1"] = np.sum(~np.isnan(s["VH"]), axis=1).astype(float)
    feats["n_s2"] = np.sum(~np.isnan(s["blue"]), axis=1).astype(float)
    feats["frac_cloud"] = 1 - feats["n_s2"] / np.maximum(feats["n_s1"], 1)

    # --- features "métier" : identique à features.py
    feats["water_frac"] = np.nanmean((s["MNDWI"] > 0).astype(float), axis=1)
    feats["ndvi_low_frac"] = np.nanmean((s["NDVI"] < 0.2).astype(float), axis=1)

    return pd.DataFrame(feats).replace([np.inf, -np.inf], np.nan)


def featurize_many(cubes):
    """Applique featurize() à une liste de cubes et concatène (après augmentation)."""
    return pd.concat([featurize(c) for c in cubes], ignore_index=True)

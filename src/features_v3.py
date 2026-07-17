"""
features_v3.py
---------------
Variante de src/features.py testant l'hypothèse H3 de l'EDA (étapes 6-7) :

    "NDVI, NDWI, MNDWI, NDRE, AWEI ont tous une forte séparabilité (0.90-0.95)
    mais sont corrélés entre eux à 0.95-0.99 -> un seul signal mesuré 5 fois.
    Les réduire à 2 représentants (MNDWI + NDRE, les plus présents en tête des
    3 méthodes d'importance à l'étape 8) devrait alléger le modèle sans perte
    de score."

Contrairement à H5 (features_v2, REJETÉE), on garde ici les BANDES brutes
intactes (re2/re3/nira/swir2/green sont restées utiles) et on ne touche
QU'AUX indices dérivés les plus redondants entre eux.

NDMI, NDWI2, BSI, RVI, VH_VV_diff, VH_VV_ratio sont conservés (séparabilité
faible à modérée à l'étape 6, pas identifiés comme redondants entre eux).

Protocole : identique à exp002 -> comparer sur PLUSIEURS seeds (compare_seeds.py)
avant de conclure, vu le résultat surprenant de H5.
"""
import numpy as np
import pandas as pd

from src.features import build_series, agg  # réutilise les formules, aucune duplication

# --- Indices redondants identifiés étapes 6-7 (Pearson |r| = 0.95-0.99 entre eux)
# On garde seulement MNDWI et NDRE : les 2 indices les plus présents en tête
# des 3 méthodes d'importance à l'étape 8 (NDWI_max, NDVI_min, NDRE_p25/min).
# NDVI et NDRE se recoupent fortement (r=0.975) -> on garde NDRE (formule
# indépendante des bandes red/nir déjà bien couvertes par les bandes brutes).
DROP_SERIES = {"NDVI", "NDWI", "AWEI"}   # on garde MNDWI + NDRE


def featurize(cube):
    """Comme src.features.featurize(), mais sans NDVI/NDWI/AWEI (garde MNDWI+NDRE)."""
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

    # --- contexte d'observation : identique à features.py
    feats["n_s1"] = np.sum(~np.isnan(s["VH"]), axis=1).astype(float)
    feats["n_s2"] = np.sum(~np.isnan(s["blue"]), axis=1).astype(float)
    feats["frac_cloud"] = 1 - feats["n_s2"] / np.maximum(feats["n_s1"], 1)

    # --- features "métier" : identique à features.py (basées sur MNDWI/NDVI,
    # calculées directement depuis s, indépendamment de DROP_SERIES ci-dessus)
    feats["water_frac"] = np.nanmean((s["MNDWI"] > 0).astype(float), axis=1)
    feats["ndvi_low_frac"] = np.nanmean((s["NDVI"] < 0.2).astype(float), axis=1)

    return pd.DataFrame(feats).replace([np.inf, -np.inf], np.nan)


def featurize_many(cubes):
    """Applique featurize() à une liste de cubes et concatène (après augmentation)."""
    return pd.concat([featurize(c) for c in cubes], ignore_index=True)

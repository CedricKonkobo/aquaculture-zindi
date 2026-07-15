"""
features.py
------------
Ingénierie de features à partir du cube (n, 12 mois, n_bands).

Deux couches :
1) build_series()  : dérive des séries temporelles (une valeur par mois) à partir des bandes
                      brutes -> indices spectraux (NDVI, NDWI, MNDWI, ...) et combinaisons S1.
2) featurize()      : agrège chaque série sur les mois VALIDES uniquement (mean/std/min/max/
                      median/p25/p75 + range/iqr/cv), ce qui rend les features indépendantes
                      du nombre de mois observés ou de leur position dans l'année.

Toute nouvelle feature doit être ajoutée ici et nulle part ailleurs (pas de duplication
entre EDA, train et inference : tout le monde importe featurize()).
"""
import numpy as np
import pandas as pd

from src.utils import BANDS, safe_ratio


def build_series(cube):
    """Construit un dict {nom_serie: array (n, 12)} à partir du cube brut."""
    b = {name: cube[:, :, i] for i, name in enumerate(BANDS)}
    s = dict(b)

    # --- S1 (radar), quasi toujours disponible, peu sensible aux nuages
    s["VH_VV_diff"] = b["VH"] - b["VV"]
    s["VH_VV_ratio"] = safe_ratio(b["VH"], b["VV"])
    s["RVI"] = 4 * b["VH"] / np.where(np.abs(b["VH"] + b["VV"]) < 1e-6, np.nan, b["VH"] + b["VV"])

    # --- S2 (optique) : indices normalisés -> robustes au changement de période/saison
    s["NDVI"] = safe_ratio(b["nir"], b["red"])          # végétation
    s["NDWI"] = safe_ratio(b["green"], b["nir"])         # eau libre
    s["MNDWI"] = safe_ratio(b["green"], b["swir1"])      # eau vs bâti -> clé pour les bassins
    s["NDMI"] = safe_ratio(b["nir"], b["swir1"])         # humidité
    s["NDWI2"] = safe_ratio(b["nir"], b["swir2"])
    s["AWEI"] = 4 * (b["green"] - b["swir1"]) - (0.25 * b["nir"] + 2.75 * b["swir2"])
    s["NDRE"] = safe_ratio(b["nira"], b["re1"])          # red-edge
    s["BSI"] = safe_ratio(b["swir1"] + b["red"], b["nir"] + b["blue"])  # sol nu
    s["brightness"] = np.nanmean(
        np.stack([b[x] for x in ["blue", "green", "red", "nir"]], -1), axis=-1
    )
    return s


def agg(x):
    """Statistiques calculées sur les mois valides uniquement (ignore les NaN)."""
    with np.errstate(all="ignore"):
        return {
            "mean": np.nanmean(x, 1), "std": np.nanstd(x, 1),
            "min": np.nanmin(x, 1), "max": np.nanmax(x, 1),
            "med": np.nanmedian(x, 1),
            "p25": np.nanpercentile(x, 25, axis=1), "p75": np.nanpercentile(x, 75, axis=1),
        }


def featurize(cube):
    """Transforme un cube (n, 12, n_bands) en DataFrame de features tabulaires."""
    s = build_series(cube)
    feats = {}
    for name, x in s.items():
        if np.all(np.isnan(x)):
            continue
        for k, v in agg(x).items():
            feats[f"{name}_{k}"] = v
        feats[f"{name}_range"] = feats[f"{name}_max"] - feats[f"{name}_min"]
        feats[f"{name}_iqr"] = feats[f"{name}_p75"] - feats[f"{name}_p25"]
        feats[f"{name}_cv"] = feats[f"{name}_std"] / (np.abs(feats[f"{name}_mean"]) + 1e-6)

    # --- contexte d'observation : le modèle doit savoir ce qu'il voit (combien de mois, etc.)
    feats["n_s1"] = np.sum(~np.isnan(s["VH"]), axis=1).astype(float)
    feats["n_s2"] = np.sum(~np.isnan(s["blue"]), axis=1).astype(float)
    feats["frac_cloud"] = 1 - feats["n_s2"] / np.maximum(feats["n_s1"], 1)

    # --- features "métier" dérivées des indices d'eau/végétation
    feats["water_frac"] = np.nanmean((s["MNDWI"] > 0).astype(float), axis=1)
    feats["ndvi_low_frac"] = np.nanmean((s["NDVI"] < 0.2).astype(float), axis=1)

    return pd.DataFrame(feats).replace([np.inf, -np.inf], np.nan)


def featurize_many(cubes):
    """Applique featurize() à une liste de cubes et concatène le résultat (utilisé après augmentation)."""
    return pd.concat([featurize(c) for c in cubes], ignore_index=True)

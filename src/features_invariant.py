"""
features_invariant.py
----------------------
Features INVARIANTES INTRA-ÉCHANTILLON, conçues pour survivre au changement
de période temporelle entre train et test.

Motivation (exp011 -> exp014) :
  - AUC adversariale = 0.99 : train et test sont quasi disjoints.
  - exp012 (LB 0.778) : aligner les distributions PAR DOMAINE écrase de
    l'information vraie -> échec.
  - exp014 : supprimer les features décalées dégrade de façon monotone
    (0.9472 -> 0.9333) : elles sont décalées ET indispensables.
  - Conclusion : l'AUC test (~0.89 estimée) contre 0.994 en OOF traduit un
    échec de GÉNÉRALISATION, pas un problème de seuil ni de features
    instables.

Idée directrice : toutes les features de src/features.py comparent un
échantillon aux AUTRES échantillons, donc à la distribution de sa période.
Ici on décrit la FORME INTERNE de chaque échantillon :
  - le rang des bandes les unes par rapport aux autres, dans le MÊME pixel ;
  - le rang de chaque mois dans sa PROPRE série temporelle ;
  - des contrastes normalisés par les statistiques du pixel lui-même.

Un bassin aquacole est de l'eau : NIR et SWIR bas, bleu/vert relativement
hauts. Cet ORDRE survit à un changement de saison, d'atmosphère ou de
calibration capteur -- là où les valeurs absolues, et même les indices
normalisés, ne survivent pas (cf. exp011 : NDWI/NDVI/BSI dominent le top des
features décalées).

Différence essentielle avec exp012 : on ne normalise PAS par domaine (ce qui
forçait les marginales train/test à coïncider et détruisait du signal). On
normalise PAR ÉCHANTILLON, indépendamment de tout autre échantillon. Le
calcul d'un échantillon ne dépend donc d'aucune population de référence.
"""
import numpy as np
import pandas as pd

from src.utils import BANDS, S2_BANDS
from src.features import build_series

# Ordre spectral de référence pour le classement intra-pixel.
SPECTRAL_ORDER = ["blue", "green", "red", "re1", "re2", "re3", "nir", "nira",
                   "swir1", "swir2"]


def _nanrank_along_axis(a, axis):
    """
    Rang normalisé dans [0, 1] le long d'un axe, en ignorant les NaN.
    Les NaN restent NaN. Implémentation vectorisée via argsort.
    """
    a = np.asarray(a, dtype=float)
    n_valid = np.sum(~np.isnan(a), axis=axis, keepdims=True)

    order = np.argsort(np.where(np.isnan(a), np.inf, a), axis=axis)
    ranks = np.empty_like(order, dtype=float)
    idx = np.arange(a.shape[axis])
    shape = [1] * a.ndim
    shape[axis] = -1
    np.put_along_axis(ranks, order, np.broadcast_to(idx.reshape(shape), a.shape).copy(),
                       axis=axis)

    with np.errstate(invalid="ignore", divide="ignore"):
        ranks = ranks / np.maximum(n_valid - 1, 1)
    ranks[np.isnan(a)] = np.nan
    return ranks


def spectral_shape_features(cube):
    """
    Forme spectrale intra-pixel : pour chaque mois, on classe les bandes
    optiques les unes par rapport aux autres, puis on agrège sur les mois.

    Ne dépend que du pixel lui-même -> insensible à un décalage global de
    niveau ou d'échelle entre périodes.
    """
    band_idx = [BANDS.index(b) for b in SPECTRAL_ORDER]
    sub = cube[:, :, band_idx]                       # (n, 12, n_spectral)
    ranks = _nanrank_along_axis(sub, axis=2)         # rang de chaque bande, par mois

    out = {}
    for j, name in enumerate(SPECTRAL_ORDER):
        r = ranks[:, :, j]
        with np.errstate(invalid="ignore"):
            out[f"srank_{name}_mean"] = np.nanmean(r, axis=1)
            out[f"srank_{name}_std"] = np.nanstd(r, axis=1)
    return out


def temporal_shape_features(series):
    """
    Forme temporelle intra-série : position relative de chaque mois dans la
    propre série de l'échantillon.

    On extrait des descripteurs de forme (et non de niveau) : où se situe le
    pic, quelle est l'amplitude relative, quelle est la régularité.
    """
    out = {}
    for name, arr in series.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            mn = np.nanmin(arr, axis=1)
            mx = np.nanmax(arr, axis=1)
            med = np.nanmedian(arr, axis=1)
            rng = mx - mn

            # position relative de la médiane dans l'étendue : forme de la
            # distribution temporelle, sans dépendre du niveau absolu
            out[f"tshape_{name}_medpos"] = np.where(rng > 1e-9, (med - mn) / rng, np.nan)

            # position temporelle (normalisée) du min et du max
            valid = ~np.isnan(arr)
            n_valid = valid.sum(axis=1)
            filled_max = np.where(valid, arr, -np.inf)
            filled_min = np.where(valid, arr, np.inf)
            argmax = filled_max.argmax(axis=1).astype(float)
            argmin = filled_min.argmin(axis=1).astype(float)
            denom = np.maximum(arr.shape[1] - 1, 1)
            out[f"tshape_{name}_argmax"] = np.where(n_valid > 0, argmax / denom, np.nan)
            out[f"tshape_{name}_argmin"] = np.where(n_valid > 0, argmin / denom, np.nan)

            # variation moyenne absolue rapportée à l'étendue : régularité de
            # la série, indépendante de son amplitude
            diffs = np.abs(np.diff(arr, axis=1))
            mad = np.nanmean(diffs, axis=1)
            out[f"tshape_{name}_smooth"] = np.where(rng > 1e-9, mad / rng, np.nan)
    return out


def self_normalized_contrasts(series):
    """
    Contrastes normalisés par les statistiques du pixel lui-même
    (z-score intra-échantillon), et non par la population.
    """
    out = {}
    for name, arr in series.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.nanmean(arr, axis=1)
            sd = np.nanstd(arr, axis=1)
            mn = np.nanmin(arr, axis=1)
            mx = np.nanmax(arr, axis=1)
            # amplitude rapportée au niveau moyen : "de combien ça varie
            # RELATIVEMENT à soi-même"
            out[f"selfz_{name}_range_ratio"] = np.where(
                np.abs(mu) > 1e-9, (mx - mn) / np.abs(mu), np.nan)
            out[f"selfz_{name}_cv_self"] = np.where(
                np.abs(mu) > 1e-9, sd / np.abs(mu), np.nan)
    return out


def featurize_invariant(cube, include_base_shape=True):
    """
    Construit le DataFrame des features invariantes intra-échantillon.

    Aucune statistique de population n'est utilisée : le calcul d'un
    échantillon ne dépend d'aucun autre échantillon, ni du train, ni du test.
    """
    series = build_series(cube)

    feats = {}
    if include_base_shape:
        feats.update(spectral_shape_features(cube))
    feats.update(temporal_shape_features(series))
    feats.update(self_normalized_contrasts(series))

    df = pd.DataFrame(feats)
    return df.replace([np.inf, -np.inf], np.nan)


def featurize_invariant_many(cubes):
    """Version multi-répliques (augmentation), aligne les colonnes."""
    parts = [featurize_invariant(c) for c in cubes]
    cols = parts[0].columns
    return pd.concat([p[cols] for p in parts], ignore_index=True)

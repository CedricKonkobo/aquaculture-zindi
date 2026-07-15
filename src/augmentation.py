"""
augmentation.py
----------------
Augmentation par masquage temporel.

Contexte du problème (voir notebook 03_Baseline_Analysis pour le détail) :
- Le train contient 12 mois complets par échantillon.
- Le test ne contient qu'une fenêtre CONSÉCUTIVE de 4 à 6 mois, avec en plus des trous
  S2 dus aux nuages (le radar S1 est presque toujours disponible, l'optique S2 non).
- Un modèle entraîné sur 12 mois complets verrait donc une distribution de features
  différente à l'inférence (plus de NaN, moins de mois) -> risque de décalage train/test.

Solution : on simule artificiellement, sur le train, les mêmes conditions que le test
(mêmes longueurs de fenêtre, même distribution de trous nuageux) avant de calculer les
features. Le modèle apprend alors directement sur la distribution qu'il verra en prod.
"""
import numpy as np

from src.utils import S2_GAP_P, S1_BANDS, get_rng


def mask_window(cube, rng, min_len=4, max_len=6, gap_p=S2_GAP_P):
    """
    Applique, pour chaque échantillon du cube (n, 12, n_bands) :
      1) une fenêtre consécutive aléatoire de `min_len` à `max_len` mois (les autres mois -> NaN),
      2) un nombre de trous S2 tiré selon `gap_p` (distribution empirique observée dans le test),
         appliqués uniquement aux bandes optiques (S2), le radar (S1) restant présent.

    Retourne un nouveau cube (copie), le cube d'entrée n'est jamais modifié en place.
    """
    n, n_months, n_bands = cube.shape
    out = np.full_like(cube, np.nan)
    n_s1 = len(S1_BANDS)

    for i in range(n):
        length = rng.integers(min_len, max_len + 1)
        start = rng.integers(0, n_months - length + 1)
        out[i, start:start + length] = cube[i, start:start + length]

        n_gaps = rng.choice(len(gap_p), p=gap_p)
        if n_gaps > 0:
            drop_months = rng.choice(np.arange(start, start + length),
                                      size=min(n_gaps, length), replace=False)
            out[i, drop_months, n_s1:] = np.nan  # on ne retire que les bandes S2
    return out


def build_augmented_replicas(cube, y, n_aug, seed):
    """
    Génère `n_aug` répliques masquées du cube d'entraînement.

    Retourne (cubes, ys, groups) :
    - cubes : liste de `n_aug` cubes masqués
    - ys    : liste de `n_aug` copies de y
    - groups: liste de `n_aug` tableaux d'indices d'origine (0..n-1), pour un GroupKFold
              qui empêche deux répliques du MÊME échantillon d'être dans des folds différents.
    """
    cubes, ys, groups = [], [], []
    for r in range(n_aug):
        rr = get_rng(seed + r)
        cubes.append(mask_window(cube, rr))
        ys.append(y)
        groups.append(np.arange(len(y)))
    return cubes, ys, groups

# exp001_baseline — Résultats

## Contexte
Première baseline fonctionnelle. Objectif : établir une référence solide et
robuste au décalage temporel train/test avant toute optimisation.

## Configuration
Voir `config.yaml`. Résumé : LightGBM + augmentation par masquage (10 répliques)
+ features statistiques invariantes à la fenêtre (mean/std/min/max/med/p25/p75
sur indices spectraux et bandes brutes).

## Résultats

| Métrique | Valeur (OOF) |
|---|---|
| AUC |0.9940 |
| F1 (seuil 0.5) | 0.9618 |
| Score pondéré (0.6·F1 + 0.4·AUC) | 0.9747 |
| Score leaderboard Zindi | ~100e place (baseline d'origine) |

## Top features (importance LightGBM)
VV_min       208
NDWI_max     185
green_min    137
VV_p75       123
VV_mean      117
NDRE_min     116
NDMI_max     115
VV_max       115
blue_min     102
swir2_max     96
MNDWI_min     94
NDWI2_max     91
green_p25     82
BSI_max       82
red_min       79

## Observations
-

## Limites identifiées
- Seuil de décision fixé arbitrairement à 0.5 (à optimiser, voir Étape 6).
- Un seul modèle testé (LightGBM) — pas de comparaison avec CatBoost/XGBoost/RF/MLP.
- Pas d'ablation feature par feature : impossible de savoir laquelle apporte
  réellement du signal vs. bruit.
- L'augmentation masque une fenêtre uniforme aléatoire ; la vraie distribution
  saisonnière des fenêtres de test n'a pas été vérifiée (à confirmer en EDA).

## Prochaine étape
→ EDA complète (notebook `01_EDA.ipynb`) pour identifier les pistes
  d'amélioration factuelles avant de créer exp002.

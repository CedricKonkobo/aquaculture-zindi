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
| AUC | _à remplir après exécution_ |
| F1 (seuil 0.5) | _à remplir après exécution_ |
| Score pondéré (0.6·F1 + 0.4·AUC) | _à remplir après exécution_ |
| Score leaderboard Zindi | ~100e place (baseline d'origine) |

## Top features (importance LightGBM)
_à remplir après exécution — copier la sortie de `feature_importance()`_

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

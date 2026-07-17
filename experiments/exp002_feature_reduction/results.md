# exp002_feature_reduction — Résultats

## Hypothèse testée (H5, EDA étape 7)
Les bandes/indices suivants sont quasi-redondants (Pearson |r| > 0.95 sur les
moyennes) : `re2/re3/nira` avec `nir`, `swir2` avec `swir1`, `green` avec
`blue`, `VH_VV_ratio` avec `RVI` (doublon strict, r=1.000). Les retirer des
features agrégées devrait alléger le modèle sans faire baisser le score.

## Changement vs exp001_baseline
Uniquement `src/features.py` → `src/features_v2.py` (voir `DROP_SERIES`).
Tout le reste (hyperparamètres, augmentation, CV, seed) est identique.

## Résultats — multi-seed (compare_seeds.py, seeds 42/1/7)

| Métrique | exp001_baseline (v1, 245 feat.) | exp002 (v2, 185 feat.) | Delta |
|---|---|---|---|
| Nb features | 245 | 185 | -60 |
| AUC (OOF, moy. 3 seeds) | 0.9939 | 0.9936 | -0.0004 |
| F1 (OOF, moy. 3 seeds) | 0.9628 | 0.9617 | -0.0011 |
| Score pondéré (moy. 3 seeds) | 0.9752 | 0.9744 | **-0.0008** |
| Score pondéré — std du delta | — | — | 0.0002 |

Détail par seed :

| seed | score_v1 | score_v2 | delta |
|---|---|---|---|
| 42 | 0.9747 | 0.9739 | -0.0007 |
| 1  | 0.9751 | 0.9745 | -0.0007 |
| 7  | 0.9759 | 0.9749 | -0.0010 |

## Décision
**H5 REJETÉE — on garde `src/features.py` (v1, 245 features) en référence.**

Application stricte de la règle définie dans `config.yaml` :
`|delta moyen| (0.0008) > delta std (0.0002)`, et delta < 0 sur les 3 seeds
sans exception → l'effet n'est pas du bruit de CV, la perte est réelle.

## Observations
- Les features supprimées (re2, re3, nira, swir2, green, VH_VV_ratio) sont
  fortement corrélées à leur représentant conservé (nir, swir1, blue, RVI)
  en **moyenne linéaire (Pearson)**, mais LightGBM en tirait apparemment un
  signal résiduel non redondant (relation non-linéaire, ou interaction avec
  d'autres features) — une forte corrélation Pearson sur les `_mean` ne
  garantit pas une redondance totale sur toutes les statistiques dérivées
  (`_std`, `_p25`, `_cv`, etc.), qui elles n'ont pas été testées pour
  corrélation.
- L'effet est petit en absolu (~0.08% relatif) : peu susceptible de changer
  le classement Zindi, mais bien réel et reproductible sur 3 seeds distincts.
- Deux `RuntimeWarning: Mean of empty slice` apparaissent à chaque run
  (`features.py:41`, calcul de `brightness`) : au moins un mois a les 4
  bandes optiques (blue/green/red/nir) simultanément NaN dans certaines
  répliques augmentées. Sans conséquence sur le résultat (nanmean gère le
  cas), mais à nettoyer (garde silencieuse ou message explicite) pour ne pas
  polluer les logs des futures expériences.

## Limites / à vérifier avant de conclure définitivement
- Uniquement testé avec LightGBM — un modèle plus sensible à la
  colinéarité (ex. régression logistique, MLP) pourrait réagir différemment
  à cette réduction ; à garder en tête à l'Étape 5 (comparaison de modèles).
- H5 rejetée pour la features finale, mais `features_v2.py` reste utile
  comme variante "légère" pour accélérer l'exploration Optuna (Étape 6) —
  le coût de -0.0008 est probablement acceptable pendant une recherche
  d'hyperparamètres, à réévaluer avec la config optimale trouvée sur v1.

## Prochaine étape
→ Revenir à `src/features.py` (v1) comme référence pour exp003.
→ Conserver `features_v2.py` de côté, à ré-essayer une fois qu'Optuna aura
  déterminé si un modèle moins gourmand en interactions non-linéaires
  (ex. moins de `num_leaves`) referme l'écart avec v1.
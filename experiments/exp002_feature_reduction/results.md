# exp002_feature_reduction — Résultats

## Hypothèse testée (H5, EDA étape 7)
Les bandes/indices suivants sont quasi-redondants (Pearson |r| > 0.95 sur les
moyennes) : `re2/re3/nira` avec `nir`, `swir2` avec `swir1`, `green` avec
`blue`, `VH_VV_ratio` avec `RVI` (doublon strict, r=1.000). Les retirer des
features agrégées devrait alléger le modèle sans faire baisser le score.

## Changement vs exp001_baseline
Uniquement `src/features.py` → `src/features_v2.py` (voir `DROP_SERIES`).
Tout le reste (hyperparamètres, augmentation, CV, seed) est identique.

## Résultats

Run initial (seed 42) :

| Métrique | exp001_baseline | exp002 (features_v2) | Delta |
|---|---|---|---|
| Nb features | 245 | 185 | -60 |
| AUC (OOF) | 0.9940 | 0.9936 | -0.0004 |
| F1 (OOF, seuil 0.5) | 0.9618 | 0.9608 | -0.0010 |
| Score pondéré | 0.9747 | 0.9739 | -0.0007 |

Confirmation multi-seed (`compare_seeds.py`, seeds 42 / 1 / 7) :

| Seed | AUC v1 | AUC v2 | F1 v1 | F1 v2 | Score v1 | Score v2 | Delta |
|---|---|---|---|---|---|---|---|
| 42 | 0.9940 | 0.9936 | 0.9618 | 0.9608 | 0.9747 | 0.9739 | -0.0007 |
| 1  | 0.9938 | 0.9933 | 0.9627 | 0.9620 | 0.9751 | 0.9745 | -0.0007 |
| 7  | 0.9940 | 0.9938 | 0.9638 | 0.9623 | 0.9759 | 0.9749 | -0.0010 |

**Delta moyen : -0.0008 — Delta std : 0.0002**

Règle de décision (`|delta moyen| < delta std` → bruit) : **0.0008 > 0.0002 → non vérifiée.**
Delta négatif sur les 3 seeds, sans exception → perte cohérente, pas du bruit de CV.

## Décision
- [ ] Gain net → fusionner `features_v2.py` dans `features.py`
- [ ] Neutre → garder pour la légèreté/interprétabilité
- [x] **Perte → rejeter, hypothèse infirmée**

**Action :** `features_v2.py` abandonné. `features.py` (245 features) reste la référence pour la suite.

## Observations
- La redondance statistique (Pearson |r| > 0.95) ne garantit pas une redondance
  informationnelle totale pour LightGBM : re2/re3/nira, swir2, green et
  VH_VV_ratio portent chacun un petit signal résiduel exploitable par les
  splits d'arbres, malgré leur forte corrélation avec leur représentant conservé.
- Le protocole multi-seed était nécessaire : sur seed 42 seule, le delta
  (-0.0007) aurait pu sembler négligeable et faire garder v2 à tort. La
  cohérence du signe sur 3 seeds est ce qui a permis de trancher correctement.

## Limites / à vérifier avant de conclure
- Testé sur 3 seeds seulement — la perte est faible en valeur absolue (~0.08
  point de score), donc reste à surveiller si elle se cumule avec d'autres
  simplifications de features plus tard.
- Le choix du représentant gardé par groupe (nir, swir1, blue, RVI) était basé
  sur l'étape 8 (le plus présent en tête des 3 méthodes d'importance) — non
  retesté avec un autre représentant, puisque l'hypothèse est déjà rejetée.
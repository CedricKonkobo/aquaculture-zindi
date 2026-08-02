# exp009_hybrid_blend — Résultats

## Contexte
exp008 a montré qu'Optuna n'améliore QUE CatBoost (+0.0008 vs manuel) ; le
tuning LightGBM (-0.0009) et XGBoost (-0.0008) est une perte. Cette
expérience teste le blend "meilleur des deux mondes" : LightGBM manuel +
XGBoost manuel + CatBoost tuné (poids optimisés), contre exp006
(0.9756, 3 modèles manuels, équipoids).

## Résultats seed=42 (`run.py`)

| Modèle | AUC | F1 | Score |
|---|---|---|---|
| lightgbm_manual | _à remplir_ | _à remplir_ | _à remplir_ |
| xgboost_manual | _à remplir_ | _à remplir_ | _à remplir_ |
| catboost_manual | _à remplir_ | _à remplir_ | _à remplir_ |
| catboost_tuned | _à remplir_ | _à remplir_ | _à remplir_ |

| Blend | Score |
|---|---|
| all_manual (= exp006, ce seed) | _à remplir_ |
| hybrid_equal | _à remplir_ |
| **hybrid_weighted** | _à remplir_ |

Poids optimaux : _à remplir_

**hybrid_weighted vs exp006 (0.9756) : delta = _à remplir_**

## Confirmation multi-seed (`compare_seeds.py`, seeds 42/1/7)

| Seed | all_manual | hybrid_weighted | Delta |
|---|---|---|---|
| 42 | _à remplir_ | _à remplir_ | _à remplir_ |
| 1  | _à remplir_ | _à remplir_ | _à remplir_ |
| 7  | _à remplir_ | _à remplir_ | _à remplir_ |

**Delta moyen : _à remplir_ — Delta std : _à remplir_**

Règle de décision (`|delta moyen| < delta std` → bruit) : _à remplir_

## Décision
- [ ] Gain net cohérent → adopter exp009 (hybrid_weighted) comme référence
      finale pour la soumission
- [ ] Bruit (delta non significatif) → garder exp006 (plus simple, moins
      de risque d'overfit aux OOF)

## Observations
-

## Limites / à vérifier avant de conclure
- Les poids optimisés sont choisis sur les mêmes OOF qui servent à les
  évaluer (même limite que exp008) — à surveiller sur le score Zindi réel.
- Le grid search ne considère que 3 poids sur un pas de 0.05 — suffisant
  ici (3 modèles), mais ne scale pas si on ajoute d'autres variantes de
  modèles au blend.
- Si le gain se confirme mais reste modeste (< 0.001-0.002), envisager un
  stacking (régression logistique sur les OOF) comme prochaine étape avant
  d'explorer une direction différente (feature engineering additif, ou
  vérification de gap_p vs distribution réelle du test).

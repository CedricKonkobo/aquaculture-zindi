# exp008_optuna_lgbm_catboost — Résultats

## Objectif
Compléter l'optimisation Optuna commencée à exp007 (XGBoost) sur les deux
autres boostings du blend, puis reconstruire le blend d'exp006 avec les
3 modèles tunés et des poids optimisés (au lieu de la moyenne simple).

## Partie 1 — LightGBM tuné (phase 2, config complète)

| | AUC | F1 | Score |
|---|---|---|---|
| LightGBM manuel (exp005, référence) | - | - | 0.9750 |
| LightGBM Optuna (config complète) | _à remplir_ | _à remplir_ | _à remplir_ |

**Delta : _à remplir_**

Meilleurs hyperparamètres :
```
_à remplir (coller la sortie console)_
```

## Partie 2 — CatBoost tuné (phase 2, config complète)

| | AUC | F1 | Score |
|---|---|---|---|
| CatBoost manuel (exp005, référence) | - | - | 0.9741 |
| CatBoost Optuna (config complète) | _à remplir_ | _à remplir_ | _à remplir_ |

**Delta : _à remplir_**

Meilleurs hyperparamètres :
```
_à remplir (coller la sortie console)_
```

## Partie 3 — Blend à 3 modèles tunés (mêmes folds)

| Modèle | AUC | F1 | Score |
|---|---|---|---|
| lightgbm_tuned | _à remplir_ | _à remplir_ | _à remplir_ |
| xgboost_tuned (exp007) | _à remplir_ | _à remplir_ | _à remplir_ |
| catboost_tuned | _à remplir_ | _à remplir_ | _à remplir_ |
| **Blend équipoids** | _à remplir_ | _à remplir_ | _à remplir_ |
| **Blend poids optimisés (OOF)** | - | - | _à remplir_ |

Poids optimaux trouvés (grid search, pas 0.05) :
```
_à remplir_
```

**Blend poids optimisés vs exp006 (0.9756, blend équipoids modèles manuels) : delta = _à remplir_**
**Blend poids optimisés vs blend équipoids (3 modèles tunés) : delta = _à remplir_**

## Décision
- [ ] LightGBM tuné : gain net → l'adopter dans le blend
- [ ] LightGBM tuné : neutre/perte → garder la version manuelle d'exp005 dans le blend
- [ ] CatBoost tuné : gain net → l'adopter dans le blend
- [ ] CatBoost tuné : neutre/perte → garder la version manuelle d'exp005 dans le blend
- [ ] Poids optimisés : gain net vs équipoids (> bruit CV ~0.0002-0.0003) → adopter comme
      candidat de référence pour la soumission finale
- [ ] Poids optimisés : neutre → garder le blend équipoids (plus simple, moins de risque
      d'overfit aux OOF), mais avec les modèles individuellement gagnants tunés

## Observations
-

## Limites / à vérifier avant de conclure
- Un seul seed pour le blend final — si le gain (tuning individuel ou poids
  optimisés) est proche de 0.001-0.002, confirmer sur 2-3 seeds avant de
  considérer ce blend comme définitif (même logique que H3/H5/exp002/exp007).
- Les poids optimisés sont choisis sur les mêmes OOF qui servent à les
  évaluer (pas de split train/validation séparé pour le choix des poids) —
  léger risque d'optimisme sur le score annoncé. À surveiller si le score
  leaderboard Zindi confirmé est nettement en dessous du score OOF.
- Le grid search sur le simplexe (pas=0.05) n'explore pas les poids négatifs
  ni > 1 (stacking généralisé) — une régression logistique sur les OOF
  (stacking, cf. suggestion précédente) resterait la prochaine étape si ce
  blend pondéré déçoit.
- Si exp007_optuna_best_params.csv n'existait pas au moment de l'exécution,
  le blend utilise le XGBoost manuel d'exp005 à la place du XGBoost tuné —
  vérifier dans la sortie console laquelle des deux versions a été utilisée.

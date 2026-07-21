# exp006_ensemble_blend — Résultats

## Hypothèse testée
Les 3 boostings (exp005) ont des scores OOF proches (0.9741-0.9754) mais
probablement des erreurs différentes -> un blend (moyenne simple, mêmes
folds) devrait dépasser le meilleur modèle individuel (XGBoost, 0.9754).

## Résultats (`run.py`, seed 42, mêmes folds pour les 3 modèles)

| Modèle | AUC (OOF) | F1 (OOF) | Score (OOF) |
|---|---|---|---|
| LightGBM | _à remplir_ | _à remplir_ | _à remplir_ |
| XGBoost | _à remplir_ | _à remplir_ | _à remplir_ |
| CatBoost | _à remplir_ | _à remplir_ | _à remplir_ |
| **Blend** | _à remplir_ | _à remplir_ | _à remplir_ |

**Blend vs meilleur modèle seul : delta = _à remplir_**

## Décision
- [ ] Blend meilleur → l'adopter comme candidat pour la soumission finale
- [ ] Blend neutre/pire → revenir au meilleur modèle seul (XGBoost)
- [ ] Un modèle plombe le blend → tester un blend à 2 (lightgbm + xgboost)

## Observations
-

## Limites / à vérifier avant de conclure
- Un seul seed — confirmer sur 2-3 seeds si le blend est proche (< 0.001-0.002)
  du meilleur modèle seul, même logique que H3/H5/exp005.
- Blend à poids égaux testé ici ; un blend pondéré (poids proportionnel au
  score OOF individuel) est une variante possible si le blend simple déçoit.

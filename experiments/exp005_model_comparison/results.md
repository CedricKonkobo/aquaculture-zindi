# exp005_model_comparison — Résultats

## Objectif (étape 5 du plan)
Comparer LightGBM, CatBoost, XGBoost, RandomForest, ExtraTrees sur exactement
la même CV (features.py référence, augmentation identique, GroupKFold,
seuil 0.5, scale_pos_weight=1.6 retenu par H2), avant de passer à
l'optimisation d'hyperparamètres (Optuna, étape 6).

## Résultats (`run.py`, seed 42)

| Modèle | AUC (OOF) | F1 (OOF) | Score (OOF) | Taux positifs (test) |
|---|---|---|---|---|
| LightGBM | _à remplir_ | _à remplir_ | _à remplir_ | _à remplir_ |
| CatBoost | _à remplir_ | _à remplir_ | _à remplir_ | _à remplir_ |
| XGBoost | _à remplir_ | _à remplir_ | _à remplir_ | _à remplir_ |
| RandomForest | _à remplir_ | _à remplir_ | _à remplir_ | _à remplir_ |
| ExtraTrees | _à remplir_ | _à remplir_ | _à remplir_ | _à remplir_ |

## Décision
- Modèle retenu pour l'étape 6 (Optuna) : _à remplir_
- Écart avec le 2e meilleur : _à remplir_ (>0.001 → choix net / <0.001 → envisager un blend)

## Observations
-

## Limites / à vérifier avant de conclure
- Un seul seed : comme pour H3/H5/H2, un écart < ~0.001-0.002 entre deux
  modèles peut être du bruit de CV — à confirmer sur 2-3 seeds si les deux
  meilleurs modèles sont proches avant de trancher définitivement.
- RandomForest/ExtraTrees utilisent une imputation médiane (pas de NaN natif),
  contrairement aux 3 autres modèles — désavantage structurel possible, à
  garder en tête si leur score est nettement plus faible.
- Pas d'early stopping pour RandomForest/ExtraTrees (n_estimators fixe à 500) :
  comparaison pas parfaitement à isoefforts avec les boostings.

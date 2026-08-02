# exp007_optuna_xgboost — Résultats

## Objectif
Optimiser les hyperparamètres XGBoost via Optuna (TPE, 40 essais par défaut),
comparé au meilleur XGBoost manuel de exp005 (score OOF 0.9754).

## Protocole
- Phase 1 (recherche) : proxy rapide, n_aug=5, CV à 3 folds, 40 essais.
- Phase 2 (confirmation) : meilleurs hyperparamètres réévalués en config
  complète (n_aug=10, 5 folds) -- seule cette valeur est comparable à 0.9754.

## Meilleurs hyperparamètres trouvés
```
_à remplir (coller la sortie "Meilleurs hyperparamètres" de la console)_
```

## Résultat (phase 2, config complète)

| | AUC | F1 | Score |
|---|---|---|---|
| XGBoost manuel (exp005, référence) | - | - | 0.9754 |
| XGBoost Optuna (config complète) | _à remplir_ | _à remplir_ | _à remplir_ |

**Delta : _à remplir_**

## Décision
- [ ] Gain net (> ~0.001-0.002) → adopter ces hyperparamètres comme référence XGBoost
- [ ] Neutre/négligeable → garder les hyperparamètres manuels (plus simples, déjà documentés)
- [ ] Perte → rejeter, config manuelle reste la référence

## Observations
-

## Limites / à vérifier avant de conclure
- Le score phase 1 (proxy) n'est PAS comparable à 0.9754 -- seule la phase 2
  (config complète) l'est.
- Un seul seed en phase 2 -- si le gain est proche de 0.001-0.002, confirmer
  sur 2-3 seeds avant de considérer la config Optuna comme définitive.
- scale_pos_weight fixé à 1.6 (H2), non retuné ici -- une interaction entre
  scale_pos_weight et les hyperparamètres XGBoost reste possible mais non
  explorée dans cette expérience.

# exp017_blend_pseudo — Résultats

## Contexte

| Exp | Approche | LB |
|---|---|---|
| exp014 | drop features décalées | 0.825 |
| exp015 | base + invariantes | 0.8324 |
| exp016 | + pseudo-étiquetage (pw=0.5) | **0.8365** |

**Deux gisements attaqués ici :**

**A) `pseudo_weight`** — exp016 a montré que les seuils 0.90 et 0.99 donnent
le **même** holdout (0.9489) malgré 838 vs 499 pseudo-étiquettes. Le gain vient
donc de l'**accès** au domaine cible, pas du **volume** ajouté. L'intensité de
l'adaptation compte probablement plus que le nombre d'échantillons — or ce
paramètre est figé à 0.5 depuis exp016 et n'a jamais été balayé.

**B) Blend multi-familles** — toute la chaîne exp015-016 tourne sur un
**seul** LightGBM. Le dernier blend date d'exp009, sur des features
obsolètes. La diversité a payé deux fois sur ce problème (+0.0073 puis +0.0041).

## Holdout figé — correction méthodologique

À exp016, le holdout n'avait pas la même composition qu'à exp015 (propension
calculée sur 245 puis 409 features), d'où un tour 0 à 0.9454 contre 0.9524
annoncé pour la même configuration. Les deltas restaient valides, les niveaux
absolus non.

**Corrigé** : le holdout est désormais calculé une seule fois sur les features
de base et sauvegardé dans `outputs/models/exp017_fixed_holdout_groups.csv`.
**Toutes les expériences suivantes doivent le recharger tel quel.**

AUC adversariale (features de base) : _à remplir_
Holdout : _à remplir_ lignes / _à remplir_ groupes

## Résultats

Pseudo-étiquettes (seuil 0.90) : _à remplir_ / 1030 — dont _à remplir_ % de positifs

| Config | Modèles | pseudo_weight | Holdout | Holdout AUC | Holdout F1 | CV std | Taux positifs |
|---|---|---|---|---|---|---|---|
| lgbm_nopl | lightgbm | — | | | | | |
| lgbm_pw0p25 | lightgbm | 0.25 | | | | | |
| lgbm_pw0p5 | lightgbm | 0.5 | | | | | |
| lgbm_pw1p0 | lightgbm | 1.0 | | | | | |
| lgbm_pw2p0 | lightgbm | 2.0 | | | | | |
| blend_nopl | lgbm+xgb+cat | — | | | | | |
| blend_pw{best} | lgbm+xgb+cat | _best_ | | | | | |

Meilleur `pseudo_weight` : _à remplir_
Meilleure config globale : _à remplir_ (delta vs lgbm_nopl : _à remplir_)

## Soumission

| Config | Score LB |
|---|---|
| _à remplir_ | _à remplir_ |

Référence : 0.8365 (exp016)

## Décision
- [ ] `pseudo_weight` > 0.5 améliore nettement → l'intensité de l'adaptation
      était sous-exploitée → pousser plus loin (4.0, voire davantage)
- [ ] Le blend améliore → le retenir comme configuration finale + bagging
      multi-seed
- [ ] Les deux améliorent → soumettre la combinaison, puis bagging
- [ ] Aucun n'améliore → plafond atteint pour cette famille d'approches →
      passer à un modèle de nature différente (réseau temporel 1D sur les
      séries mensuelles), seule source de diversité encore inexploitée

**Retenu : _à remplir_**

## Observations
-

## Limites / risques
1. **Pas d'early stopping** : les modèles sont entraînés à `n_estimators`
   fixe. Raison assumée — l'early stopping exige un `eval_set` dont la
   gestion diffère selon les librairies et interagit mal avec les poids
   d'échantillons ; à `n_estimators` fixe les trois familles reçoivent un
   traitement strictement identique, condition d'une comparaison honnête.
   Coût : hyperparamètres légèrement sous-optimaux par modèle.
2. **Hyperparamètres XGBoost/CatBoost non retunés** sur les 409 features
   actuelles (repris d'exp005/exp008, calibrés sur 245 features). Si le blend
   apporte un gain, un tuning dédié pourrait l'amplifier.
3. Les pseudo-étiquettes proviennent du LightGBM seul, y compris pour le
   blend. Une variante serait de les générer depuis le blend lui-même
   (pseudo-étiquettes de meilleure qualité) — non testé ici pour isoler les
   effets.
4. Le blend utilise des poids égaux. L'optimisation des poids (exp009) avait
   donné un gain de +0.0004 seulement, avec des poids très instables d'un
   seed à l'autre — délibérément non refaite.
5. `scale_pos_weight` figé à 1.6 (exp013 : levier mort), seuil de
   pseudo-étiquetage figé à 0.90 et 1 tour (exp016 : le tour 2 dégrade).

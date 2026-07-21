# exp004_class_weight — Résultats

## Hypothèse testée (H2, EDA étape 3)
Le prior train (40.4% positifs) est probablement différent du prior test réel
(non chiffré par les organisateurs, juste "plus élevé"). Le seuil 0.5 est
imposé par le règlement — le levier légitime est `scale_pos_weight` à
l'entraînement.

## Particularité de ce test
Contrairement à H3/H5, **pas de règle de décision automatique**. Le score OOF
baisse mécaniquement avec le poids (la CV reste jugée au prior train 40/60) —
ce n'est pas un signal d'échec. L'indicateur pertinent est le **taux de
positifs prédit sur Test.csv**.

## Résultats du balayage (`run.py`, seed 42, features.py 245 features)

| scale_pos_weight | AUC (OOF) | F1 (OOF) | Score (OOF) | Taux positifs (test) |
|---|---|---|---|---|
| 1.0 (référence) | 0.9940 | 0.9618 | 0.9747 | 0.6107 |
| 1.3 | 0.9941 | 0.9621 | 0.9749 | 0.6175 |
| 1.6 | 0.9941 | 0.9622 | 0.9750 | 0.6252 |
| 2.0 | 0.9941 | 0.9609 | 0.9742 | 0.6272 |
| 2.5 | 0.9941 | 0.9615 | 0.9746 | 0.6320 |

## Lecture des résultats
- Contrairement à l'avertissement initial, le score OOF ne baisse quasiment
  pas sur toute la grille (écart max 0.0008, comparable au bruit de CV
  observé sur H3/H5) — LightGBM absorbe bien la pondération à seuil 0.5.
- Le taux de positifs test progresse de 61.1% à 63.2%, mais avec un
  **rendement décroissant** : la majeure partie du mouvement se fait entre
  1.0 et 1.6 (+1.4 point), puis plafonne entre 1.6 et 2.5 (+0.7 point sur
  un intervalle 1.6x plus large).
- `scale_pos_weight = 1.6` obtient le meilleur score OOF de la grille
  (0.9750, légèrement au-dessus de la référence).

## Décision
- Poids retenu : **1.6**
- Justification : meilleur score OOF de la grille (léger mieux, pas une
  perte) + capture l'essentiel du rapprochement vers un prior test plus
  élevé avant la zone de rendement décroissant (2.0-2.5, gain marginal,
  risque de sur-correction si le vrai prior test n'est pas si extrême).

## Observations
- Le levier scale_pos_weight a moins d'effet que prévu sur ce dataset/modèle
  — probablement parce que l'augmentation par masquage + les features
  agrégées font déjà un travail robuste indépendant du seuil de décision.

## Limites / à vérifier avant de conclure
- Le vrai prior test est inconnu — toute décision ici reste une hypothèse non
  vérifiable avant soumission Zindi (rappel : variable la moins contrôlée,
  cf. conclusions EDA).
- Le taux de positifs prédit à w=1.0 dépend aussi du modèle/features, pas
  seulement du prior réel — ne pas sur-interpréter un seul chiffre.
- Envisager de tester également `is_unbalance=True` (LightGBM) en
  comparaison, qui rééquilibre automatiquement sans poids explicite.
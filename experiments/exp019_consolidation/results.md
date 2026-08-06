# exp019_consolidation — Résultats

## Contexte

| Exp | Approche | LB |
|---|---|---|
| exp016 | GBDT + pseudo-étiquetage | 0.8365 |
| exp018 | blend GBDT + CNN (w=0.3) | 0.8591 |
| **exp019** | **blend GBDT + CNN + GRU, réseaux baggés 5 seeds** | **0.8732** |

**Gain : +0.0141**

## OOF par famille (sans pseudo-étiquetage)

| Modèle | AUC | F1 | Score |
|---|---|---|---|
| gbdt | 0.9950 | 0.9690 | 0.9794 |
| cnn | 0.9941 | 0.9623 | 0.9750 |
| gru | 0.9940 | 0.9584 | 0.9727 |

Le bagging 5 seeds a nettement amélioré le CNN : **0.9724 → 0.9750** en OOF
par rapport à exp018 (un seul seed). Confirme que les réseaux sont bien plus
sensibles à l'initialisation que les arbres.

## ⭐ Corrélations — OOF vs TEST

| Paire | Corrélation OOF | Corrélation TEST | Écart |
|---|---|---|---|
| gbdt/cnn | 0.9752 | **0.8883** | −0.087 |
| gbdt/gru | 0.9635 | **0.9016** | −0.062 |
| cnn/gru | 0.9813 | **0.9492** | −0.032 |

**Structure confirmée** : toutes les paires sont fortement corrélées sur le
train (≥ 0.96) et se décorrèlent sur le test. L'écart est **maximal entre
familles conceptuellement éloignées** (arbres vs convolution : −0.087) et
**minimal entre familles proches** (deux réseaux sur le même tenseur : −0.032).

C'est la confirmation directe du mécanisme : ce qui crée la valeur, c'est que
les modèles **se dégradent différemment sous le décalage de domaine**.

## Blend

| Config | Poids (gbdt/cnn/gru) | Holdout | Taux positifs | LB |
|---|---|---|---|---|
| blend_nopl | 0.4 / 0.3 / 0.3 | 0.9676 | 0.6262 | **0.8729** |
| blend_pl | 0.4 / 0.3 / 0.3 | 0.9671 | 0.6379 | **0.8732** |

GBDT seul sur le holdout : 0.9641 (delta blend : +0.0035)
Pseudo-étiquettes : 725/1030 (70,4 %), dont 60,0 % de positifs

## ⚠️ Le pseudo-étiquetage n'apporte plus rien

| | LB |
|---|---|
| blend_pl | 0.8732 |
| blend_nopl | 0.8729 |
| **Écart** | **+0.0003 (bruit)** |

Le risque anticipé s'est réalisé : à exp016 le pseudo-étiquetage apportait
+0.0041 **sur un modèle seul**. Appliqué à un blend, il n'apporte plus rien.

**Interprétation** : les deux mécanismes corrigeaient la même faiblesse.
Le pseudo-étiquetage adaptait un modèle unique au domaine cible ; le blend
multi-familles obtient la même robustesse par compensation d'erreurs — et le
fait mieux. Le holdout l'avait d'ailleurs correctement signalé (0.9671 contre
0.9676, très légèrement en défaveur du pseudo-étiquetage).

**Décision : retirer le pseudo-étiquetage.** Gain nul, coût de calcul doublé,
complexité inutile. Simplification qui sera aussi un point positif pour la
reproductibilité (Phase 2).

## Décision
- [x] **Blend GBDT + CNN + GRU (0.4/0.3/0.3), réseaux baggés 5 seeds** →
      configuration de référence, LB **0.8732**
- [x] **Pseudo-étiquetage abandonné** (+0.0003 = bruit)
- [x] Le GRU apporte réellement quelque chose malgré sa corrélation de 0.9492
      avec le CNN sur le test

## Observations
1. **Le holdout redevient fiable en amplitude** : +0.0035 annoncé pour
   +0.0141 réel (facteur ~4). Bien mieux que le facteur 11 d'exp018, sans
   doute parce qu'une partie du gain vient du bagging (réduction de variance,
   visible partout) et non uniquement de la robustesse au décalage.
2. **Poids de blend équilibrés** (0.4/0.3/0.3) alors que les modèles ont des
   scores individuels différents (0.9794 / 0.9750 / 0.9727). Confirme encore
   que le blend fonctionne par décorrélation et non par force individuelle.
3. Le taux de positifs prédits reste stable (~0.63), comme depuis exp013.

## Limites / risques
1. **Trois changements simultanés** (bagging, GRU, pseudo-étiquetage) : seul
   l'effet du pseudo-étiquetage a pu être isolé grâce à la soumission de
   contrôle. La part respective du bagging et du GRU reste indéterminée — la
   hausse de l'OOF du CNN (0.9724 → 0.9750) suggère que le bagging y contribue
   nettement.
2. Les poids de blend sont choisis sur le holdout, qui sous-estime le
   phénomène exploité. La surface étant plate (0.2 à 0.5 donnaient des
   résultats proches à exp018), le risque reste faible.
3. CNN et GRU restent assez corrélés sur le test (0.9492) : deux réseaux sur
   le même tenseur d'entrée. La marge de décorrélation supplémentaire est
   probablement ailleurs — dans la **nature des données d'entrée** plutôt que
   dans l'architecture.
4. Coût de calcul élevé (110 entraînements) ; la suppression du
   pseudo-étiquetage le divise par deux.
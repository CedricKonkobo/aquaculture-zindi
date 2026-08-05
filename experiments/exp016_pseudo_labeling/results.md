# exp016_pseudo_labeling — Résultats

## Contexte

| Exp | Approche | LB |
|---|---|---|
| baseline | features brutes, spw=1.6 | 0.820 |
| exp012 | alignement par domaine | 0.778 |
| exp013 | spw = 0.5 | 0.805 |
| exp014 | drop features décalées | 0.825 |
| exp015 | base + invariantes | 0.8324 |
| **exp016** | **+ pseudo-étiquetage (seuil 0.90)** | **0.8365** |

## ⚠️ Note de comparabilité du holdout

Le tour 0 donne 0.9454, alors qu'exp015 annonçait 0.9524 pour la **même**
configuration. Ce n'est pas une régression : le holdout n'est pas composé des
mêmes groupes. À exp015, la propension adversariale était calculée sur les
245 features de base ; ici elle l'est sur les 409 features (base + invariantes),
ce qui sélectionne des groupes différents — l'AUC adversariale de contrôle
passe d'ailleurs de 0.9925 à **0.9977**.

**Conséquence : la référence valide pour cette expérience est le tour 0
(0.9454), pas le 0.9524 d'exp015.** Les deltas du script sont calculés
correctement par rapport au tour 0.

À retenir pour la suite : **le holdout doit être reconstruit à l'identique
pour toute comparaison entre expériences**, sinon les niveaux absolus ne sont
pas comparables.

## Résultats

Holdout test-like : 4550 lignes (455 groupes sur 1821)
AUC adversariale de contrôle : 0.9977

| Seuil | Tour | Nb pseudo | % positifs | Holdout | Δ vs tour 0 | CV std | Taux positifs test |
|---|---|---|---|---|---|---|---|
| — | 0 (réf.) | 0 | — | 0.9454 | — | 0.9780 | 0.6175 |
| **0.90** | 1 | 838 (81.4 %) | 61.2 % | **0.9489** | **+0.0036** | 0.9807 | 0.6311 |
| 0.95 | 1 | 765 (74.3 %) | 59.5 % | 0.9481 | +0.0028 | 0.9785 | 0.6282 |
| 0.99 | 1 | 499 (48.4 %) | 49.9 % | 0.9489 | +0.0036 | 0.9793 | 0.6252 |
| 0.99 | 2 | 560 (54.4 %) | 53.9 % | 0.9476 | +0.0023 | 0.9787 | 0.6252 |

## Soumission

| Variante | Nb pseudo | Score LB |
|---|---|---|
| seuil 0.90, tour 1 | 838 | **0.8365** (vs 0.8324) |

**Gain : +0.0041**

## Décision
- [x] **Holdout stable/en hausse + LB en hausse → piste adoptée**

Configuration retenue : base + invariantes (409 features), pseudo-étiquetage
seuil 0.90, 1 tour, poids 0.5.

## Observations — trois enseignements

**1. Le pseudo-étiquetage fonctionne, et c'est la confirmation que
l'adaptation de domaine est la bonne famille de techniques.**
Après quatre échecs (transformer, recalibrer, supprimer), la première approche
qui montre au modèle des données de la période cible produit un gain net. La
cause du problème était bien le décalage de domaine, et il fallait le
traverser, pas le contourner.

**2. Le tour 2 dégrade** (0.9476 contre 0.9489 au tour 1). C'est le début de
l'amplification d'erreurs : le modèle commence à réapprendre ses propres
approximations. **Un seul tour est le bon réglage** — l'itération
supplémentaire n'est pas rentable.

**3. ⭐ Le holdout confirme sa fiabilité directionnelle — troisième point.**

| | Holdout | LB |
|---|---|---|
| exp014 → exp015 | +0.0052 | +0.0073 |
| exp015 → exp016 | +0.0036 | +0.0041 |

Deux transitions consécutives où le holdout a prédit le **signe** et l'**ordre
de grandeur** du gain réel. C'est désormais un instrument sur lequel on peut
s'appuyer pour itérer sans brûler de soumissions.

**4. Le déséquilibre des pseudo-étiquettes n'a pas explosé.** Le garde-fou
`pseudo_pos_rate` donne 61,2 % au seuil 0.90, proche du taux prédit (62,3 %),
et descend même à 49,9 % au seuil 0.99. Pas d'emballement du biais de classe.
Fait notable : les seuils 0.90 et 0.99 donnent le **même** holdout (0.9489)
avec des volumes très différents (838 vs 499 pseudo-étiquettes) — le gain vient
donc de l'accès au domaine cible, pas du volume de données ajoutées.

## Limites / risques
1. Le holdout ne pouvait pas valider seul le gain (il est construit sur le
   train, le bénéfice porte sur le test) — d'où la validation LB. Il s'est
   avéré malgré tout directionnellement correct, ce qui renforce sa crédibilité.
2. Les échantillons pseudo-étiquetés font partie du test qu'on prédit : leurs
   prédictions sont mécaniquement renforcées. Le gain n'est donc pas uniforme
   sur l'ensemble du test.
3. `pseudo_weight` figé à 0.5, non balayé — paramètre potentiellement
   sous-exploité (cf. étape suivante).
4. **Pipeline mono-modèle** : toute la chaîne exp015-016 tourne sur un seul
   LightGBM. Le blend d'exp006 n'a jamais été reporté sur la configuration
   actuelle — c'est le gisement le plus évident encore inexploité.
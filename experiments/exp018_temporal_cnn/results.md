# exp018_temporal_cnn — Résultats

## Contexte

| Exp | Approche | LB |
|---|---|---|
| exp015 | base + invariantes | 0.8324 |
| exp016 | + pseudo-étiquetage (pw=0.5) | 0.8365 |
| exp017 | pw=2.0 / blend 3 GBDT | 0.8362 (plat) |
| **exp018** | **blend GBDT + CNN temporel (w=0.3)** | **0.8591** |

**Gain : +0.0226 — le plus important de tout le projet.**

## Architecture du CNN

| hidden | dropout | Holdout AUC | Holdout F1 | Holdout score |
|---|---|---|---|---|
| **32** | **0.3** | 0.9790 | 0.9276 | **0.9482** |
| 64 | 0.3 | 0.9767 | 0.9272 | 0.9470 |
| 64 | 0.5 | 0.9739 | 0.9111 | 0.9362 |

Retenue : hidden=32, dropout=0.3 — la plus petite architecture gagne, cohérent
avec la faible taille du jeu de données.

## OOF conjointes (mêmes folds)

| Modèle | AUC | F1 | Score |
|---|---|---|---|
| GBDT (LightGBM) | 0.9950 | 0.9690 | 0.9794 |
| CNN temporel | 0.9930 | 0.9587 | 0.9724 |

Le CNN est individuellement **plus faible** que le GBDT — sans importance,
comme prévu : le critère était la décorrélation.

## ⭐ Corrélation des prédictions — ET L'ERREUR D'INTERPRÉTATION

| | Corrélation |
|---|---|
| **OOF (domaine train)** | **0.9701** |
| **Test (domaine cible)** | **0.8716** |

**Le script a conclu « TROP CORRÉLÉ → impasse ». Cette conclusion était
FAUSSE**, et la soumission l'a démontré (+0.0226).

**Pourquoi** : la règle de décision utilisait la corrélation **OOF**. Or l'OOF
est calculée sur le domaine train, où les deux modèles s'accordent (0.97).
C'est sur le **domaine test** qu'ils divergent (0.87) — et c'est précisément là
que le blend opère.

**Interprétation** : les deux modèles se dégradent **différemment** sous le
décalage de domaine. Le GBDT s'appuie sur des statistiques agrégées, le CNN sur
la dynamique temporelle ; le changement de période ne les affecte pas de la
même manière. Leurs erreurs sur le test sont donc partiellement indépendantes,
et leur moyenne les compense.

**RÈGLE À RETENIR : pour juger de la complémentarité de deux modèles sous
décalage de domaine, utiliser la corrélation sur le TEST, pas sur l'OOF.**

*Note : la corrélation entre les trois GBDT d'exp017 n'avait pas été mesurée.
Elle était probablement élevée sur le test aussi (mêmes features, même famille
d'algorithmes), ce qui expliquerait la neutralité de ce blend — contrairement
à celui-ci.*

## Blend GBDT + CNN

| Config | w_cnn | Holdout | OOF | Taux positifs |
|---|---|---|---|---|
| **blend_w0.3** | **0.3** | **0.9662** | 0.9797 | 0.6136 |
| blend_w0.4 | 0.4 | 0.9659 | 0.9799 | 0.6126 |
| blend_w0.5 | 0.5 | 0.9644 | 0.9796 | 0.6117 |
| blend_w0.2 | 0.2 | 0.9643 | 0.9793 | 0.6126 |
| gbdt_alone | 0.0 | 0.9641 | 0.9794 | 0.6097 |
| cnn_alone | 1.0 | 0.9502 | 0.9724 | 0.5670 |

## Soumission

| Config | Holdout | LB |
|---|---|---|
| blend_w0.3 | 0.9662 (+0.0021) | **0.8591** (+0.0226) |

## ⚠️ Second enseignement : le holdout sous-estime massivement

| Transition | Holdout | LB | Ratio |
|---|---|---|---|
| exp014 → exp015 (features) | +0.0052 | +0.0073 | ~1.4× |
| exp015 → exp016 (pseudo-étiq.) | +0.0036 | +0.0041 | ~1.1× |
| exp016 → exp017 (intensité pw) | +0.0021 | −0.0003 | ✗ circularité |
| **exp017 → exp018 (blend CNN)** | **+0.0021** | **+0.0226** | **~11×** |

Le holdout a correctement prédit le **signe**, mais a sous-estimé l'amplitude
d'un facteur 10.

**Explication** : le holdout est construit sur le **train** (propension
moyenne 0.008 — à peine test-like). Le bénéfice du blend provient de la
**robustesse au décalage de domaine**, qui ne se manifeste que sur le vrai
domaine cible. Le holdout ne peut structurellement pas la mesurer.

**Règle : le holdout sous-estime systématiquement tout gain provenant de la
robustesse au décalage. Un gain holdout faible mais positif sur ce type de
changement mérite une soumission.**

## Décision
- [x] **Blend GBDT + CNN adopté** — nouvelle configuration de référence, LB **0.8591**
- [x] La corrélation **test** (0.87) est le bon critère, pas l'OOF (0.97)
- [x] Le plafond n'était PAS atteint : la diversité de **nature de modèle**
      fonctionne, contrairement à la diversité intra-famille (exp017)

## Observations
1. **Le motif du projet se confirme une troisième fois** : ce qui paie est
   l'apport d'information réellement nouvelle. Features invariantes (+0.0073),
   accès au domaine cible (+0.0041), dynamique temporelle via une famille de
   modèles différente (+0.0226). Empiler des modèles de même nature ne donne
   rien (exp017 : +0.0011, neutre au LB).
2. **Le poids optimal est faible (0.3)** alors que le CNN est nettement plus
   faible individuellement. C'est la signature d'un blend qui fonctionne par
   décorrélation et non par force du second modèle.
3. Le taux de positifs prédits reste stable (~0.61), comme dans toutes les
   configurations depuis exp013.

## Limites / risques
1. **Ce blend n'inclut PAS le pseudo-étiquetage** (délibérément, pour isoler
   l'effet du CNN). Le gain d'exp016 (+0.0041) n'est donc pas dans ce 0.8591 —
   les deux devraient s'additionner. **C'est l'action prioritaire suivante.**
2. **Variance du CNN non maîtrisée** : un seul seed par fold. Les réseaux de
   neurones sont bien plus sensibles à l'initialisation que les GBDT. Un
   bagging multi-seed devrait à la fois améliorer le score et sécuriser le
   classement privé.
3. Grille d'architecture minimale (3 configurations). Maintenant que la piste
   est validée, un travail d'architecture dédié est justifié.
4. Le poids de blend (0.3) a été choisi sur le holdout, qui sous-estime le
   phénomène. L'optimum réel sur le test pourrait être différent — mais les
   valeurs 0.2 à 0.5 donnent des holdouts très proches, donc le risque est
   faible.
5. Pas d'early stopping sur le CNN (30 epochs fixes).
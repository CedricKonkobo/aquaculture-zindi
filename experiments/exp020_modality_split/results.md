# exp020_modality_split — Résultats

## Contexte

| Exp | Approche | LB |
|---|---|---|
| exp018 | blend GBDT + CNN | 0.8591 |
| **exp019** | **blend GBDT + CNN + GRU, baggés 5 seeds** | **0.8732** |

**Structure des corrélations établie à exp019** (écart OOF → TEST) :

| Paire | OOF | TEST | Écart |
|---|---|---|---|
| arbres vs convolution | 0.9752 | 0.8883 | **−0.087** |
| arbres vs récurrent | 0.9635 | 0.9016 | −0.062 |
| convolution vs récurrent | 0.9813 | 0.9492 | −0.032 |

Plus les familles sont **conceptuellement éloignées**, plus elles se
décorrèlent sur le test, et plus le blend gagne. CNN et GRU partagent le même
tenseur → marge la plus faible. **Varier l'architecture donne des rendements
décroissants.**

**Hypothèse testée ici : le prochain axe de diversité est la NATURE DE
L'ENTRÉE.** Sentinel-1 (radar actif) traverse les nuages et est peu sensible à
la saison ; Sentinel-2 (optique passif) dépend fortement de la phénologie, de
l'angle solaire et de l'atmosphère. Les deux modalités ne se dégradent pas de
la même façon quand la période change.

**Prédiction : la paire radar/optique doit montrer l'écart le plus fort
mesuré jusqu'ici (à battre : −0.087).**

Le pseudo-étiquetage est abandonné (exp019 : +0.0003 = bruit une fois le
blend en place).

## Répartition des features
Total : 409 — radar : _à remplir_ — optique : _à remplir_
(les 5 features de contexte sont attribuées aux deux groupes)

## Scores OOF individuels

| Modèle | AUC | F1 | Score |
|---|---|---|---|
| gbdt_all | | | |
| gbdt_s1 | | | |
| gbdt_s2 | | | |
| cnn_all | | | |
| cnn_s1 | | | |
| cnn_s2 | | | |
| gru_all | | | |

## ⭐ Corrélations OOF vs TEST (triées par écart croissant)

```
_à remplir (coller le tableau)_
```

Meilleure paire radar/optique : _à remplir_ (écart _à remplir_)

- [ ] Écart < −0.087 → **prédiction validée** : la modalité décorrèle mieux
      que l'architecture
- [ ] Écart ≥ −0.087 → prédiction non validée : la séparation par modalité
      n'apporte pas plus que la séparation par architecture

**Constat : _à remplir_**

## Blends

| Config | Poids | Holdout | Taux positifs | LB |
|---|---|---|---|---|
| blend_greedy | | | | |
| blend_equal7 | 1/7 chacun | | | |

Référence exp019 : holdout 0.9676 / LB 0.8732

## Décision
- [ ] Un des deux blends bat 0.8732 → nouvelle configuration de référence
- [ ] Le blend glouton bat le blend uniforme → la sélection sur holdout est
      fiable, on peut continuer à l'utiliser
- [ ] Le blend uniforme bat le glouton → la sélection gloutonne sur-ajuste le
      holdout ; privilégier les poids uniformes pour la soumission finale
- [ ] Les modèles par modalité ont un poids nul → la modalité seule est trop
      faible, mais les corrélations restent une information utile pour le
      rapport

**Retenu : _à remplir_**

## Observations
-

## Limites / risques
1. **Déséquilibre des modalités** : 85 features radar contre 329 optiques, et
   2 bandes contre 10. Les modèles radar seront individuellement bien plus
   faibles — ce qui n'est pas le critère (cf. exp018 : le CNN était plus
   faible et a rapporté +0.0226). Mais s'ils sont *trop* faibles, ils
   n'entreront pas dans le blend malgré leur décorrélation.
2. **La sélection gloutonne peut sur-ajuster le holdout**, surtout avec 7
   modèles. D'où le contrôle systématique par un blend à poids égaux, plus
   robuste. La comparaison des deux au LB tranchera.
3. Les modèles par modalité partagent les 5 features de contexte, ce qui
   introduit une légère dépendance entre eux (choix assumé : ces features sont
   informatives sur la structure d'acquisition et peu coûteuses).
4. **Coût de calcul très élevé** : 5 folds × (3 GBDT + 4 réseaux × 10 seeds)
   = 215 entraînements. Lancer en arrière-plan.
5. Architectures et epochs identiques pour tous les réseaux, sans réglage
   dédié par modalité — le tenseur radar n'a que 4 canaux contre 20 pour
   l'optique, une architecture plus petite serait peut-être préférable.

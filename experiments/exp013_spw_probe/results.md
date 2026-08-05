# exp013_spw_probe — Résultats

## Contexte
exp012 (LB 0.778 vs 0.82) a montré que l'alignement de distributions
**dégrade** : forcer les marginales train/test à être identiques détruit une
information réelle. Le test diffère génuinement du train dans sa composition.
→ Retour aux features **brutes**.

Constat déclencheur : le modèle prédit **48,8 %** de positifs sur le test
alors que le train en contient ~38 %. Il sur-prédit les bassins → faux
positifs → précision dégradée → **F1 effondré au seuil 0.5** imposé. Or le F1
pèse 60 % du score. C'est l'explication la plus probable du gap OOF/LB.

`scale_pos_weight` est le levier **légal** : il déplace la distribution des
probabilités, donc l'endroit où tombe la coupe à 0.5. Le seuil lui-même reste
strictement à 0.5.

## Résultats du balayage

Taux de positifs dans le train : _à remplir_

| spw | OOF AUC | OOF F1 | OOF score | Taux positifs test | Écart au train |
|---|---|---|---|---|---|
| 0.5 | | | | | |
| 0.7 | | | | | |
| 1.0 | | | | | |
| 1.3 | | | | | |
| 1.6 | | | | | |
| 2.0 | | | | | |

_à remplir (coller la synthèse)_

## ⚠️ Pourquoi l'OOF ne doit PAS décider
L'OOF est calculé sur la distribution du **train**. Il sélectionnera donc le
spw optimal pour le train (~1.6, cf. exp004), pas pour le test. C'est
exactement l'erreur d'exp004 : le bon réglage y avait été choisi sur un
indicateur qui ne reflète pas les conditions du test.
La seule information locale exploitable est le **taux de positifs prédits**.

## Soumissions envoyées

| spw | Taux positifs | Score LB |
|---|---|---|
| 1.6 (référence) | 0.488 | 0.820 |
| _à remplir_ | | |
| _à remplir_ | | |

Référence exp012 (aligné) : 0.778

## Décision
- [ ] LB monte quand spw baisse → continuer à descendre jusqu'à inversion ;
      la zone optimale correspond au taux de positifs le plus proche de la
      vraie proportion de bassins du test
- [ ] LB baisse quand spw baisse → le modèle sous-prédisait, remonter spw
      au-dessus de 1.6
- [ ] LB ne bouge pas → le F1 n'est pas le facteur limitant, le problème est
      la capacité de généralisation pure → revenir au feature engineering

**spw retenu : _à remplir_**

## Observations
-

## Limites / risques
1. **Risque d'overfit au leaderboard public.** On règle un hyperparamètre
   directement sur le retour du LB. Garde-fou appliqué : 3 soumissions
   maximum, chacune répondant à une question décidée à l'avance, et on
   retient une **zone** plutôt qu'une valeur ultra-précise. Le décalage
   temporel étant présent aussi bien sur le split public que privé, un spw
   qui corrige un biais structurel de calibration devrait se transférer —
   contrairement à un réglage qui grappillerait des décimales.
2. Le taux de positifs prédits est un **proxy** de la vraie proportion du
   test, qui reste inconnue. On ne peut que l'encadrer par les retours LB.
3. Modèle unique (LightGBM manuel) pour isoler l'effet de spw. Une fois la
   zone identifiée, re-vérifier avec le blend (exp014) : l'optimum de spw
   peut légèrement se déplacer avec un ensemble.
4. Légalité : `scale_pos_weight` est un hyperparamètre d'**entraînement**,
   pas un seuil de décision. Le seuil reste 0.5 dans `make_submission()`.
   Conforme au règlement.

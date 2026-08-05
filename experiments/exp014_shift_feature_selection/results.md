# exp014_shift_feature_selection — Résultats

## Contexte : ce que les deux échecs précédents nous ont appris

| Exp | Approche | LB | Leçon |
|---|---|---|---|
| baseline | features brutes, spw=1.6 | **0.820** | référence |
| exp012 | alignement des distributions | 0.778 | **transformer** les features écrase de l'information vraie |
| exp013 | spw = 0.5 | 0.805 | `scale_pos_weight` est un **levier mort** |

**Pourquoi spw est mort** : en faisant varier spw d'un facteur 4 (0.5 → 2.0),
le taux de positifs prédits ne bouge que de 0.594 à 0.627. Le modèle est
**saturé** : avec une AUC OOF de 0.994, ses probabilités sont collées à 0 et
1, donc presque aucun échantillon n'est près de la frontière 0.5. Changer les
poids ne fait traverser la frontière qu'à une poignée de points.

**Diagnostic clé** — décomposition du LB : 0.82 = 0.6·F1 + 0.4·AUC. Les
valeurs plausibles sont **AUC ≈ 0.90-0.93 et F1 ≈ 0.75**. L'AUC elle-même
s'effondre (0.994 en OOF → ~0.91 sur le test). Or **l'AUC ne dépend d'aucun
seuil ni d'aucune calibration** : c'est donc un vrai **échec de
généralisation temporelle**, pas un problème de placement de seuil. Aucune
astuce de calibration ne le corrigera.

→ D'où cette expérience : **supprimer** les features qui ne se transfèrent
pas d'une période à l'autre (au lieu de les déformer comme à exp012).

## Résultats

Holdout test-like : _à remplir_ lignes (_à remplir_ groupes)
AUC adversariale de contrôle : _à remplir_

| K retirées | Nb features | Holdout score | Holdout AUC | Holdout F1 | CV standard | Taux positifs test |
|---|---|---|---|---|---|---|
| 0 | 245 | | | | | |
| 25 | 220 | | | | | |
| 50 | 195 | | | | | |
| 100 | 145 | | | | | |
| 150 | 95 | | | | | |

Écart moyen CV standard − holdout : _à remplir_

## Le holdout est-il un meilleur juge ?
- [ ] Oui — holdout nettement < CV standard : il capture une partie de la
      difficulté réelle du test → l'utiliser pour les décisions suivantes
- [ ] Non — les deux sont proches : la propension adversariale au niveau
      groupe ne capture pas le décalage → continuer à trancher au LB

**Constat : _à remplir_**

## Soumission
Meilleur K selon le holdout : _à remplir_
Fichier envoyé : _à remplir_ — Score LB : _à remplir_
(référence à battre : 0.820)

## Décision
- [ ] Le holdout s'améliore quand K augmente ET le LB confirme → les features
      décalées nuisaient bien ; retenir ce K comme nouvelle base
- [ ] Le holdout se dégrade dès K=25 → les features décalées sont aussi les
      plus **prédictives** (décalées ET indispensables) ; ne pas les
      supprimer → passer aux features intrinsèquement invariantes
      (statistiques **intra-échantillon** : rang d'un mois par rapport aux
      autres mois du *même* échantillon, formes de séries normalisées
      individuellement)
- [ ] LB ≈ 0.82 → le drop est neutre ; garder le pipeline le plus simple

## Observations
-

## Limites / risques
1. **Une feature peut être décalée ET indispensable.** Le classement
   adversarial mesure uniquement le degré de décalage, pas l'utilité pour la
   tâche. C'est précisément ce que le balayage sur K permet d'arbitrer
   empiriquement.
2. **Le holdout test-like reste du train.** Avec une AUC adversariale de
   0.99, même les groupes « les plus test-like » ne ressemblent que
   modérément au test. Le holdout est donc un juge *meilleur*, pas un juge
   *parfait* — le LB reste l'arbitre final.
3. Découpage au niveau **groupe** et non ligne : les répliques
   d'augmentation d'un même échantillon restent du même côté, sans quoi il y
   aurait fuite de données et le holdout serait trivialement facile.
4. `scale_pos_weight` figé à 1.6 (exp013 : levier mort). Modèle unique
   (LightGBM manuel) pour isoler l'effet du drop ; blend et bagging viendront
   après, une fois la direction validée.
5. Budget de soumission : **une seule** sur cette expérience (le meilleur K).
   Il reste peu de soumissions utiles d'ici la clôture, et la Phase 2
   (35 % de la note) demande du temps de rédaction.

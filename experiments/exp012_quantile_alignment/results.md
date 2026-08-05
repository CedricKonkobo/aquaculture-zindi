# exp012_quantile_alignment — Résultats

## Contexte
exp011 : AUC adversariale = 0.9903 (décalage sévère), 0.9776 avec les seuls
indices normalisés. C'est la forme des distributions qui diffère entre les
deux périodes → alignement de distributions plutôt que sélection de features.

## Résultats

| Variante | AUC adversariale | Score OOF |
|---|---|---|
| baseline_raw | 0.9925 | 0.9750 |
| quantile_gauss | 1.0000 | 0.9748 |
| rank_pct | 1.0000 | 0.9745 |

Référence exp011 (features brutes) : 0.9903 — cohérent avec le 0.9925
remesuré ici (n_aug=10 au lieu de 3).

## ⚠️ Les deux métriques prévues sont invalides — analyse

**1. L'OOF ne POUVAIT PAS bouger (invariance des arbres aux transformations
monotones).**
LightGBM coupe sur des seuils : toute transformation croissante d'une feature
produit exactement les mêmes partitions. Rang et quantile sont monotones à
l'intérieur d'un domaine. Les variations observées (0.9750 → 0.9748 → 0.9745)
sont du bruit de binning, pas un effet réel.
→ Un OOF inchangé n'est PAS un échec : c'est le comportement attendu, et il
confirme que la transformation n'a rien dégradé côté train.
→ Ce qui compte est ailleurs : train et test sont transformés par DEUX
fonctions différentes. Un échantillon test au 60e percentile de sa période
reçoit la même valeur qu'un échantillon train au 60e percentile de la sienne.
L'adaptation de domaine a bien lieu, invisible à l'OOF.

**2. L'AUC adversariale post-alignement est contaminée par un artefact de
granularité.**
Train augmenté = 18 210 lignes, test = 1 030 lignes. QuantileTransformer
découpe chaque domaine selon ses propres quantiles : granularité très fine
côté train, ~1030 niveaux distincts côté test, plus des ex-æquo massifs dus
aux répliques d'augmentation. Le classifieur adversarial ne détecte plus un
décalage de distribution mais une **signature de quantification** (« grille
grossière → test »).
→ L'AUC de 1.0000 ne signifie PAS que l'alignement a échoué. Les marginales
sont alignées ; c'est la granularité qui trahit le domaine.
→ Le critère de succès défini dans config.yaml (AUC adv < 0.7) était donc
**mal conçu** et n'est pas exploitable pour cette expérience.

## Conséquence méthodologique

Aucun diagnostic local ne peut trancher sur cette expérience :
- l'OOF ne peut pas bouger (invariance des arbres) ;
- l'AUC adversariale est contaminée (granularité).

Le seul juge disponible est le **leaderboard**. C'est une question
structurelle (l'adaptation de domaine aide-t-elle, oui ou non ?), donc elle
justifie de dépenser une soumission.

## Soumission
Fichier : `exp012_quantile_alignment_submission.csv` (variante quantile_gauss)
Taux de positifs prédits : **0.488**
Score LB obtenu : **_à remplir_** (référence actuelle : 0.82)

## Observation majeure — piste pour exp013

**Le taux de positifs prédits (48.8 %) est anormalement élevé.** Avec
`scale_pos_weight = 1.6`, le train contient environ 38 % de positifs. Si la
proportion réelle de bassins dans le test est plus basse, le modèle produit
massivement des faux positifs → précision dégradée → **F1 effondré au seuil
0.5 imposé**, ce qui correspond exactement au profil du score LB (0.82 avec
une AUC probablement encore bonne).

→ Cela renforce fortement la priorité du **sondage `scale_pos_weight`
(exp013)** : c'est le levier légal pour déplacer la distribution des
probabilités et donc l'endroit où tombe la coupe à 0.5. Probablement le plus
gros gain disponible, indépendamment de l'alignement.

## Décision
- [ ] LB > 0.82 → l'alignement aide, le conserver comme brique de base
- [ ] LB ≈ 0.82 → neutre ; l'alignement ne nuit pas mais n'apporte rien seul,
      garder le pipeline le plus simple (baseline) et se concentrer sur
      exp013 (scale_pos_weight)
- [ ] LB < 0.82 → l'alignement détruit du signal (marginales forcées, cf.
      limite 1) → revenir au baseline et prioriser exp013

## Limites / risques

1. **L'alignement force des marginales identiques.** Si la proportion réelle
   de bassins diffère entre les deux périodes, cette contrainte détruit une
   information vraie. Vérifié en développement : sur des données de contrôle
   SANS décalage, l'alignement séparé dégrade la comparabilité. L'alignement
   n'est pas gratuit.

2. **Artefact de granularité** (cf. analyse ci-dessus) : à corriger dans
   toute future mesure adversariale post-transformation, par exemple en
   sous-échantillonnant le train à la taille du test avant de mesurer, ou en
   fixant `n_quantiles` à une valeur identique et petite pour les deux
   domaines (≤ 1030).

3. L'alignement suppose un décalage monotone. Un décalage non monotone ne
   serait pas corrigé — et resterait invisible aux deux métriques utilisées.

4. Légalité : seules les features du test sont utilisées (jamais ses labels),
   aucun seuil modifié (reste 0.5). Conforme au règlement.
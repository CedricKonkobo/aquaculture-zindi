# exp015_invariant_features — Résultats

## Contexte : où on en est après trois pistes fermées

| Exp | Approche | LB | Leçon |
|---|---|---|---|
| baseline | features brutes, spw=1.6 | 0.820 | — |
| exp012 | alignement par domaine | 0.778 | **transformer** écrase de l'information vraie |
| exp013 | spw = 0.5 | 0.805 | `scale_pos_weight` = **levier mort** (modèle saturé) |
| exp014 | drop features décalées (K=0) | 0.825 | les features décalées sont **indispensables** |
| **exp015** | **base + invariantes** | **0.8324** | **la DIVERSITÉ de features paie** |

## Contrôle d'invariance

| Jeu de features | AUC adversariale |
|---|---|
| base (245) | 0.9925 |
| invariantes (164) | **0.9939** |

Référence exp011 : 0.9903 — Delta : **+0.0014**

- [x] **AUC invariante ≈ base (légèrement PIRE) → hypothèse d'invariance FAUSSE**

**Constat : l'hypothèse de départ est réfutée.** Les features intra-échantillon
ne sont pas plus transférables que les features de base — elles sont même
marginalement plus décalées. Le décalage entre périodes ne porte donc pas sur
les niveaux absolus mais sur la **structure interne elle-même** : la forme
spectrale et la forme temporelle des pixels changent réellement d'une période
à l'autre. Aucune normalisation, quelle qu'elle soit, ne peut y remédier.

## Résultats (triés par holdout_score — la métrique de décision)

| Variante | Nb features | Holdout | Holdout AUC | Holdout F1 | CV std | Taux positifs |
|---|---|---|---|---|---|---|
| **base_invariant** | 409 | **0.9524** | 0.9834 | 0.9318 | 0.9780 | 0.6175 |
| union_regularized | 409 | 0.9477 | 0.9811 | 0.9254 | 0.9757 | 0.6136 |
| base | 245 | 0.9472 | 0.9798 | 0.9255 | 0.9744 | 0.6233 |
| base_regularized | 245 | 0.9396 | 0.9773 | 0.9145 | 0.9716 | 0.6184 |
| invariant | 164 | 0.9238 | 0.9665 | 0.8952 | 0.9669 | 0.6592 |

Référence (exp014, K=0) : holdout 0.9472 / LB 0.825

## Décision
- [x] **`base_invariant` bat la référence (+0.0052) → soumis**
- [x] Les invariantes seules (0.9238) ne battent PAS la base : elles sont un
      **complément**, pas un remplacement
- [x] La régularisation forte **dégrade** (base_regularized −0.0076)

**Variante retenue : `base_invariant` — Score LB : 0.8324** (vs 0.825, **+0.0073**)

## Observations — les trois enseignements majeurs

**1. Le gain vient de la DIVERSITÉ, pas de l'invariance.**
L'hypothèse théorique est fausse (contrôle adversarial), et pourtant l'union
gagne. Les features intra-échantillon capturent des aspects du signal que les
features de base ne capturent pas — elles apportent une information
**complémentaire**, indépendamment de leur transférabilité. C'est un résultat
inattendu qui redéfinit la direction : ce qui paie n'est pas de chercher des
features « robustes au temps » mais d'**augmenter la diversité des angles de
description**.

**2. Le modèle ne surapprend PAS.** La régularisation forte dégrade
(−0.0076 sur le holdout, −0.0028 en CV). L'écart AUC 0.994 local vs ~0.89 test
n'est donc pas de la mémorisation : c'est un décalage de distribution réel que
le modèle subit, pas une faute de capacité. Il faut au contraire **plus**
d'information, pas moins — ce que confirme le point 1.

**3. ⭐ Le holdout test-like fonctionne comme prédicteur.**
Pour la première fois, une métrique locale a correctement anticipé le
leaderboard :

| | Holdout | LB |
|---|---|---|
| exp014 K=0 | 0.9472 | 0.825 |
| exp015 base_invariant | 0.9524 | 0.8324 |
| Delta | **+0.0052** | **+0.0073** |

Même direction, magnitude comparable. Après quatre expériences où la CV
standard mentait (0.975 pour 0.82), on dispose enfin d'un thermomètre
directionnellement fiable. **C'est l'acquis méthodologique le plus important
du projet** : il permet d'itérer sans brûler de soumissions.

## Limites / risques
1. **Une seule observation de corrélation holdout↔LB.** Deux points alignés ne
   font pas une loi. Le holdout reste optimiste en niveau absolu (0.95 vs 0.83)
   et sa propension moyenne est de 0.008 contre 0.000 pour le reste — il est
   *directionnellement* utile, pas calibré. À re-vérifier à chaque soumission.
2. Le profil régularisé est un point unique, pas un balayage : on conclut que
   « plus de contrainte nuit », pas que le réglage actuel est optimal.
3. `scale_pos_weight` figé à 1.6 (exp013 : levier mort).
4. 409 features pour 1821 groupes réels : le risque de sur-ajustement croît
   avec la dimension. Le holdout n'a pas détecté de dégradation, mais un
   bagging multi-seed est souhaitable avant de figer le pipeline final.
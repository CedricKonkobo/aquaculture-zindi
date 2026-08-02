# exp011_adversarial_validation — Résultats

## Contexte
exp010 a confirmé que la **structure** des mois manquants est fidèlement
reproduite par l'augmentation (écart max 0.0003 sur `S2_GAP_P`). Le gap
entre l'OOF local (0.975) et le leaderboard (0.82) doit donc venir d'un
décalage sur les **valeurs**, cohérent avec l'énoncé du challenge (train et
test sur des périodes temporelles différentes).

**Point méthodologique** : le train est augmenté (`mask_window`) AVANT
featurisation, pour que les deux côtés aient la même structure de mois
manquants — sinon le classifieur détecterait trivialement « 12 mois vs
4-6 mois » au lieu du vrai décalage de valeurs.

Dataset adversarial : 5463 lignes train (augmentées ×3), 1030 lignes test,
245 features.

## Résultats

| Run | Nb features | AUC adversariale |
|---|---|---|
| Toutes les features | 245 | **0.9903** |
| Indices normalisés + contexte | 105 | **0.9776** |

Delta (normalisé − toutes) : **−0.0127**

**Interprétation retenue : décalage SÉVÈRE.** Un classifieur sépare train et
test à 0.99 — les deux distributions sont quasi disjointes. Ceci explique
intégralement le gap OOF 0.975 vs LB 0.82 : la CV locale valide sur une
distribution que le test ne contient pas.

## Top features séparant train/test
feature  adv_importance
     BSI_max           184.6
    NDWI_min           152.2
      VH_min           145.2
   swir1_min           121.4
    NDVI_max           118.8
    NDMI_max           109.6
    NDRE_p75           108.2
      VV_p25           106.6
    blue_max           101.6
    VH_VV_diff_max 99.6
NDMI_min 98.8
swir2_iqr 89.6
NDRE_cv 82.2
VH_VV_diff_min 81.4
swir2_cv 81.0
VH_cv 80.8
VV_max 80.8
VH_VV_ratio_mean 79.8
VH_p25 79.2
BSI_p25 78.8
MNDWI_min 77.8
NDRE_med 76.4
VV_min 76.2
NDWI_max 75.2
NDRE_max 74.6

Features de bandes brutes dans le top 25 : **13 / 25**

## Poids de propension
Moyenne : 1.000 — Max après écrêtage : 64.53 — **Max avant écrêtage : 1062.0**
Lignes écrêtées (plafond 10.0) : **117**

## Analyse

1. **Le décalage n'est pas seulement une question de niveaux absolus.**
   Retirer toutes les bandes brutes ne fait baisser l'AUC que de 0.0127
   (0.9903 → 0.9776). Les indices normalisés (NDVI, NDWI, MNDWI, BSI, NDRE)
   sont eux-mêmes fortement décalés et dominent le top 25. C'est la **forme**
   des distributions qui diffère, pas seulement leur échelle. **Conclusion :
   une simple sélection/normalisation par indices ne réglera pas le problème.**

2. **Les statistiques d'extrêmes et de dispersion sont les plus décalées.**
   Le top est dominé par des `_min`, `_max`, `_cv`, `_iqr`, `_p25`, `_p75`.
   Les extrêmes sont les statistiques les plus sensibles à un changement de
   période (un `max` capture le pic saisonnier, qui se déplace). Les
   statistiques centrales (`_mean`, `_med`) sont moins représentées.

3. **⚠️ La repondération par propension (plan initial exp014) est à
   abandonner.** L'*importance weighting* exige un recouvrement entre les
   distributions train et test. Avec une AUC de 0.99, ce recouvrement est
   quasi inexistant — ce que confirment les poids bruts jusqu'à **1062** et
   les 117 lignes écrêtées : presque aucun échantillon train ne ressemble au
   test. Repondérer ferait porter toute la CV sur une poignée d'échantillons
   non représentatifs, ce qui serait pire que la CV actuelle.

## Décision / suite

- [x] **Décalage sévère** → exp013 (alignement de distributions) devient la
      priorité absolue
- [ ] ~~exp014 CV repondérée par propension~~ → **abandonné** (pas de
      recouvrement, cf. point 3)

**Stratégie retenue : alignement par quantiles (rank-gauss).**
Pour chaque feature, transformer train et test *séparément* en leurs rangs
respectifs, mappés sur une distribution commune. Cela détruit tout décalage
monotone (niveau, échelle, étirement) tout en préservant l'**ordre relatif**
des échantillons — or c'est l'ordre qui porte le signal « bassin vs autre ».
Légal : n'utilise que les features du test (jamais ses labels), ne touche à
aucun seuil.

**Critère de succès mesurable d'exp013** : relancer exp011 après alignement
doit faire chuter l'AUC adversariale (cible < 0.7). Cette boucle de
vérification remplace la CV cassée comme signal de progrès.

En parallèle : **exp012** (sondage leaderboard sur `scale_pos_weight`) reste
prioritaire pour corriger le placement du seuil 0.5, indépendamment du
décalage de features.

## Limites
- Le classifieur adversarial mesure un décalage de **covariables**
  (distribution des features), pas un décalage de la relation
  features → label. Il ne dit pas si P(y|X) a changé entre les deux périodes.
  Si c'est le cas, même un alignement parfait des features ne suffira pas.
- `n_aug=3` pour le train : 5463 lignes train vs 1030 test. L'AUC reste
  interprétable (insensible au déséquilibre), mais les importances peuvent
  être bruitées sur les features rares.
- L'alignement par quantiles prévu à exp013 suppose que le décalage est
  **monotone**. S'il ne l'est pas entièrement, l'AUC adversariale ne
  descendra que partiellement — ce sera en soi une information utile sur la
  nature du décalage.
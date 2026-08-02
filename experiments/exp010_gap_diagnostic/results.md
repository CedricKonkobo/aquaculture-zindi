# exp010_gap_diagnostic — Résultats

## Objectif
Vérifier que `S2_GAP_P` et la fenêtre `[4, 6]` mois codées en dur dans
`src/utils.py` (utilisées par `src/augmentation.py` pour toute l'augmentation
depuis exp001) correspondent à ce qu'on observe réellement dans `Test.csv`.

## Longueur de fenêtre observée
window_len
4 345
5 343
6 342

Hypothèse codée : uniforme sur [4, 6].
% hors de cette plage : **0.0 %** (0 échantillon sur 1030)
Distribution quasi parfaitement uniforme sur {4, 5, 6} — conforme à
`rng.integers(min_len, max_len + 1)` dans `mask_window`.

## Consécutivité
% de fenêtres réellement consécutives : **100.0 %**
L'hypothèse de fenêtre continue de `mask_window` est exacte, aucun cas
de fenêtre fragmentée dans le test.

## Trous S2 dans la fenêtre

| n_gaps | p_empirique | p_code (S2_GAP_P) |
|---|---|---|
| 0 | 0.7350 | 0.735 |
| 1 | 0.2243 | 0.224 |
| 2 | 0.0359 | 0.036 |
| 3 | 0.0049 | 0.005 |

Écart absolu max : **0.0003**
Trous observés au-delà de 3 mois ? **Non** — `S2_GAP_P` couvre toute la plage
observée, aucun cas que `mask_window` serait incapable de générer.

## Verdict du script
- [x] **Cohérent — pas de changement nécessaire**
- [ ] Écart notable — mise à jour de `src/utils.py` nécessaire

## Observations

- **La correspondance est quasi exacte (écart max 0.0003).** Cette précision
  indique très probablement que `S2_GAP_P` avait été dérivé du test lui-même
  lors de l'EDA initial (`01_EDA.ipynb`). Ce diagnostic est donc une
  **vérification de cohérence** (la constante codée correspond bien au test,
  sans erreur de transcription) plutôt qu'une validation indépendante.
  Sa valeur reste réelle : l'hypothèse est confirmée sans ambiguïté et peut
  être définitivement écartée de la liste des causes suspectes.

- **Conséquence majeure pour la suite.** L'augmentation reproduit fidèlement
  la *structure* des données manquantes du test (quels mois sont présents,
  quels mois ont un trou S2). Elle ne reproduit en revanche **pas** la
  *distribution des valeurs* à ces mois-là. Or le défi est explicitement un
  décalage de période temporelle ("trained on data from one time period and
  tested on data from a different one") : le test provient d'une autre
  saison/année, donc réflectances, niveaux d'eau, phénologie de la végétation
  et cycles de production des bassins diffèrent.

- **Cela recentre le diagnostic du gap CV/leaderboard (OOF 0.975 vs LB 0.82)
  sur le décalage de distribution des VALEURS**, pas sur la structure des
  trous. Le masquage temporel, aussi bien calibré soit-il, ne peut rien
  contre ce type de décalage.

## Décision / suite

Aucune modification de `src/utils.py` ni de `src/augmentation.py`.
Hypothèse "gap mal calibré" **écartée définitivement**.

Prochaine étape : **exp011 — validation adversariale** (classifieur train vs
test) pour :
1. mesurer l'ampleur réelle du décalage de covariables (AUC adversariale) ;
2. identifier **quelles features décalent le plus** entre les deux périodes,
   afin de les retirer ou de les remplacer par des équivalents invariants au
   temps (indices spectraux, statistiques d'ordre, features radar S1).

## Limites
- Le diagnostic suppose que le radar S1 (VH/VV) est un indicateur fiable de
  "dans la fenêtre" (cf. docstring d'augmentation.py). Les 100 % de fenêtres
  consécutives et les longueurs strictement dans [4, 6] confirment que cette
  hypothèse tient sur le test : aucun trou S1 parasite ne vient fausser
  l'estimation de la longueur de fenêtre.
- Ce diagnostic ne dit **rien** sur la distribution des valeurs elles-mêmes
  (réflectances, rétrodiffusion radar) entre train et test — c'est
  précisément l'objet d'exp011.
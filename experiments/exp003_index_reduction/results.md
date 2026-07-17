# exp003_index_reduction — Résultats

## Hypothèse testée (H3, EDA étapes 6-7)
NDVI, NDWI, MNDWI, NDRE, AWEI ont une forte séparabilité individuelle
(0.90-0.95, étape 6) mais sont corrélés entre eux à 0.95-0.99 (étape 7) — un
seul signal mesuré 5 fois. Les réduire à 2 représentants (MNDWI + NDRE)
devrait alléger le modèle sans perte de score.

## Changement vs exp001_baseline (référence — H5/exp002 a été rejetée)
Uniquement `src/features.py` → `src/features_v3.py` (`DROP_SERIES = {NDVI,
NDWI, AWEI}`, garde MNDWI + NDRE). Rien d'autre ne change.

## Résultats multi-seed (`compare_seeds.py`, seeds 42/1/7 — dès le départ, leçon de H5)

| Seed | AUC v1 | AUC v3 | F1 v1 | F1 v3 | Score v1 | Score v3 | Delta |
|---|---|---|---|---|---|---|---|
| 42 | 0.9940 | 0.9938 | 0.9618 | 0.9609 | 0.9747 | 0.9741 | -0.0006 |
| 1  | 0.9938 | 0.9935 | 0.9627 | 0.9615 | 0.9751 | 0.9743 | -0.0008 |
| 7  | 0.9940 | 0.9937 | 0.9638 | 0.9633 | 0.9759 | 0.9755 | -0.0004 |

**Delta moyen : -0.0006 — Delta std : 0.0002**

Nb features : v1 = 245, v3 = 215

Règle de décision (`|delta moyen| < delta std` → bruit, sinon signal réel) :
**0.0006 > 0.0002 → condition non vérifiée → delta significatif, pas du bruit.**

## Décision
- [ ] Gain net → fusionner `features_v3.py` dans `features.py`
- [ ] Neutre (delta < std) → garder pour la légèreté/interprétabilité
- [x] **Perte cohérente sur les 3 seeds → rejeter, hypothèse infirmée (comme H5)**

**Action :** `features_v3.py` abandonné. `features.py` (245 features) reste la référence.

## Observations
- Même pattern que H5 (exp002) : delta petit mais négatif et cohérent sur les
  3 seeds → NDVI, NDWI et AWEI portent chacun un signal résiduel exploitable
  par LightGBM malgré leur forte corrélation (0.95-0.99) avec MNDWI/NDRE.
- Deux hypothèses de réduction de redondance rejetées d'affilée (H5 puis H3) :
  suggère que la piste "suppression de features corrélées" n'est probablement
  pas le bon levier sur ce dataset — abandon de cette direction, pas de H4
  (une 3e variante de suppression).

## Limites / à vérifier avant de conclure
- Rappel H5 : la corrélation forte entre features n'implique pas une
  redondance informationnelle totale pour LightGBM — confirmé une 2e fois ici.
- Piste abandonnée pour la suite : ne pas retester une 3e variante de
  suppression de features corrélées sans nouvelle preuve motivant un cas
  différent — deux échecs cohérents suffisent à changer de direction.
- Prochaine hypothèse à privilégier : class weights (H2, décalage de prior
  train/test) ou comparaison de modèles (LightGBM vs CatBoost/XGBoost),
  plutôt que la réduction de features.
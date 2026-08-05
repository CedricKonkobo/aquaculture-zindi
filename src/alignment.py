"""
alignment.py
-------------
Adaptation de domaine par alignement de quantiles (rank-gauss).

Contexte (voir exp011) : le classifieur adversarial sépare train et test avec
une AUC de 0.99 -- les deux distributions sont quasi disjointes, parce que
train et test proviennent de périodes temporelles différentes. Retirer les
bandes brutes ne suffit pas (AUC 0.9776 avec les seuls indices normalisés) :
c'est la FORME des distributions qui diffère, pas seulement leur échelle.

Principe : pour chaque feature, on transforme train et test SÉPARÉMENT en
leurs rangs respectifs, mappés sur une même distribution gaussienne. Tout
décalage monotone (niveau, échelle, étirement) est ainsi détruit, tandis que
l'ORDRE RELATIF des échantillons est préservé -- or c'est l'ordre qui porte
le signal "bassin vs autre" : un bassin reste dans le haut de la distribution
NDWI de sa propre période, quelle que soit la saison.

Légalité vis-à-vis du règlement : on n'utilise que les FEATURES du test
(jamais ses labels), et on ne touche à aucun seuil de classification. C'est
de l'adaptation de domaine transductive standard.

Limite à garder en tête : cette approche suppose que le décalage est
MONOTONE. S'il ne l'est pas entièrement, l'alignement ne le corrigera que
partiellement (mesurable via l'AUC adversariale post-alignement).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


def align_by_quantiles(X_train, X_test, n_quantiles=1000, output_distribution="normal",
                        seed=42):
    """
    Aligne X_train et X_test en les transformant SÉPARÉMENT en quantiles.

    Chaque domaine est mappé sur la même distribution cible à partir de ses
    PROPRES rangs -- c'est ce qui neutralise le décalage entre périodes.

    Les NaN sont ignorés lors de l'ajustement et propagés tels quels par
    QuantileTransformer (comportement natif de scikit-learn), ce qui est
    important ici : les NaN portent l'information "mois nuageux" que
    src/features.py exploite déjà via n_s2 / frac_cloud.

    Retourne (X_train_aligned, X_test_aligned), mêmes colonnes, mêmes index.
    """
    common_cols = list(X_train.columns)
    assert list(X_test.columns) == common_cols, \
        "X_train et X_test doivent avoir exactement les mêmes colonnes, dans le même ordre"

    # n_quantiles ne peut pas dépasser le nombre d'échantillons du domaine
    nq_train = int(min(n_quantiles, len(X_train)))
    nq_test = int(min(n_quantiles, len(X_test)))

    qt_train = QuantileTransformer(n_quantiles=nq_train,
                                    output_distribution=output_distribution,
                                    subsample=int(1e9), random_state=seed)
    qt_test = QuantileTransformer(n_quantiles=nq_test,
                                   output_distribution=output_distribution,
                                   subsample=int(1e9), random_state=seed)

    Xtr = pd.DataFrame(qt_train.fit_transform(X_train), columns=common_cols,
                        index=X_train.index)
    Xte = pd.DataFrame(qt_test.fit_transform(X_test), columns=common_cols,
                        index=X_test.index)
    return Xtr, Xte


def align_by_rank(X_train, X_test):
    """
    Variante plus simple : rang normalisé dans [0, 1] par domaine.

    Moins lisse que l'alignement gaussien mais totalement non paramétrique et
    sans dépendance à n_quantiles. Utile comme point de comparaison pour
    vérifier que le gain vient bien de l'alignement et non d'un réglage fin
    de QuantileTransformer.
    """
    Xtr = X_train.rank(pct=True, na_option="keep")
    Xte = X_test.rank(pct=True, na_option="keep")
    return Xtr, Xte

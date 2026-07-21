"""
train.py
--------
Entraînement et validation croisée.

Principe de validation "honnête" :
- GroupKFold (via StratifiedGroupKFold) sur l'ID D'ORIGINE de l'échantillon, pas sur l'index
  de la réplique augmentée. Sans ça, deux répliques masquées du même pond pourraient se
  retrouver l'une en train et l'autre en validation -> fuite de données et score trop optimiste.
- Le score OOF est donc calculé sur des répliques masquées, c'est-à-dire dans des conditions
  proches du test réel (fenêtre courte + trous nuageux), pas sur des échantillons 12 mois complets.

Ce module ne connaît PAS le détail des features ni du chargement : il prend X, Y, groups déjà
construits et un constructeur de modèle. Ça permet de comparer facilement plusieurs familles de
modèles (LightGBM, CatBoost, XGBoost, ...) sans dupliquer la boucle de CV.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score

from src.utils import SEED


def weighted_score(f1, auc):
    """Métrique composite utilisée par la compétition : 0.6*F1 + 0.4*AUC."""
    return 0.6 * f1 + 0.4 * auc


def default_fit_fn(model, X_tr, y_tr, X_val, y_val, early_stopping_rounds=100):
    """
    Fit générique historique (LightGBM) : eval_set + eval_metric="auc", avec
    early stopping si le modèle est un LightGBM. Sert de comportement par
    défaut de cross_validate -- inchangé pour exp001-004.

    D'autres familles de modèles (CatBoost, XGBoost, RandomForest sans
    eval_set...) ont une API de fit différente : voir le paramètre fit_fn de
    cross_validate, utilisé par exp005_model_comparison pour les brancher
    sans dupliquer la boucle de CV.
    """
    fit_kwargs = dict(eval_set=[(X_val, y_val)], eval_metric="auc")
    if "lightgbm" in type(model).__module__:
        import lightgbm as lgb
        fit_kwargs["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
    model.fit(X_tr, y_tr, **fit_kwargs)


def cross_validate(model_factory, X, Y, groups, n_splits=5, threshold=0.5, seed=SEED,
                    verbose=True, early_stopping_rounds=100, fit_fn=None):
    """
    Effectue une StratifiedGroupKFold CV.

    Paramètres
    ----------
    model_factory : callable() -> objet sklearn-like avec .fit(X, y, ...) et .predict_proba(X)
        Permet de brancher n'importe quel modèle (LightGBM, CatBoost, RandomForest, ...)
        sans changer cette fonction. Voir experiments/ pour des exemples de factories.
    X : DataFrame de features
    Y : array de labels binaires
    groups : array d'IDs d'origine (pour éviter la fuite entre répliques augmentées)
    threshold : seuil de décision pour le F1 (0.5 par défaut, voir Étape 6 pour l'optimisation)
    fit_fn : callable(model, X_tr, y_tr, X_val, y_val, early_stopping_rounds), optionnel
        Permet de personnaliser l'appel .fit() pour des modèles dont l'API diffère
        de LightGBM (CatBoost, XGBoost, RandomForest...). Par défaut : default_fit_fn
        (comportement historique, inchangé pour les expériences existantes).

    Retourne
    --------
    dict avec oof (probas out-of-fold), models (liste des modèles entraînés par fold),
    et les métriques par fold + globales.
    """
    fit_fn = fit_fn or default_fit_fn
    oof = np.zeros(len(X))
    models = []
    fold_metrics = []

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tri, vai) in enumerate(skf.split(X, Y, groups=groups)):
        model = model_factory()
        fit_fn(model, X.iloc[tri], Y[tri], X.iloc[vai], Y[vai], early_stopping_rounds)
        proba = model.predict_proba(X.iloc[vai])[:, 1]
        oof[vai] = proba
        models.append(model)

        fold_auc = roc_auc_score(Y[vai], proba)
        fold_f1 = f1_score(Y[vai], (proba > threshold).astype(int))
        fold_metrics.append({"fold": fold, "auc": fold_auc, "f1": fold_f1})
        if verbose:
            print(f"fold {fold}  AUC={fold_auc:.4f}  F1@{threshold}={fold_f1:.4f}")

    auc = roc_auc_score(Y, oof)
    f1 = f1_score(Y, (oof > threshold).astype(int))
    result = {
        "oof": oof,
        "models": models,
        "fold_metrics": pd.DataFrame(fold_metrics),
        "auc": auc,
        "f1": f1,
        "score": weighted_score(f1, auc),
    }
    if verbose:
        print(f"\nOOF global  AUC={auc:.4f}  F1={f1:.4f}  score={result['score']:.4f}")
    return result


def find_best_threshold(y_true, proba, thresholds=None):
    """
    Recherche le seuil de décision qui maximise le F1 (au lieu du 0.5 fixe).
    Utilisé à l'Étape 6 (optimisation), une fois le modèle et les features stabilisés.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, (proba > t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(scores))
    return thresholds[best_idx], scores[best_idx]


def feature_importance(model, feature_names, top_n=20):
    """Retourne les top_n features les plus importantes pour un modèle déjà entraîné."""
    imp = pd.Series(model.feature_importances_, index=feature_names)
    return imp.sort_values(ascending=False).head(top_n)


def cross_validate_blend(model_specs, X, Y, groups, n_splits=5, threshold=0.5, seed=SEED,
                          verbose=True, early_stopping_rounds=100, weights=None):
    """
    Entraîne plusieurs modèles sur EXACTEMENT les mêmes folds (même split
    StratifiedGroupKFold) et moyenne leurs probabilités OOF -> blend simple
    ou pondéré. Sert à comparer un ensemble à ses modèles individuels sur une
    base équitable (contrairement à moyenner des OOF venant de CV séparées,
    dont les folds ne coïncideraient pas forcément de la même façon).

    Paramètres
    ----------
    model_specs : liste de (nom, model_factory, fit_fn)
        Un triplet par modèle à inclure dans le blend (voir experiments/
        exp005_model_comparison/run.py pour des factories/fit_fn existants).
    weights : dict {nom: poids}, optionnel (défaut : poids égaux -> moyenne simple)

    Retourne
    --------
    dict avec oof_per_model (probas OOF par modèle), oof_blend (moyenne
    pondérée), models_per_model (modèles entraînés par fold, pour l'inférence
    test), et metrics (AUC/F1/score par modèle + pour le blend).
    """
    weights = weights or {name: 1.0 for name, _, _ in model_specs}
    n = len(X)
    oof_per_model = {name: np.zeros(n) for name, _, _ in model_specs}
    models_per_model = {name: [] for name, _, _ in model_specs}

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tri, vai) in enumerate(skf.split(X, Y, groups=groups)):
        for name, factory, fit_fn in model_specs:
            model = factory()
            fit_fn(model, X.iloc[tri], Y[tri], X.iloc[vai], Y[vai], early_stopping_rounds)
            proba = model.predict_proba(X.iloc[vai])[:, 1]
            oof_per_model[name][vai] = proba
            models_per_model[name].append(model)
        if verbose:
            print(f"fold {fold} entraîné pour tous les modèles du blend")

    total_w = sum(weights.values())
    oof_blend = sum(weights[name] * oof_per_model[name] for name in oof_per_model) / total_w

    metrics = {}
    for name, proba in oof_per_model.items():
        auc = roc_auc_score(Y, proba)
        f1 = f1_score(Y, (proba > threshold).astype(int))
        metrics[name] = {"auc": auc, "f1": f1, "score": weighted_score(f1, auc)}
    auc_b = roc_auc_score(Y, oof_blend)
    f1_b = f1_score(Y, (oof_blend > threshold).astype(int))
    metrics["blend"] = {"auc": auc_b, "f1": f1_b, "score": weighted_score(f1_b, auc_b)}

    if verbose:
        for name, m in metrics.items():
            print(f"{name:12s}  AUC={m['auc']:.4f}  F1={m['f1']:.4f}  score={m['score']:.4f}")

    return {
        "oof_per_model": oof_per_model,
        "oof_blend": oof_blend,
        "models_per_model": models_per_model,
        "metrics": metrics,
    }

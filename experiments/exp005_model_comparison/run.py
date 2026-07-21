"""
experiments/exp005_model_comparison/run.py
----------------------------------------------
Étape 5 du plan initial : comparer plusieurs familles de modèles avec
EXACTEMENT la même validation (mêmes répliques augmentées, mêmes folds,
même seuil) avant de passer à l'optimisation d'hyperparamètres (Étape 6).

Modèles comparés :
    - LightGBM     (référence, reprend scale_pos_weight=1.6 validé par H2)
    - CatBoost      (gère les NaN nativement, souvent fort sur peu de données)
    - XGBoost       (gère les NaN nativement)
    - RandomForest  (sklearn -- imputation médiane nécessaire, pas de NaN natif)
    - ExtraTrees    (idem RandomForest)

Chaque famille a une API .fit() différente (eval_set, early stopping...) :
voir fit_fn dans src/train.cross_validate (généralisé pour ce besoin précis,
sans dupliquer la boucle de CV -- un seul endroit connaît le détail du split).

Usage :
    python -m experiments.exp005_model_comparison.run
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.preprocessing import load_train, load_test
from src.augmentation import build_augmented_replicas
from src.features import featurize_many, featurize   # référence (245 features)
from src.train import cross_validate, default_fit_fn
from src.inference import predict_proba_ensemble, make_submission, save_models
from src.utils import SEED, MODELS_DIR

N_AUG = 10
SPW = 1.6   # scale_pos_weight retenu à l'issue de exp004 (H2)


# ------------------------------------------------------------------ factories
def lightgbm_factory():
    return lgb.LGBMClassifier(
        objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=40,
        feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
        lambda_l1=0.5, lambda_l2=5.0, n_estimators=1200, verbose=-1, seed=SEED,
        scale_pos_weight=SPW,
    )


def catboost_factory():
    return CatBoostClassifier(
        iterations=1200, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
        eval_metric="AUC", random_seed=SEED, scale_pos_weight=SPW,
        verbose=False, allow_writing_files=False,
    )


def xgboost_factory():
    return XGBClassifier(
        n_estimators=1200, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=5.0,
        scale_pos_weight=SPW, eval_metric="auc", random_state=SEED,
        early_stopping_rounds=100,
    )


def randomforest_factory():
    # Pas de NaN natif chez sklearn -> imputation médiane dans un Pipeline
    # (fit_fn ne passe alors QUE X_tr, y_tr : pas d'eval_set/early stopping ici).
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight={0: 1, 1: SPW},
            random_state=SEED, n_jobs=-1,
        )),
    ])


def extratrees_factory():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight={0: 1, 1: SPW},
            random_state=SEED, n_jobs=-1,
        )),
    ])


# -------------------------------------------------------------------- fit_fn
def catboost_fit_fn(model, X_tr, y_tr, X_val, y_val, early_stopping_rounds=100):
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
              early_stopping_rounds=early_stopping_rounds, verbose=False)


def xgboost_fit_fn(model, X_tr, y_tr, X_val, y_val, early_stopping_rounds=100):
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)


def sklearn_fit_fn(model, X_tr, y_tr, X_val, y_val, early_stopping_rounds=100):
    # Pipeline sklearn : pas d'eval_set / early stopping, juste .fit(X, y)
    model.fit(X_tr, y_tr)


MODELS = {
    "lightgbm": (lightgbm_factory, default_fit_fn),
    "catboost": (catboost_factory, catboost_fit_fn),
    "xgboost": (xgboost_factory, xgboost_fit_fn),
    "randomforest": (randomforest_factory, sklearn_fit_fn),
    "extratrees": (extratrees_factory, sklearn_fit_fn),
}


def main():
    tr_ids, y, tr_cube = load_train()
    te_ids, _, te_cube = load_test()

    cubes, ys, groups = build_augmented_replicas(tr_cube, y, n_aug=N_AUG, seed=SEED)
    X = featurize_many(cubes)
    Y = np.concatenate(ys)
    G = np.concatenate(groups)
    Xte = featurize(te_cube)[X.columns]
    print(f"train augmenté : {X.shape} | test : {Xte.shape}\n")

    rows = []
    for name, (factory, fit_fn) in MODELS.items():
        print(f"=== {name} ===")
        result = cross_validate(factory, X, Y, G, n_splits=5, threshold=0.5,
                                 seed=SEED, verbose=False, fit_fn=fit_fn)
        proba_test = predict_proba_ensemble(result["models"], Xte)

        rows.append({
            "model": name,
            "auc_oof": round(result["auc"], 4),
            "f1_oof": round(result["f1"], 4),
            "score_oof": round(result["score"], 4),
            "test_pos_rate": round((proba_test > 0.5).mean(), 4),
        })
        print(f"  AUC={result['auc']:.4f}  F1={result['f1']:.4f}  "
              f"score={result['score']:.4f}  taux_positif(test)={(proba_test > 0.5).mean():.3f}")

        make_submission(te_ids, proba_test, threshold=0.5,
                         filename=f"exp005_{name}_submission.csv")
        save_models(result["models"], name=f"exp005_{name}")

    df = pd.DataFrame(rows).sort_values("score_oof", ascending=False)
    print("\n" + df.to_string(index=False))
    out = MODELS_DIR / "exp005_model_comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nRésultats écrits : {out}")


if __name__ == "__main__":
    main()

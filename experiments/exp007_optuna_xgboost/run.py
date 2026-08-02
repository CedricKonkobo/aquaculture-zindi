"""
experiments/exp007_optuna_xgboost/run.py
--------------------------------------------
Étape 6 (première partie) : optimisation d'hyperparamètres XGBoost avec
Optuna, comparé au meilleur XGBoost manuel de exp005 (score OOF 0.9754).

Deux phases pour rester rapide malgré l'augmentation lourde (18210 lignes
avec n_aug=10) :
    1. RECHERCHE : n_aug réduit (5) + CV à 3 folds -> proxy rapide et moins
       cher pour explorer l'espace des hyperparamètres (N_TRIALS essais).
    2. CONFIRMATION : les meilleurs hyperparamètres trouvés sont réévalués
       avec la config complète (n_aug=10, 5 folds), pour comparaison directe
       et équitable avec exp005/exp006 (mêmes conditions).

scale_pos_weight reste fixé à 1.6 (décision H2, exp004) -- non retuné ici
pour ne pas mélanger deux hypothèses différentes dans la même recherche.

Usage :
    python -m experiments.exp007_optuna_xgboost.run [n_trials]
    (n_trials par défaut : 40)
"""
import sys
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBClassifier

from src.preprocessing import load_train, load_test
from src.augmentation import build_augmented_replicas
from src.features import featurize_many, featurize
from src.train import cross_validate
from src.inference import predict_proba_ensemble, make_submission, save_models
from src.utils import SEED, MODELS_DIR
from experiments.exp005_model_comparison.run import xgboost_fit_fn

SPW = 1.6                      # décision H2, non retunée ici
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEARCH_N_AUG = 5               # proxy rapide pour la recherche (vs 10 en config finale)
SEARCH_N_SPLITS = 3
FINAL_N_AUG = 10
FINAL_N_SPLITS = 5
BASELINE_SCORE = 0.9754        # meilleur XGBoost manuel, exp005 (référence à battre)


def build_dataset(n_aug, seed=SEED):
    tr_ids, y, tr_cube = load_train()
    cubes, ys, groups = build_augmented_replicas(tr_cube, y, n_aug=n_aug, seed=seed)
    X = featurize_many(cubes)
    Y = np.concatenate(ys)
    G = np.concatenate(groups)
    return X, Y, G


def make_factory(params):
    def factory():
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            gamma=params["gamma"],
            scale_pos_weight=SPW,
            eval_metric="auc",
            random_state=SEED,
            early_stopping_rounds=100,
        )
    return factory


def sample_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 400, 2000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
    }


def main():
    print(f"Phase 1 -- recherche Optuna ({N_TRIALS} essais, proxy rapide "
          f"n_aug={SEARCH_N_AUG}, {SEARCH_N_SPLITS} folds)")
    X_search, Y_search, G_search = build_dataset(SEARCH_N_AUG, seed=SEED)
    print(f"train (recherche) : {X_search.shape}")

    def objective(trial):
        params = sample_params(trial)
        factory = make_factory(params)
        result = cross_validate(factory, X_search, Y_search, G_search,
                                 n_splits=SEARCH_N_SPLITS, threshold=0.5, seed=SEED,
                                 verbose=False, fit_fn=xgboost_fit_fn)
        return result["score"]

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\nMeilleur score (recherche, proxy) : {study.best_value:.4f}")
    print("Meilleurs hyperparamètres :")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    print(f"\nPhase 2 -- confirmation en config complète "
          f"(n_aug={FINAL_N_AUG}, {FINAL_N_SPLITS} folds)")
    X, Y, G = build_dataset(FINAL_N_AUG, seed=SEED)
    te_ids, _, te_cube = load_test()
    Xte = featurize(te_cube)[X.columns]

    best_factory = make_factory(study.best_params)
    result = cross_validate(best_factory, X, Y, G, n_splits=FINAL_N_SPLITS, threshold=0.5,
                             seed=SEED, verbose=True, fit_fn=xgboost_fit_fn)

    print(f"\nOptuna XGBoost (config complète) : AUC={result['auc']:.4f}  "
          f"F1={result['f1']:.4f}  score={result['score']:.4f}")
    print(f"Référence exp005 (XGBoost manuel) : score={BASELINE_SCORE:.4f}")
    print(f"Delta : {result['score'] - BASELINE_SCORE:+.4f}")

    proba_test = predict_proba_ensemble(result["models"], Xte)
    make_submission(te_ids, proba_test, threshold=0.5,
                     filename="exp007_optuna_xgboost_submission.csv")
    save_models(result["models"], name="exp007_optuna_xgboost")

    trials_df = study.trials_dataframe().sort_values("value", ascending=False)
    out = MODELS_DIR / "exp007_optuna_trials.csv"
    trials_df.to_csv(out, index=False)
    print(f"\nHistorique des essais Optuna écrit : {out}")

    best_params_df = pd.DataFrame([study.best_params])
    out2 = MODELS_DIR / "exp007_optuna_best_params.csv"
    best_params_df.to_csv(out2, index=False)
    print(f"Meilleurs hyperparamètres écrits : {out2}")


if __name__ == "__main__":
    main()

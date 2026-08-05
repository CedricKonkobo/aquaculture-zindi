"""
experiments/exp012_quantile_alignment/run.py
------------------------------------------------
Réponse au décalage sévère mesuré à exp011 (AUC adversariale = 0.9903).

exp011 a montré que retirer les bandes brutes ne suffit pas (0.9776 avec les
seuls indices normalisés) : c'est la FORME des distributions qui diffère
entre les deux périodes. La réponse n'est donc pas une sélection de features
mais un ALIGNEMENT de distributions (src/alignment.py).

MÉTRIQUE PRINCIPALE : l'AUC adversariale POST-alignement, pas le score OOF.
Tant que la CV locale est cassée (OOF 0.975 vs LB 0.82), l'AUC adversariale
est le seul indicateur fiable de progrès dont on dispose sans brûler une
soumission.

Une BAISSE de l'OOF après alignement est ATTENDUE et plutôt bon signe : elle
indique que la CV cesse d'exploiter des raccourcis spécifiques au train.

Usage :
    python -m experiments.exp012_quantile_alignment.run
"""
import numpy as np
import pandas as pd

from src.preprocessing import load_train, load_test
from src.augmentation import build_augmented_replicas
from src.features import featurize_many, featurize
from src.alignment import align_by_quantiles, align_by_rank
from src.train import cross_validate, default_fit_fn
from src.inference import predict_proba_ensemble, make_submission
from src.utils import SEED, MODELS_DIR

from experiments.exp005_model_comparison.run import lightgbm_factory
from experiments.exp011_adversarial_validation.run import run_adversarial

N_AUG = 10
N_SPLITS = 5

ADV_AUC_BASELINE = 0.9903      # exp011, features brutes
LB_CURRENT = 0.82              # leaderboard actuel, pour mémoire
OOF_BASELINE = 0.9750          # exp005 lightgbm manuel (NON comparable, cf. docstring)


def build_features():
    """Construit les features train (augmentées) et test, non alignées."""
    tr_ids, y, tr_cube = load_train()
    te_ids, _, te_cube = load_test()

    cubes, ys, groups = build_augmented_replicas(tr_cube, y, n_aug=N_AUG, seed=SEED)
    X_train = featurize_many(cubes)
    Y = np.concatenate(ys)
    G = np.concatenate(groups)
    X_test = featurize(te_cube)[X_train.columns]

    print(f"Train augmenté : {X_train.shape}  |  Test : {X_test.shape}")
    return X_train, Y, G, X_test, te_ids


def measure_adversarial(X_train, X_test, label):
    """AUC adversariale entre X_train et X_test (mesure du décalage résiduel)."""
    X = pd.concat([X_train, X_test], ignore_index=True)
    y_adv = np.concatenate([
        np.zeros(len(X_train), dtype=int),
        np.ones(len(X_test), dtype=int),
    ])
    auc, _ = run_adversarial(X, y_adv, label)
    return auc


def evaluate_variant(name, X_train, Y, G, X_test, run_cv=True):
    """Mesure l'AUC adversariale et (optionnellement) la CV pour une variante."""
    print("\n" + "=" * 70)
    print(f"VARIANTE : {name}")
    print("=" * 70)

    adv_auc = measure_adversarial(X_train, X_test, f"adv_{name}")

    cv_score = None
    models = None
    if run_cv:
        result = cross_validate(lightgbm_factory, X_train, Y, G, n_splits=N_SPLITS,
                                 threshold=0.5, seed=SEED, verbose=True,
                                 fit_fn=default_fit_fn)
        cv_score = result["score"]
        models = result["models"]
        print(f"[{name}] OOF : AUC={result['auc']:.4f}  F1={result['f1']:.4f}  "
              f"score={cv_score:.4f}")

    return {"variant": name, "adv_auc": adv_auc, "cv_score": cv_score}, models


def interpret_adv(auc):
    if auc < 0.7:
        return ("SUCCÈS -- l'alignement fonctionne, les domaines deviennent "
                "comparables. Soumettre et comparer au LB : c'est le test décisif.")
    if auc < 0.9:
        return ("PARTIEL -- le décalage n'est pas entièrement monotone. Soumettre "
                "quand même (gain probable), puis envisager à exp014 une "
                "normalisation PAR ÉCHANTILLON (chaque ligne normalisée par ses "
                "propres statistiques temporelles).")
    return ("ÉCHEC -- le décalage n'est pas monotone. Abandonner l'alignement par "
            "domaine et se rabattre sur des features intrinsèquement invariantes "
            "(rangs INTRA-échantillon, formes de séries normalisées individuellement).")


def main():
    X_train_raw, Y, G, X_test_raw, te_ids = build_features()

    rows = []
    models_per_variant = {}

    # --------------------------------------------------- 1) référence brute
    r, m = evaluate_variant("baseline_raw", X_train_raw, Y, G, X_test_raw)
    rows.append(r)
    models_per_variant["baseline_raw"] = m

    # ------------------------------------------ 2) alignement quantile-gauss
    Xtr_q, Xte_q = align_by_quantiles(X_train_raw, X_test_raw, seed=SEED)
    r, m = evaluate_variant("quantile_gauss", Xtr_q, Y, G, Xte_q)
    rows.append(r)
    models_per_variant["quantile_gauss"] = m

    # ----------------------------------------------- 3) alignement rang [0,1]
    Xtr_r, Xte_r = align_by_rank(X_train_raw, X_test_raw)
    r, m = evaluate_variant("rank_pct", Xtr_r, Y, G, Xte_r)
    rows.append(r)
    models_per_variant["rank_pct"] = m

    # ------------------------------------------------------------- synthèse
    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("SYNTHÈSE")
    print("=" * 70)
    print(df.to_string(index=False))

    base_adv = df.loc[df["variant"] == "baseline_raw", "adv_auc"].iloc[0]
    print(f"\nAUC adversariale de référence (exp011, pour mémoire) : {ADV_AUC_BASELINE:.4f}")
    print(f"AUC adversariale mesurée ici sur features brutes      : {base_adv:.4f}")

    aligned = df[df["variant"] != "baseline_raw"]
    best = aligned.loc[aligned["adv_auc"].idxmin()]
    print(f"\nMeilleure variante alignée : {best['variant']} "
          f"(AUC adv = {best['adv_auc']:.4f}, soit {best['adv_auc'] - base_adv:+.4f})")
    print(f"\nVerdict : {interpret_adv(best['adv_auc'])}")

    print("\nRappel sur la lecture du score OOF :")
    print(f"  - OOF de référence (exp005 lightgbm brut) : {OOF_BASELINE:.4f}")
    print(f"  - Leaderboard actuel                      : {LB_CURRENT:.4f}")
    print("  Une BAISSE de l'OOF après alignement est ATTENDUE : la CV cesse")
    print("  d'exploiter des raccourcis propres au train. Ne PAS choisir la")
    print("  variante sur l'OOF -- choisir sur l'AUC adversariale, puis trancher")
    print("  au leaderboard.")

    # ------------------------------------------------------------ soumission
    # On soumet la variante la MIEUX ALIGNÉE (pas la meilleure OOF).
    best_name = best["variant"]
    Xte_best = {"quantile_gauss": Xte_q, "rank_pct": Xte_r}[best_name]
    proba_test = predict_proba_ensemble(models_per_variant[best_name], Xte_best)
    make_submission(te_ids, proba_test, threshold=0.5,
                     filename="exp012_quantile_alignment_submission.csv")
    print(f"\nSoumission générée à partir de la variante '{best_name}' "
          "(la mieux alignée, pas la meilleure OOF).")

    df.to_csv(MODELS_DIR / "exp012_alignment_adversarial.csv", index=False)
    print(f"Résultats écrits : {MODELS_DIR / 'exp012_alignment_adversarial.csv'}")


if __name__ == "__main__":
    main()

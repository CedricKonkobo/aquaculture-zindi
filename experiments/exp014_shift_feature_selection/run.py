"""
experiments/exp014_shift_feature_selection/run.py
------------------------------------------------------
Deux objectifs :
  1) Retirer les features les plus décalées (classement adversarial d'exp011)
     et mesurer l'effet sur la généralisation.
  2) Construire un JUGE LOCAL fiable : un holdout "test-like" composé des
     groupes du train que le classifieur adversarial juge les plus proches du
     test.

Pourquoi cette direction (après exp012 et exp013) :
  - exp012 (LB 0.778) : TRANSFORMER les features pour aligner les
    distributions écrase de l'information vraie -> échec.
  - exp013 (LB 0.805) : scale_pos_weight est un levier MORT (taux de positifs
    0.594 -> 0.627 pour un facteur 4 sur spw : modèle saturé).
  - Décomposition du LB 0.82 = 0.6*F1 + 0.4*AUC -> AUC test ~0.90-0.93 alors
    que l'AUC OOF est 0.994. L'AUC ne dépendant d'AUCUN seuil, c'est un vrai
    échec de GÉNÉRALISATION. D'où : supprimer les features instables.

Différence avec exp012 : on ne DÉFORME rien, on SUPPRIME.
Différence avec la repondération abandonnée : on ne pondère rien, on DÉCOUPE
(pas de poids extrêmes possibles).

Usage :
    python -m experiments.exp014_shift_feature_selection.run
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, f1_score

from src.preprocessing import load_train, load_test
from src.augmentation import build_augmented_replicas
from src.features import featurize_many, featurize
from src.train import cross_validate, default_fit_fn, weighted_score
from src.inference import predict_proba_ensemble, make_submission
from src.utils import SEED, MODELS_DIR

K_GRID = [0, 25, 50, 100, 150]
N_AUG = 10
N_SPLITS = 5
SPW = 1.6                 # figé : exp013 a montré que ce levier est mort
HOLDOUT_FRACTION = 0.25

LB_BASELINE = 0.82        # spw=1.6, features complètes
LB_ALIGNED = 0.778        # exp012
LB_SPW05 = 0.805          # exp013


def lgbm_factory():
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=1200, learning_rate=0.03,
        num_leaves=31, min_child_samples=30, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.5, lambda_l2=5.0,
        scale_pos_weight=SPW, verbose=-1, seed=SEED,
    )


def adversarial_propensity(X, Xte):
    """
    Probabilité, pour chaque ligne du train, d'être classée "test" par un
    classifieur adversarial. Sert uniquement à ORDONNER les groupes du train
    du moins au plus test-like (aucune pondération n'est appliquée).
    """
    X_all = pd.concat([X, Xte], ignore_index=True)
    y_adv = np.concatenate([np.zeros(len(X)), np.ones(len(Xte))]).astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_all))
    for tr_idx, va_idx in skf.split(X_all, y_adv):
        m = lgb.LGBMClassifier(objective="binary", n_estimators=300,
                               learning_rate=0.05, num_leaves=31,
                               verbose=-1, seed=SEED)
        m.fit(X_all.iloc[tr_idx], y_adv[tr_idx])
        oof[va_idx] = m.predict_proba(X_all.iloc[va_idx])[:, 1]

    adv_auc = roc_auc_score(y_adv, oof)
    print(f"AUC adversariale (contrôle) : {adv_auc:.4f}")
    return oof[:len(X)], adv_auc


def build_testlike_holdout(propensity, G, fraction=HOLDOUT_FRACTION):
    """
    Sélectionne les groupes du train les plus test-like.

    Découpage au niveau GROUPE et non ligne : les répliques d'augmentation
    d'un même échantillon doivent rester du même côté, sinon fuite de données.
    """
    dfp = pd.DataFrame({"group": G, "p": propensity})
    grp_mean = dfp.groupby("group")["p"].mean().sort_values(ascending=False)
    n_hold = max(1, int(round(len(grp_mean) * fraction)))
    holdout_groups = set(grp_mean.index[:n_hold])

    mask_holdout = np.isin(G, list(holdout_groups))
    print(f"Holdout test-like : {mask_holdout.sum()} lignes "
          f"({len(holdout_groups)} groupes sur {len(grp_mean)}), "
          f"propension moyenne {grp_mean.iloc[:n_hold].mean():.3f} "
          f"vs {grp_mean.iloc[n_hold:].mean():.3f} pour le reste")
    return mask_holdout


def score_holdout(X, Y, G, mask_holdout, cols):
    """
    Entraîne sur la partie non-holdout et évalue sur le holdout test-like.
    Retourne le score composite (0.6*F1 + 0.4*AUC) au seuil 0.5.
    """
    Xtr, Ytr = X.loc[~mask_holdout, cols], Y[~mask_holdout]
    Xho, Yho = X.loc[mask_holdout, cols], Y[mask_holdout]

    model = lgbm_factory()
    model.fit(Xtr, Ytr)
    proba = model.predict_proba(Xho)[:, 1]

    auc = roc_auc_score(Yho, proba)
    f1 = f1_score(Yho, (proba > 0.5).astype(int))
    return weighted_score(f1, auc), auc, f1, float((proba > 0.5).mean())


def main():
    # -------------------------------------------------------------- données
    tr_ids, y, tr_cube = load_train()
    te_ids, _, te_cube = load_test()
    cubes, ys, groups = build_augmented_replicas(tr_cube, y, n_aug=N_AUG, seed=SEED)
    X = featurize_many(cubes)
    Y = np.concatenate(ys)
    G = np.concatenate(groups)
    Xte = featurize(te_cube)[X.columns]
    print(f"Train augmenté : {X.shape}  |  Test : {Xte.shape}\n")

    # ------------------------------------- classement adversarial (exp011)
    imp_path = MODELS_DIR / "exp011_adversarial_importance.csv"
    if not imp_path.exists():
        raise FileNotFoundError(
            f"{imp_path} introuvable -- relance exp011 d'abord (il produit le "
            "classement des features par degré de décalage)."
        )
    imp = pd.read_csv(imp_path).sort_values("adv_importance", ascending=False)
    ranked_features = [f for f in imp["feature"].tolist() if f in X.columns]
    print(f"Classement adversarial chargé : {len(ranked_features)} features\n")

    # ---------------------------------------- holdout test-like (le juge)
    print("=" * 70)
    print("CONSTRUCTION DU HOLDOUT TEST-LIKE")
    print("=" * 70)
    propensity, adv_auc = adversarial_propensity(X, Xte)
    mask_holdout = build_testlike_holdout(propensity, G)
    print()

    # ------------------------------------------------- balayage sur K
    rows = []
    for K in K_GRID:
        dropped = ranked_features[:K]
        cols = [c for c in X.columns if c not in set(dropped)]

        print("=" * 70)
        print(f"K = {K} features retirées  ->  {len(cols)} features conservées")
        print("=" * 70)

        # juge principal : holdout test-like
        ho_score, ho_auc, ho_f1, ho_pos = score_holdout(X, Y, G, mask_holdout, cols)
        print(f"HOLDOUT test-like : AUC={ho_auc:.4f}  F1={ho_f1:.4f}  "
              f"score={ho_score:.4f}  (taux positifs {ho_pos:.3f})")

        # référence optimiste : CV standard
        cv = cross_validate(lgbm_factory, X[cols], Y, G, n_splits=N_SPLITS,
                             threshold=0.5, seed=SEED, verbose=False,
                             fit_fn=default_fit_fn)
        print(f"CV standard       : AUC={cv['auc']:.4f}  F1={cv['f1']:.4f}  "
              f"score={cv['score']:.4f}")

        proba_test = predict_proba_ensemble(cv["models"], Xte[cols])
        fname = f"exp014_dropK{K}_submission.csv"
        make_submission(te_ids, proba_test, threshold=0.5, filename=fname)
        print()

        rows.append({
            "K": K, "n_features": len(cols),
            "holdout_score": round(ho_score, 4),
            "holdout_auc": round(ho_auc, 4),
            "holdout_f1": round(ho_f1, 4),
            "cv_score": round(cv["score"], 4),
            "test_pos_rate": round(float((proba_test > 0.5).mean()), 4),
            "submission": fname,
        })

    # ------------------------------------------------------------ synthèse
    df = pd.DataFrame(rows)
    print("=" * 70)
    print("SYNTHÈSE")
    print("=" * 70)
    print(df.to_string(index=False))

    gap = df["cv_score"].mean() - df["holdout_score"].mean()
    print(f"\nÉcart moyen CV standard - holdout test-like : {gap:+.4f}")
    if gap < 0.005:
        print(">> ATTENTION : le holdout n'est PAS plus difficile que la CV standard.")
        print("   Il ne constitue donc pas un meilleur juge (le décalage train/test")
        print("   n'est pas capturé par la propension adversariale au niveau groupe).")
        print("   Continuer à trancher au leaderboard, et le noter dans results.md.")
    else:
        print(">> Le holdout est nettement plus difficile que la CV standard : il")
        print("   capture une partie de la difficulté réelle du test. C'est le juge")
        print("   à utiliser pour les prochaines décisions.")

    best = df.loc[df["holdout_score"].idxmax()]
    print(f"\nMeilleur K selon le holdout : K={best['K']} "
          f"(score={best['holdout_score']:.4f}, {best['n_features']} features)")
    print(f"-> soumettre {best['submission']}  (UNE seule soumission)")
    print(f"\nRéférences LB : brut spw=1.6 -> {LB_BASELINE} | aligné -> {LB_ALIGNED} "
          f"| spw=0.5 -> {LB_SPW05}")

    df.to_csv(MODELS_DIR / "exp014_shift_selection.csv", index=False)
    pd.DataFrame({"rank": range(1, len(ranked_features) + 1),
                  "feature": ranked_features}).to_csv(
        MODELS_DIR / "exp014_dropped_features.csv", index=False)
    print(f"\nRésultats écrits : {MODELS_DIR / 'exp014_shift_selection.csv'}")


if __name__ == "__main__":
    main()

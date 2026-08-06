"""
experiments/exp017_blend_pseudo/run.py
-------------------------------------------
Deux gisements combinés :

  A) Balayage de pseudo_weight (intensité de l'adaptation au domaine cible).
     exp016 a montré que les seuils 0.90 et 0.99 donnent le MÊME holdout
     (0.9489) malgré 838 vs 499 pseudo-étiquettes : le gain vient de l'ACCÈS
     au domaine cible, pas du VOLUME ajouté. L'intensité de l'adaptation
     compte donc probablement plus que le nombre d'échantillons -- or ce
     paramètre est figé à 0.5 depuis exp016 et n'a jamais été balayé.

  B) Blend LightGBM + XGBoost + CatBoost sur la configuration actuelle.
     Toute la chaîne exp015-016 tourne sur un SEUL LightGBM ; le dernier
     blend date d'exp009, sur des features désormais obsolètes. La diversité
     a payé deux fois sur ce problème (+0.0073 puis +0.0041).

HOLDOUT FIGÉ : à exp016, le holdout n'avait pas la même composition qu'à
exp015 (propension calculée sur 245 puis 409 features), d'où un tour 0 à
0.9454 contre 0.9524 annoncé pour la même configuration. Ici le holdout est
calculé UNE FOIS sur les features de base et sauvegardé sur disque ; les
expériences suivantes doivent le recharger tel quel.

Usage :
    python -m experiments.exp017_blend_pseudo.run
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score

from src.preprocessing import load_train, load_test
from src.augmentation import build_augmented_replicas
from src.features import featurize_many, featurize
from src.features_invariant import featurize_invariant_many, featurize_invariant
from src.train import weighted_score
from src.inference import make_submission
from src.utils import SEED, MODELS_DIR

PSEUDO_WEIGHT_GRID = [0.25, 0.5, 1.0, 2.0]
PSEUDO_CONF = 0.90        # meilleur seuil, exp016
N_AUG = 10
N_SPLITS = 5
SPW = 1.6
HOLDOUT_FRACTION = 0.25

LB_EXP016 = 0.8365
HOLDOUT_PATH = MODELS_DIR / "exp017_fixed_holdout_groups.csv"


# --------------------------------------------------------------- modèles
def f_lgbm():
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=1200, learning_rate=0.03,
        num_leaves=31, min_child_samples=30, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.5, lambda_l2=5.0,
        scale_pos_weight=SPW, verbose=-1, seed=SEED,
    )


def f_xgb():
    return XGBClassifier(
        n_estimators=1200, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=5, reg_alpha=0.5, reg_lambda=5.0,
        scale_pos_weight=SPW, eval_metric="auc", random_state=SEED,
        tree_method="hist",
    )


def f_cat():
    return CatBoostClassifier(
        iterations=1200, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
        eval_metric="AUC", random_seed=SEED, scale_pos_weight=SPW,
        verbose=False, allow_writing_files=False,
    )


MODEL_FACTORIES = {"lightgbm": f_lgbm, "xgboost": f_xgb, "catboost": f_cat}


def fit_weighted(model, X, y, w):
    """Entraînement uniforme avec poids, sans early stopping (cf. config.yaml)."""
    model.fit(X, y, sample_weight=w)
    return model


# ------------------------------------------------------------- données
def build_data():
    tr_ids, y, tr_cube = load_train()
    te_ids, _, te_cube = load_test()
    cubes, ys, groups = build_augmented_replicas(tr_cube, y, n_aug=N_AUG, seed=SEED)
    Y = np.concatenate(ys)
    G = np.concatenate(groups)

    X_base = featurize_many(cubes)
    X_inv = featurize_invariant_many(cubes)
    X = pd.concat([X_base, X_inv], axis=1)
    Xte = pd.concat([
        featurize(te_cube)[X_base.columns],
        featurize_invariant(te_cube)[X_inv.columns],
    ], axis=1)

    print(f"Train augmenté : {X.shape}  |  Test : {Xte.shape}")
    return X, X_base, Y, G, Xte, te_ids


def get_fixed_holdout(X_base, Xte_base, G):
    """
    Holdout test-like FIGÉ : calculé une seule fois sur les features de BASE
    (245), puis rechargé depuis le disque pour toutes les expériences
    ultérieures. Garantit la comparabilité des niveaux absolus.
    """
    if HOLDOUT_PATH.exists():
        holdout_groups = set(pd.read_csv(HOLDOUT_PATH)["group"].tolist())
        print(f"Holdout figé rechargé depuis {HOLDOUT_PATH.name} "
              f"({len(holdout_groups)} groupes)")
    else:
        X_all = pd.concat([X_base, Xte_base], ignore_index=True)
        y_adv = np.concatenate([np.zeros(len(X_base)), np.ones(len(Xte_base))]).astype(int)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = np.zeros(len(X_all))
        for tr, va in skf.split(X_all, y_adv):
            m = lgb.LGBMClassifier(objective="binary", n_estimators=300,
                                   learning_rate=0.05, num_leaves=31,
                                   verbose=-1, seed=SEED)
            m.fit(X_all.iloc[tr], y_adv[tr])
            oof[va] = m.predict_proba(X_all.iloc[va])[:, 1]
        print(f"AUC adversariale (features de base) : {roc_auc_score(y_adv, oof):.4f}")

        dfp = pd.DataFrame({"group": G, "p": oof[:len(X_base)]})
        grp = dfp.groupby("group")["p"].mean().sort_values(ascending=False)
        n_hold = max(1, int(round(len(grp) * HOLDOUT_FRACTION)))
        holdout_groups = set(grp.index[:n_hold])
        pd.DataFrame({"group": sorted(holdout_groups)}).to_csv(HOLDOUT_PATH, index=False)
        print(f"Holdout figé calculé et sauvegardé ({len(holdout_groups)} groupes) "
              f"-> {HOLDOUT_PATH.name}")

    mask = np.isin(G, list(holdout_groups))
    print(f"Holdout : {mask.sum()} lignes sur {len(G)}")
    return mask


def select_pseudo(proba_test, conf=PSEUDO_CONF):
    pos = np.where(proba_test >= conf)[0]
    neg = np.where(proba_test <= 1 - conf)[0]
    idx = np.concatenate([pos, neg])
    lab = np.concatenate([np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)])
    return idx, lab


# --------------------------------------------------- évaluation générique
def eval_config(names, X, Y, G, Xte, mask_holdout, pseudo=None, label=""):
    """
    Évalue une configuration (un ou plusieurs modèles, avec ou sans
    pseudo-étiquettes) sur le holdout figé, puis produit la prédiction test
    par CV complète.

    pseudo : None, ou dict {"X": DataFrame, "y": array, "w": float}
    """
    print("=" * 70)
    print(f"CONFIG : {label}  [{'+'.join(names)}]"
          + (f"  pseudo_weight={pseudo['w']}" if pseudo else "  sans pseudo"))
    print("=" * 70)

    # ---------------- holdout : entraîner hors holdout, évaluer sur holdout
    Xtr, Ytr = X.loc[~mask_holdout], Y[~mask_holdout]
    Wtr = np.ones(len(Xtr))
    if pseudo:
        Xtr = pd.concat([Xtr, pseudo["X"]], ignore_index=True)
        Ytr = np.concatenate([Ytr, pseudo["y"]])
        Wtr = np.concatenate([Wtr, np.full(len(pseudo["X"]), pseudo["w"])])

    probas_ho = []
    for name in names:
        m = fit_weighted(MODEL_FACTORIES[name](), Xtr, Ytr, Wtr)
        probas_ho.append(m.predict_proba(X.loc[mask_holdout])[:, 1])
    proba_ho = np.mean(probas_ho, axis=0)

    auc = roc_auc_score(Y[mask_holdout], proba_ho)
    f1 = f1_score(Y[mask_holdout], (proba_ho > 0.5).astype(int))
    ho_score = weighted_score(f1, auc)
    print(f"HOLDOUT : AUC={auc:.4f}  F1={f1:.4f}  score={ho_score:.4f}")

    # ---------------- CV complète pour la prédiction test
    X_cv, Y_cv, G_cv = X, Y, G
    W_cv = np.ones(len(X))
    if pseudo:
        X_cv = pd.concat([X, pseudo["X"]], ignore_index=True)
        Y_cv = np.concatenate([Y, pseudo["y"]])
        G_cv = np.concatenate([G, np.arange(len(pseudo["X"])) + G.max() + 1])
        W_cv = np.concatenate([W_cv, np.full(len(pseudo["X"]), pseudo["w"])])

    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_cv))
    test_preds = []
    for tri, vai in skf.split(X_cv, Y_cv, groups=G_cv):
        fold_va, fold_te = [], []
        for name in names:
            m = fit_weighted(MODEL_FACTORIES[name](), X_cv.iloc[tri], Y_cv[tri], W_cv[tri])
            fold_va.append(m.predict_proba(X_cv.iloc[vai])[:, 1])
            fold_te.append(m.predict_proba(Xte)[:, 1])
        oof[vai] = np.mean(fold_va, axis=0)
        test_preds.append(np.mean(fold_te, axis=0))

    cv_auc = roc_auc_score(Y_cv, oof)
    cv_f1 = f1_score(Y_cv, (oof > 0.5).astype(int))
    cv_score = weighted_score(cv_f1, cv_auc)
    proba_test = np.mean(test_preds, axis=0)
    print(f"CV std  : AUC={cv_auc:.4f}  F1={cv_f1:.4f}  score={cv_score:.4f}")

    fname = f"exp017_{label}_submission.csv"
    make_submission(te_ids_global, proba_test, threshold=0.5, filename=fname)
    print()

    return {
        "config": label, "models": "+".join(names),
        "pseudo_weight": pseudo["w"] if pseudo else None,
        "n_pseudo": len(pseudo["X"]) if pseudo else 0,
        "holdout_score": round(ho_score, 4), "holdout_auc": round(auc, 4),
        "holdout_f1": round(f1, 4), "cv_score": round(cv_score, 4),
        "test_pos_rate": round(float((proba_test > 0.5).mean()), 4),
        "submission": fname,
    }, proba_test


def main():
    global te_ids_global
    X, X_base, Y, G, Xte, te_ids = build_data()
    te_ids_global = te_ids
    Xte_base = Xte[X_base.columns]

    print("\n" + "=" * 70)
    print("HOLDOUT FIGÉ")
    print("=" * 70)
    mask_holdout = get_fixed_holdout(X_base, Xte_base, G)
    print()

    rows = []

    # ------------------------- référence : LightGBM seul, sans pseudo
    r, proba_ref = eval_config(["lightgbm"], X, Y, G, Xte, mask_holdout,
                                pseudo=None, label="lgbm_nopl")
    rows.append(r)

    # ------------------------- pseudo-étiquettes issues de la référence
    idx, lab = select_pseudo(proba_ref, PSEUDO_CONF)
    X_pseudo = Xte.iloc[idx].reset_index(drop=True)
    print(f"Pseudo-étiquettes (seuil {PSEUDO_CONF}) : {len(idx)}/{len(Xte)} "
          f"({len(idx)/len(Xte):.1%}), dont {lab.mean():.1%} de positifs\n")

    # ------------------------- A) balayage de pseudo_weight
    for w in PSEUDO_WEIGHT_GRID:
        r, _ = eval_config(["lightgbm"], X, Y, G, Xte, mask_holdout,
                            pseudo={"X": X_pseudo, "y": lab, "w": w},
                            label=f"lgbm_pw{str(w).replace('.', 'p')}")
        rows.append(r)

    df_lgbm = pd.DataFrame(rows)
    best_pw_row = df_lgbm[df_lgbm["pseudo_weight"].notna()].sort_values(
        "holdout_score", ascending=False).iloc[0]
    best_pw = best_pw_row["pseudo_weight"]
    print(f">> Meilleur pseudo_weight (LightGBM) : {best_pw} "
          f"(holdout {best_pw_row['holdout_score']:.4f})\n")

    # ------------------------- B) blend, sans puis avec pseudo-étiquetage
    names = ["lightgbm", "xgboost", "catboost"]
    r, _ = eval_config(names, X, Y, G, Xte, mask_holdout, pseudo=None,
                        label="blend_nopl")
    rows.append(r)

    r, _ = eval_config(names, X, Y, G, Xte, mask_holdout,
                        pseudo={"X": X_pseudo, "y": lab, "w": best_pw},
                        label=f"blend_pw{str(best_pw).replace('.', 'p')}")
    rows.append(r)

    # ------------------------------------------------------------ synthèse
    df = pd.DataFrame(rows).sort_values("holdout_score", ascending=False)
    print("=" * 70)
    print("SYNTHÈSE (triée par holdout_score)")
    print("=" * 70)
    print(df.to_string(index=False))

    ref = df[df["config"] == "lgbm_nopl"]["holdout_score"].iloc[0]
    best = df.iloc[0]
    print(f"\nRéférence LightGBM sans pseudo (ce holdout) : {ref:.4f}")
    print(f"Meilleure config : {best['config']} (holdout {best['holdout_score']:.4f}, "
          f"delta {best['holdout_score'] - ref:+.4f})")
    print(f"LB actuel (exp016, LightGBM + pseudo pw=0.5) : {LB_EXP016}")

    print("\nRappel de fiabilité du holdout (2 transitions consécutives) :")
    print("  exp014 -> exp015 : holdout +0.0052, LB +0.0073")
    print("  exp015 -> exp016 : holdout +0.0036, LB +0.0041")
    print(f"\n-> Soumettre {best['submission']} (UNE seule soumission).")
    print("   Si le gain se confirme, ajouter le bagging multi-seed par-dessus")
    print("   pour sécuriser le classement privé.")

    df.to_csv(MODELS_DIR / "exp017_blend_pseudo.csv", index=False)
    print(f"\nRésultats écrits : {MODELS_DIR / 'exp017_blend_pseudo.csv'}")


if __name__ == "__main__":
    main()

"""
experiments/exp015_invariant_features/run.py
-------------------------------------------------
Deux angles testés ensemble, jugés sur le holdout test-like validé à exp014
(écart +0.0297 vs CV standard) :

  A) Features INVARIANTES INTRA-ÉCHANTILLON (src/features_invariant.py) :
     rang des bandes entre elles dans le même pixel, forme temporelle
     relative, contrastes auto-normalisés. Aucune statistique de population
     n'intervient -> insensibles par construction à un décalage de période.

  B) RÉGULARISATION FORTE : une AUC OOF de 0.994 pour ~0.89 estimée au test
     traduit de la mémorisation.

Historique qui motive cette expérience :
  exp012 (LB 0.778) : aligner par domaine écrase de l'information vraie.
  exp013 (LB 0.805) : scale_pos_weight est un levier mort (modèle saturé).
  exp014 (LB 0.825) : supprimer les features décalées dégrade monotonement ->
                      elles sont décalées ET indispensables.
  Reste : échec de généralisation pure (l'AUC ne dépend d'aucun seuil).

Usage :
    python -m experiments.exp015_invariant_features.run
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

from src.preprocessing import load_train, load_test
from src.augmentation import build_augmented_replicas
from src.features import featurize_many, featurize
from src.features_invariant import featurize_invariant_many, featurize_invariant
from src.train import cross_validate, default_fit_fn, weighted_score
from src.inference import predict_proba_ensemble, make_submission
from src.utils import SEED, MODELS_DIR

N_AUG = 10
N_SPLITS = 5
SPW = 1.6
HOLDOUT_FRACTION = 0.25

HOLDOUT_BASELINE = 0.9472    # exp014, K=0
LB_BEST = 0.825              # exp014 dropK0
ADV_AUC_BASE = 0.9903        # exp011, features de base


def lgbm_base():
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=1200, learning_rate=0.03,
        num_leaves=31, min_child_samples=30, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.5, lambda_l2=5.0,
        scale_pos_weight=SPW, verbose=-1, seed=SEED,
    )


def lgbm_regularized():
    """Profil fortement contraint : moins de capacité de mémorisation."""
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=600, learning_rate=0.03,
        num_leaves=15, min_child_samples=100, feature_fraction=0.5,
        bagging_fraction=0.7, bagging_freq=1, lambda_l1=5.0, lambda_l2=50.0,
        scale_pos_weight=SPW, verbose=-1, seed=SEED,
    )


def adversarial_auc(X, Xte, label=""):
    """AUC adversariale train vs test sur un jeu de features donné."""
    X_all = pd.concat([X, Xte], ignore_index=True)
    y_adv = np.concatenate([np.zeros(len(X)), np.ones(len(Xte))]).astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_all))
    for tr, va in skf.split(X_all, y_adv):
        m = lgb.LGBMClassifier(objective="binary", n_estimators=300,
                               learning_rate=0.05, num_leaves=31,
                               verbose=-1, seed=SEED)
        m.fit(X_all.iloc[tr], y_adv[tr])
        oof[va] = m.predict_proba(X_all.iloc[va])[:, 1]
    auc = roc_auc_score(y_adv, oof)
    if label:
        print(f"AUC adversariale [{label}] = {auc:.4f}")
    return auc, oof[:len(X)]


def build_testlike_holdout(propensity, G, fraction=HOLDOUT_FRACTION):
    """Groupes du train les plus test-like (découpage au niveau GROUPE)."""
    dfp = pd.DataFrame({"group": G, "p": propensity})
    grp = dfp.groupby("group")["p"].mean().sort_values(ascending=False)
    n_hold = max(1, int(round(len(grp) * fraction)))
    holdout_groups = set(grp.index[:n_hold])
    mask = np.isin(G, list(holdout_groups))
    print(f"Holdout test-like : {mask.sum()} lignes ({len(holdout_groups)} groupes "
          f"sur {len(grp)})")
    return mask


def eval_variant(name, X, Y, G, Xte, mask_holdout, te_ids, factory):
    print("=" * 70)
    print(f"VARIANTE : {name}  ({X.shape[1]} features)")
    print("=" * 70)

    # --- juge principal : holdout test-like
    model = factory()
    model.fit(X.loc[~mask_holdout], Y[~mask_holdout])
    proba_ho = model.predict_proba(X.loc[mask_holdout])[:, 1]
    ho_auc = roc_auc_score(Y[mask_holdout], proba_ho)
    ho_f1 = f1_score(Y[mask_holdout], (proba_ho > 0.5).astype(int))
    ho_score = weighted_score(ho_f1, ho_auc)
    print(f"HOLDOUT : AUC={ho_auc:.4f}  F1={ho_f1:.4f}  score={ho_score:.4f}  "
          f"(réf. exp014 = {HOLDOUT_BASELINE:.4f}, delta {ho_score - HOLDOUT_BASELINE:+.4f})")

    # --- référence optimiste
    cv = cross_validate(factory, X, Y, G, n_splits=N_SPLITS, threshold=0.5,
                         seed=SEED, verbose=False, fit_fn=default_fit_fn)
    print(f"CV std  : AUC={cv['auc']:.4f}  F1={cv['f1']:.4f}  score={cv['score']:.4f}")

    proba_test = predict_proba_ensemble(cv["models"], Xte)
    fname = f"exp015_{name}_submission.csv"
    make_submission(te_ids, proba_test, threshold=0.5, filename=fname)
    print()

    return {
        "variant": name, "n_features": X.shape[1],
        "holdout_score": round(ho_score, 4),
        "holdout_auc": round(ho_auc, 4),
        "holdout_f1": round(ho_f1, 4),
        "cv_score": round(cv["score"], 4),
        "test_pos_rate": round(float((proba_test > 0.5).mean()), 4),
        "submission": fname,
    }


def main():
    tr_ids, y, tr_cube = load_train()
    te_ids, _, te_cube = load_test()
    cubes, ys, groups = build_augmented_replicas(tr_cube, y, n_aug=N_AUG, seed=SEED)
    Y = np.concatenate(ys)
    G = np.concatenate(groups)

    # ------------------------------------------------------------ features
    X_base = featurize_many(cubes)
    Xte_base = featurize(te_cube)[X_base.columns]

    X_inv = featurize_invariant_many(cubes)
    Xte_inv = featurize_invariant(te_cube)[X_inv.columns]

    X_union = pd.concat([X_base, X_inv], axis=1)
    Xte_union = pd.concat([Xte_base, Xte_inv], axis=1)

    print(f"base : {X_base.shape} | invariant : {X_inv.shape} | "
          f"union : {X_union.shape}\n")

    # -------------------- contrôle : les features invariantes sont-elles
    # -------------------- réellement plus transférables ?
    print("=" * 70)
    print("CONTRÔLE D'INVARIANCE")
    print("=" * 70)
    adv_base, prop_base = adversarial_auc(X_base, Xte_base, "features de base")
    adv_inv, _ = adversarial_auc(X_inv, Xte_inv, "features invariantes")
    print(f"Référence exp011 (base) : {ADV_AUC_BASE:.4f}")
    print(f"Delta invariant - base  : {adv_inv - adv_base:+.4f}")
    if adv_inv < adv_base - 0.05:
        print(">> Les features invariantes SONT nettement plus transférables. "
              "L'hypothèse tient.")
    else:
        print(">> Les features invariantes NE SONT PAS plus transférables que la "
              "base. L'hypothèse d'invariance est FAUSSE : le décalage ne vient "
              "pas du niveau absolu mais de la structure interne elle-même. "
              "À noter honnêtement dans results.md.")
    print()

    # ---------------------------------------------------- holdout test-like
    mask_holdout = build_testlike_holdout(prop_base, G)
    print()

    # ------------------------------------------------------------ variantes
    rows = []
    rows.append(eval_variant("base", X_base, Y, G, Xte_base, mask_holdout,
                              te_ids, lgbm_base))
    rows.append(eval_variant("invariant", X_inv, Y, G, Xte_inv, mask_holdout,
                              te_ids, lgbm_base))
    rows.append(eval_variant("base_invariant", X_union, Y, G, Xte_union,
                              mask_holdout, te_ids, lgbm_base))
    rows.append(eval_variant("base_regularized", X_base, Y, G, Xte_base,
                              mask_holdout, te_ids, lgbm_regularized))
    rows.append(eval_variant("union_regularized", X_union, Y, G, Xte_union,
                              mask_holdout, te_ids, lgbm_regularized))

    # ------------------------------------------------------------ synthèse
    df = pd.DataFrame(rows).sort_values("holdout_score", ascending=False)
    print("=" * 70)
    print("SYNTHÈSE (triée par holdout_score -- la métrique de décision)")
    print("=" * 70)
    print(df.to_string(index=False))

    best = df.iloc[0]
    print(f"\nRéférence exp014 (K=0) : holdout {HOLDOUT_BASELINE:.4f}, LB {LB_BEST}")
    print(f"Meilleure variante : {best['variant']} "
          f"(holdout {best['holdout_score']:.4f}, "
          f"delta {best['holdout_score'] - HOLDOUT_BASELINE:+.4f})")

    if best["holdout_score"] > HOLDOUT_BASELINE + 0.002:
        print(f"\n>> Gain net sur le holdout -> soumettre {best['submission']} "
              "(UNE seule soumission).")
    else:
        print("\n>> Aucune variante ne bat nettement la référence. Le plafond de "
              "cette approche est probablement atteint.")
        print("   Recommandation : figer exp014 K=0 comme pipeline final, "
              "consacrer le temps restant au bagging multi-seed (robustesse "
              "sur le LB privé) et au rapport Phase 2 (35 % de la note).")

    df.to_csv(MODELS_DIR / "exp015_invariant.csv", index=False)
    print(f"\nRésultats écrits : {MODELS_DIR / 'exp015_invariant.csv'}")


if __name__ == "__main__":
    main()

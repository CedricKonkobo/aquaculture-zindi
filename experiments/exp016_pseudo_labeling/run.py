"""
experiments/exp016_pseudo_labeling/run.py
----------------------------------------------
Auto-apprentissage transductif (pseudo-étiquetage).

Idée : le modèle a une AUC estimée à ~0.90 sur le test. Globalement
imparfaite, mais ses prédictions EXTRÊMES (p >= seuil ou p <= 1 - seuil) sont
très majoritairement correctes. On les traite comme étiquettes provisoires,
on les ajoute au jeu d'entraînement, et on réentraîne. Le modèle apprend
alors sur des données de la PÉRIODE DU TEST.

Pourquoi c'est la piste à plus fort levier : toutes les tentatives
précédentes cherchaient à améliorer un modèle entraîné sur la période A pour
la période B, sans jamais lui montrer la période B.
  exp012 (LB 0.778) transformer les features     -> échec
  exp013 (LB 0.805) recalibrer via spw           -> levier mort
  exp014 (LB 0.825) supprimer features décalées  -> indispensables
  exp015 (LB 0.8324) ajouter des features        -> gain réel, cause intacte

Légalité : seules les FEATURES du test sont utilisées, plus les prédictions
du modèle lui-même. Les vraies étiquettes du test ne sont jamais employées.
Le seuil de décision reste 0.5. Conforme au règlement.

Usage :
    python -m experiments.exp016_pseudo_labeling.run
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

CONF_GRID = [0.90, 0.95, 0.99]
PSEUDO_WEIGHT = 0.5
N_AUG = 10
N_SPLITS = 5
SPW = 1.6
HOLDOUT_FRACTION = 0.25

HOLDOUT_BASELINE = 0.9524    # exp015 base_invariant
LB_BEST = 0.8324             # exp015 base_invariant


def lgbm_factory():
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=1200, learning_rate=0.03,
        num_leaves=31, min_child_samples=30, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.5, lambda_l2=5.0,
        scale_pos_weight=SPW, verbose=-1, seed=SEED,
    )


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
    return X, Y, G, Xte, te_ids


def testlike_holdout(X, Xte, G):
    """Holdout test-like (identique à exp014/exp015) : découpage au niveau GROUPE."""
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
    print(f"AUC adversariale (contrôle) : {roc_auc_score(y_adv, oof):.4f}")

    dfp = pd.DataFrame({"group": G, "p": oof[:len(X)]})
    grp = dfp.groupby("group")["p"].mean().sort_values(ascending=False)
    n_hold = max(1, int(round(len(grp) * HOLDOUT_FRACTION)))
    mask = np.isin(G, list(grp.index[:n_hold]))
    print(f"Holdout test-like : {mask.sum()} lignes ({n_hold} groupes sur {len(grp)})")
    return mask


def select_pseudo(proba_test, conf):
    """
    Sélectionne les échantillons test dont la prédiction est extrême.
    Retourne (indices, pseudo_labels).
    """
    pos = np.where(proba_test >= conf)[0]
    neg = np.where(proba_test <= 1 - conf)[0]
    idx = np.concatenate([pos, neg])
    lab = np.concatenate([np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)])
    return idx, lab


def eval_holdout(X, Y, W, mask_holdout, X_extra=None, Y_extra=None, W_extra=None):
    """
    Entraîne sur (train hors holdout [+ pseudo-étiquettes]) et évalue sur le
    holdout test-like. Les pseudo-étiquettes ne sont JAMAIS dans le holdout :
    elles proviennent du test, le holdout provient du train.
    """
    Xtr = X.loc[~mask_holdout]
    Ytr = Y[~mask_holdout]
    Wtr = W[~mask_holdout]

    if X_extra is not None and len(X_extra):
        Xtr = pd.concat([Xtr, X_extra], ignore_index=True)
        Ytr = np.concatenate([Ytr, Y_extra])
        Wtr = np.concatenate([Wtr, W_extra])

    model = lgbm_factory()
    model.fit(Xtr, Ytr, sample_weight=Wtr)
    proba = model.predict_proba(X.loc[mask_holdout])[:, 1]

    auc = roc_auc_score(Y[mask_holdout], proba)
    f1 = f1_score(Y[mask_holdout], (proba > 0.5).astype(int))
    return weighted_score(f1, auc), auc, f1


def full_cv_predict(X, Y, G, Xte, X_extra=None, Y_extra=None, W_extra=None):
    """
    CV complète (pour la soumission). Les pseudo-étiquettes sont ajoutées au
    train de CHAQUE fold, avec des groupes distincts pour éviter toute
    confusion avec les groupes du train.
    """
    if X_extra is None or not len(X_extra):
        cv = cross_validate(lgbm_factory, X, Y, G, n_splits=N_SPLITS, threshold=0.5,
                             seed=SEED, verbose=False, fit_fn=default_fit_fn)
        return predict_proba_ensemble(cv["models"], Xte), cv["score"]

    X_aug = pd.concat([X, X_extra], ignore_index=True)
    Y_aug = np.concatenate([Y, Y_extra])
    G_aug = np.concatenate([G, np.arange(len(X_extra)) + G.max() + 1])

    cv = cross_validate(lgbm_factory, X_aug, Y_aug, G_aug, n_splits=N_SPLITS,
                         threshold=0.5, seed=SEED, verbose=False, fit_fn=default_fit_fn)
    return predict_proba_ensemble(cv["models"], Xte), cv["score"]


def main():
    X, Y, G, Xte, te_ids = build_data()
    W = np.ones(len(X))

    print("\n" + "=" * 70)
    print("HOLDOUT TEST-LIKE")
    print("=" * 70)
    mask_holdout = testlike_holdout(X, Xte, G)

    # ------------------------------------------------ tour 0 : référence
    print("\n" + "=" * 70)
    print("TOUR 0 -- référence sans pseudo-étiquetage (= exp015 base_invariant)")
    print("=" * 70)
    ho_score, ho_auc, ho_f1 = eval_holdout(X, Y, W, mask_holdout)
    print(f"HOLDOUT : AUC={ho_auc:.4f}  F1={ho_f1:.4f}  score={ho_score:.4f}  "
          f"(réf. exp015 = {HOLDOUT_BASELINE:.4f})")

    proba_test_0, cv0 = full_cv_predict(X, Y, G, Xte)
    print(f"CV std  : {cv0:.4f}   Taux positifs test : {(proba_test_0 > 0.5).mean():.3f}")

    rows = [{"conf": None, "round": 0, "n_pseudo": 0, "pseudo_pos_rate": None,
             "holdout_score": round(ho_score, 4), "cv_score": round(cv0, 4),
             "test_pos_rate": round(float((proba_test_0 > 0.5).mean()), 4),
             "submission": None}]

    # --------------------------------- tour 1 : balayage du seuil de confiance
    best = None
    for conf in CONF_GRID:
        print("\n" + "=" * 70)
        print(f"TOUR 1 -- seuil de confiance = {conf}")
        print("=" * 70)

        idx, lab = select_pseudo(proba_test_0, conf)
        if len(idx) == 0:
            print("Aucun échantillon assez confiant -- variante ignorée.")
            continue

        X_extra = Xte.iloc[idx].reset_index(drop=True)
        W_extra = np.full(len(idx), PSEUDO_WEIGHT)
        pos_rate = float(lab.mean())
        print(f"Pseudo-étiquettes : {len(idx)}/{len(Xte)} ({len(idx)/len(Xte):.1%} du test), "
              f"dont {pos_rate:.1%} de positifs")

        ho_s, ho_a, ho_f = eval_holdout(X, Y, W, mask_holdout, X_extra, lab, W_extra)
        print(f"HOLDOUT : AUC={ho_a:.4f}  F1={ho_f:.4f}  score={ho_s:.4f}  "
              f"(delta vs tour 0 : {ho_s - ho_score:+.4f})")

        proba_t, cv_s = full_cv_predict(X, Y, G, Xte, X_extra, lab, W_extra)
        fname = f"exp016_pl_t{str(conf).replace('.', 'p')}_r1_submission.csv"
        make_submission(te_ids, proba_t, threshold=0.5, filename=fname)

        rows.append({"conf": conf, "round": 1, "n_pseudo": len(idx),
                     "pseudo_pos_rate": round(pos_rate, 4),
                     "holdout_score": round(ho_s, 4), "cv_score": round(cv_s, 4),
                     "test_pos_rate": round(float((proba_t > 0.5).mean()), 4),
                     "submission": fname})

        if best is None or ho_s > best["ho_score"]:
            best = {"conf": conf, "ho_score": ho_s, "proba": proba_t}

    # ---------------------------------- tour 2 : uniquement le meilleur seuil
    if best is not None:
        conf = best["conf"]
        print("\n" + "=" * 70)
        print(f"TOUR 2 -- seuil {conf}, pseudo-étiquettes recalculées sur le tour 1")
        print("=" * 70)

        idx, lab = select_pseudo(best["proba"], conf)
        if len(idx):
            X_extra = Xte.iloc[idx].reset_index(drop=True)
            W_extra = np.full(len(idx), PSEUDO_WEIGHT)
            print(f"Pseudo-étiquettes : {len(idx)}/{len(Xte)} "
                  f"({len(idx)/len(Xte):.1%}), dont {lab.mean():.1%} de positifs")

            ho_s, ho_a, ho_f = eval_holdout(X, Y, W, mask_holdout, X_extra, lab, W_extra)
            print(f"HOLDOUT : AUC={ho_a:.4f}  F1={ho_f:.4f}  score={ho_s:.4f}  "
                  f"(delta vs tour 0 : {ho_s - ho_score:+.4f})")

            proba_t, cv_s = full_cv_predict(X, Y, G, Xte, X_extra, lab, W_extra)
            fname = f"exp016_pl_t{str(conf).replace('.', 'p')}_r2_submission.csv"
            make_submission(te_ids, proba_t, threshold=0.5, filename=fname)

            rows.append({"conf": conf, "round": 2, "n_pseudo": len(idx),
                         "pseudo_pos_rate": round(float(lab.mean()), 4),
                         "holdout_score": round(ho_s, 4), "cv_score": round(cv_s, 4),
                         "test_pos_rate": round(float((proba_t > 0.5).mean()), 4),
                         "submission": fname})

    # ------------------------------------------------------------ synthèse
    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("SYNTHÈSE")
    print("=" * 70)
    print(df.to_string(index=False))

    print(f"\nRéférence exp015 : holdout {HOLDOUT_BASELINE:.4f} | LB {LB_BEST}")
    print("\n⚠️  LECTURE DU HOLDOUT : il est construit à partir du TRAIN, alors que")
    print("   le bénéfice du pseudo-étiquetage porte sur le domaine du TEST. Il")
    print("   SOUS-ESTIME donc probablement le gain réel. Son rôle ici est un")
    print("   GARDE-FOU : une chute nette signale une amplification d'erreurs.")
    print("   Un holdout stable est un bon signe, pas un signe d'absence d'effet.")

    cand = df[df["round"] > 0]
    if len(cand):
        stable = cand[cand["holdout_score"] >= HOLDOUT_BASELINE - 0.005]
        if len(stable):
            pick = stable.loc[stable["n_pseudo"].idxmax()]
            print(f"\n-> Soumettre en priorité : {pick['submission']} "
                  f"(seuil {pick['conf']}, tour {int(pick['round'])}, "
                  f"{int(pick['n_pseudo'])} pseudo-étiquettes)")
            print("   Choix : parmi les variantes SANS dégradation du holdout, celle")
            print("   qui exploite le PLUS d'information du domaine test.")
            print("   Puis 1 soumission de contrôle avec un seuil plus strict.")
        else:
            print("\n-> Toutes les variantes dégradent le holdout : amplification")
            print("   d'erreurs probable. Ne pas soumettre, revenir à exp015.")

    df.to_csv(MODELS_DIR / "exp016_pseudo_labeling.csv", index=False)
    print(f"\nRésultats écrits : {MODELS_DIR / 'exp016_pseudo_labeling.csv'}")


if __name__ == "__main__":
    main()

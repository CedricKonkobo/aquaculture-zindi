"""
experiments/exp019_consolidation/run.py
--------------------------------------------
Cumul des gains validés + troisième famille de modèles.

exp018 : blend GBDT + CNN -> LB 0.8365 -> 0.8591 (+0.0226), plus gros gain du
projet. Mécanisme : les deux modèles se DÉGRADENT DIFFÉREMMENT sous le
décalage de domaine (corrélation 0.97 sur l'OOF, mais 0.87 sur le TEST).
C'est la corrélation TEST qui compte, puisque c'est là que le blend opère.

Cette expérience combine :
  1) le PSEUDO-ÉTIQUETAGE, absent du 0.8591 (retiré à exp018 pour isoler le
     CNN) -- gain attendu additif, +0.0041 à exp016 ;
  2) le BAGGING MULTI-SEED des réseaux (un seul seed par fold à exp018) ;
  3) une TROISIÈME FAMILLE, GRU bidirectionnel, sélectionnée sur sa
     corrélation TEST avec les deux autres ;
  4) une recherche des poids de blend sur le simplexe.

ATTENTION AU TEMPS DE CALCUL : 5 folds x (1 GBDT + 5 CNN + 5 GRU) x 2 passes
= 110 entraînements. Plusieurs heures sur CPU. Lancer en arrière-plan.

Usage :
    python -m experiments.exp019_consolidation.run
"""
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score

from src.preprocessing import load_train, load_test
from src.augmentation import build_augmented_replicas
from src.features import featurize_many, featurize
from src.features_invariant import featurize_invariant_many, featurize_invariant
from src.temporal_cnn import (cube_to_tensor, train_cnn, predict_cnn,
                               TemporalCNN, TemporalGRU)
from src.train import weighted_score
from src.inference import make_submission
from src.utils import SEED, MODELS_DIR

N_AUG = 10
N_SPLITS = 5
SPW = 1.6
N_SEEDS_NN = 5
CNN_EPOCHS = 30
PSEUDO_CONF = 0.90
PSEUDO_WEIGHT = 0.5
BLEND_STEP = 0.1

HOLDOUT_PATH = MODELS_DIR / "exp017_fixed_holdout_groups.csv"
LB_EXP018 = 0.8591


def f_lgbm():
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=1200, learning_rate=0.03,
        num_leaves=31, min_child_samples=30, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.5, lambda_l2=5.0,
        scale_pos_weight=SPW, verbose=-1, seed=SEED,
    )


def score_of(y, proba):
    auc = roc_auc_score(y, proba)
    f1 = f1_score(y, (proba > 0.5).astype(int))
    return weighted_score(f1, auc), auc, f1


def bagged_nn(model_cls, Xt_tr, y_tr, w_tr, Xt_va, Xt_te, n_channels,
               hidden=32, dropout=0.3, n_seeds=N_SEEDS_NN):
    """
    Bagging multi-seed d'un réseau : moyenne des probabilités sur n_seeds
    initialisations. Les réseaux étant bien plus sensibles à l'initialisation
    que les arbres, c'est à la fois un gain de score et une réduction de
    variance (utile pour le classement privé).
    """
    p_va, p_te = [], []
    for s in range(n_seeds):
        m = train_cnn(Xt_tr, y_tr, w_tr=w_tr, n_channels=n_channels,
                       epochs=CNN_EPOCHS, hidden=hidden, dropout=dropout,
                       pos_weight=SPW, seed=SEED + s, model_cls=model_cls)
        p_va.append(predict_cnn(m, Xt_va))
        p_te.append(predict_cnn(m, Xt_te))
    return np.mean(p_va, axis=0), np.mean(p_te, axis=0)


def run_all_families(X, Y, G, Xte, Xt_all, Xt_te, n_channels,
                      pseudo=None, label=""):
    """
    OOF + prédictions test des 3 familles sur EXACTEMENT les mêmes folds.

    pseudo : None, ou dict {"X_feat", "X_tensor", "y", "w"} -- les
    pseudo-étiquettes sont ajoutées au TRAIN de chaque fold, jamais à la
    validation.
    """
    print(f"\n--- Entraînement des 3 familles [{label}] ---")
    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = {k: np.zeros(len(X)) for k in ("gbdt", "cnn", "gru")}
    te = {k: [] for k in ("gbdt", "cnn", "gru")}

    for fold, (tri, vai) in enumerate(skf.split(X, Y, groups=G)):
        X_tr, y_tr = X.iloc[tri], Y[tri]
        Xt_tr = Xt_all[tri]
        w_tr = np.ones(len(tri))

        if pseudo is not None:
            X_tr = pd.concat([X_tr, pseudo["X_feat"]], ignore_index=True)
            y_tr = np.concatenate([y_tr, pseudo["y"]])
            Xt_tr = np.concatenate([Xt_tr.numpy(), pseudo["X_tensor"].numpy()])
            import torch
            Xt_tr = torch.from_numpy(Xt_tr)
            w_tr = np.concatenate([w_tr, np.full(len(pseudo["y"]), pseudo["w"])])

        m = f_lgbm()
        m.fit(X_tr, y_tr, sample_weight=w_tr)
        oof["gbdt"][vai] = m.predict_proba(X.iloc[vai])[:, 1]
        te["gbdt"].append(m.predict_proba(Xte)[:, 1])

        for key, cls in (("cnn", TemporalCNN), ("gru", TemporalGRU)):
            p_va, p_te = bagged_nn(cls, Xt_tr, y_tr, w_tr, Xt_all[vai], Xt_te,
                                    n_channels)
            oof[key][vai] = p_va
            te[key].append(p_te)

        print(f"  fold {fold} terminé")

    te = {k: np.mean(v, axis=0) for k, v in te.items()}
    return oof, te


def optimize_blend(oof, Y, mask_ho, step=BLEND_STEP):
    """
    Recherche des poids sur le simplexe (gbdt, cnn, gru), jugée sur le
    holdout. Rappel : le holdout SOUS-ESTIME les gains de robustesse au
    décalage -- il sert ici à écarter les combinaisons mauvaises, pas à
    estimer l'amplitude du gain.
    """
    keys = ["gbdt", "cnn", "gru"]
    grid = [round(i * step, 3) for i in range(int(1 / step) + 1)]
    best = (None, -1.0)
    for w1, w2 in itertools.product(grid, repeat=2):
        w3 = round(1.0 - w1 - w2, 3)
        if w3 < -1e-9:
            continue
        w = {"gbdt": w1, "cnn": w2, "gru": max(w3, 0.0)}
        blend = sum(w[k] * oof[k] for k in keys)
        s, _, _ = score_of(Y[mask_ho], blend[mask_ho])
        if s > best[1]:
            best = (w, s)
    return best


def main():
    import torch

    # -------------------------------------------------------------- données
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

    cube_aug = np.concatenate(cubes, axis=0)
    Xt_all, mu, sd = cube_to_tensor(cube_aug)
    Xt_te, _, _ = cube_to_tensor(te_cube, mu, sd)
    n_channels = Xt_all.shape[1]
    print(f"Features : {X.shape} | Tenseur : {tuple(Xt_all.shape)} | Test : {Xte.shape}")

    if not HOLDOUT_PATH.exists():
        raise FileNotFoundError(f"{HOLDOUT_PATH} introuvable -- lance exp017 d'abord.")
    holdout_groups = set(pd.read_csv(HOLDOUT_PATH)["group"].tolist())
    mask_ho = np.isin(G, list(holdout_groups))
    print(f"Holdout figé : {mask_ho.sum()} lignes\n")

    # ------------------------- ÉTAPE 1 : 3 familles, sans pseudo-étiquetage
    print("=" * 70)
    print("ÉTAPE 1 -- trois familles, sans pseudo-étiquetage")
    print("=" * 70)
    oof, te = run_all_families(X, Y, G, Xte, Xt_all, Xt_te, n_channels,
                                pseudo=None, label="sans pseudo")

    for k in ("gbdt", "cnn", "gru"):
        s, auc, f1 = score_of(Y, oof[k])
        print(f"{k:5s} OOF : AUC={auc:.4f} F1={f1:.4f} score={s:.4f}")

    # ------------------------- ÉTAPE 2 : corrélations TEST (le bon critère)
    print("\n" + "=" * 70)
    print("ÉTAPE 2 -- corrélations (critère : le TEST, pas l'OOF)")
    print("=" * 70)
    pairs = [("gbdt", "cnn"), ("gbdt", "gru"), ("cnn", "gru")]
    corr_rows = []
    for a, b in pairs:
        c_oof = float(np.corrcoef(oof[a], oof[b])[0, 1])
        c_te = float(np.corrcoef(te[a], te[b])[0, 1])
        print(f"{a}/{b} : OOF={c_oof:.4f}   TEST={c_te:.4f}")
        corr_rows.append({"pair": f"{a}/{b}", "corr_oof": round(c_oof, 4),
                          "corr_test": round(c_te, 4)})
    pd.DataFrame(corr_rows).to_csv(MODELS_DIR / "exp019_test_correlations.csv",
                                    index=False)

    c_cnn_gru = [r for r in corr_rows if r["pair"] == "cnn/gru"][0]["corr_test"]
    if c_cnn_gru > 0.95:
        print(">> CNN et GRU sont redondants sur le test : le GRU n'apporte "
              "probablement rien. Vérifier que son poids optimal est faible.")
    else:
        print(">> CNN et GRU sont suffisamment décorrélés sur le test : le GRU "
              "est un apport potentiellement réel.")

    # ------------------------- ÉTAPE 3 : poids de blend
    print("\n" + "=" * 70)
    print("ÉTAPE 3 -- poids de blend (sur le holdout)")
    print("=" * 70)
    w_best, ho_best = optimize_blend(oof, Y, mask_ho)
    print(f"Poids retenus : {w_best}  (holdout {ho_best:.4f})")

    blend_te_nopl = sum(w_best[k] * te[k] for k in w_best)
    make_submission(te_ids, blend_te_nopl, threshold=0.5,
                     filename="exp019_blend_nopl_submission.csv")

    s_gbdt_ho, _, _ = score_of(Y[mask_ho], oof["gbdt"][mask_ho])
    print(f"GBDT seul sur le holdout : {s_gbdt_ho:.4f}  "
          f"(delta blend : {ho_best - s_gbdt_ho:+.4f})")

    rows = [{"config": "blend_nopl", "weights": str(w_best),
             "holdout_score": round(ho_best, 4),
             "test_pos_rate": round(float((blend_te_nopl > 0.5).mean()), 4),
             "submission": "exp019_blend_nopl_submission.csv"}]

    # ------------------------- ÉTAPE 4 : pseudo-étiquetage sur le blend
    print("\n" + "=" * 70)
    print("ÉTAPE 4 -- pseudo-étiquetage appliqué au blend")
    print("=" * 70)
    pos = np.where(blend_te_nopl >= PSEUDO_CONF)[0]
    neg = np.where(blend_te_nopl <= 1 - PSEUDO_CONF)[0]
    idx = np.concatenate([pos, neg])
    lab = np.concatenate([np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)])
    print(f"Pseudo-étiquettes : {len(idx)}/{len(Xte)} ({len(idx)/len(Xte):.1%}), "
          f"dont {lab.mean():.1%} de positifs")

    if len(idx):
        pseudo = {"X_feat": Xte.iloc[idx].reset_index(drop=True),
                  "X_tensor": Xt_te[idx], "y": lab, "w": PSEUDO_WEIGHT}
        oof_pl, te_pl = run_all_families(X, Y, G, Xte, Xt_all, Xt_te, n_channels,
                                          pseudo=pseudo, label="avec pseudo")

        w_pl, ho_pl = optimize_blend(oof_pl, Y, mask_ho)
        print(f"Poids retenus : {w_pl}  (holdout {ho_pl:.4f})")

        blend_te_pl = sum(w_pl[k] * te_pl[k] for k in w_pl)
        make_submission(te_ids, blend_te_pl, threshold=0.5,
                         filename="exp019_blend_pl_submission.csv")

        rows.append({"config": "blend_pl", "weights": str(w_pl),
                     "holdout_score": round(ho_pl, 4),
                     "test_pos_rate": round(float((blend_te_pl > 0.5).mean()), 4),
                     "submission": "exp019_blend_pl_submission.csv"})

    # ------------------------------------------------------------ synthèse
    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("SYNTHÈSE")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nLB de référence (exp018, GBDT+CNN sans pseudo) : {LB_EXP018}")

    print("\n⚠️  RAPPEL : le holdout SOUS-ESTIME les gains de robustesse au")
    print("   décalage (exp018 : +0.0021 annoncé pour +0.0226 réel, facteur ~11).")
    print("   Il sert de garde-fou, pas d'estimateur d'amplitude.")
    print("\n-> Soumettre exp019_blend_pl_submission.csv en priorité (cumul des")
    print("   deux gains), puis exp019_blend_nopl_submission.csv en contrôle")
    print("   si le premier déçoit. Budget : 2 soumissions.")

    df.to_csv(MODELS_DIR / "exp019_consolidation.csv", index=False)
    print(f"\nRésultats écrits : {MODELS_DIR / 'exp019_consolidation.csv'}")


if __name__ == "__main__":
    main()

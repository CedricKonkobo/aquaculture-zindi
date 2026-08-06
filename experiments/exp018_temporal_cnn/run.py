"""
experiments/exp018_temporal_cnn/run.py
-------------------------------------------
Sortie de la famille des arbres.

exp017 a établi que le blend LightGBM + XGBoost + CatBoost n'apporte rien
(+0.0011 holdout, neutre au LB) : trois algorithmes à base d'arbres sur les
mêmes features produisent des prédictions trop corrélées. Le projet est sur
un plateau à ~0.836.

Ce qui a fonctionné deux fois est l'apport d'information RÉELLEMENT NOUVELLE.
Or l'ORDRE TEMPOREL est inexploité : toutes les features actuelles sont des
statistiques agrégées sur les mois. Un CNN 1D consomme la séquence brute.

CRITÈRE DÉCISIF : la CORRÉLATION entre les prédictions du CNN et celles du
LightGBM, plus que le score individuel du CNN.
  corrélation < 0.90 -> le CNN voit autre chose, le blend a un potentiel
  corrélation > 0.95 -> même impasse qu'exp017

Usage :
    python -m experiments.exp018_temporal_cnn.run
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score

from src.preprocessing import load_train, load_test
from src.augmentation import build_augmented_replicas
from src.features import featurize_many, featurize
from src.features_invariant import featurize_invariant_many, featurize_invariant
from src.temporal_cnn import cube_to_tensor, train_cnn, predict_cnn
from src.train import weighted_score
from src.inference import make_submission
from src.utils import SEED, MODELS_DIR

N_AUG = 10
N_SPLITS = 5
SPW = 1.6
PSEUDO_CONF = 0.90
PSEUDO_WEIGHT = 0.5      # exp017 : au-delà, le gain est un artefact de circularité
CNN_GRID = [(32, 0.3), (64, 0.3), (64, 0.5)]   # (hidden, dropout)
CNN_EPOCHS = 30
BLEND_W_CNN = [0.2, 0.3, 0.4, 0.5]

HOLDOUT_PATH = MODELS_DIR / "exp017_fixed_holdout_groups.csv"
LB_REF = 0.8365          # exp016


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


def main():
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

    # cube augmenté empilé, pour le CNN (même ordre que X)
    cube_aug = np.concatenate(cubes, axis=0)
    Xt_all, mu, sd = cube_to_tensor(cube_aug)
    Xt_te, _, _ = cube_to_tensor(te_cube, mu, sd)
    n_channels = Xt_all.shape[1]

    print(f"Features GBDT : {X.shape}  |  Tenseur CNN : {tuple(Xt_all.shape)}  "
          f"|  Test : {Xte.shape}")

    # ---------------------------------------------------- holdout figé
    if not HOLDOUT_PATH.exists():
        raise FileNotFoundError(
            f"{HOLDOUT_PATH} introuvable -- lance exp017 d'abord (il calcule et "
            "sauvegarde le holdout figé, indispensable pour comparer les "
            "expériences entre elles)."
        )
    holdout_groups = set(pd.read_csv(HOLDOUT_PATH)["group"].tolist())
    mask_ho = np.isin(G, list(holdout_groups))
    print(f"Holdout figé rechargé : {mask_ho.sum()} lignes "
          f"({len(holdout_groups)} groupes)\n")

    # ------------------------------------- 1) CNN : choix de l'architecture
    print("=" * 70)
    print("ÉTAPE 1 -- architecture du CNN (jugée sur le holdout figé)")
    print("=" * 70)
    best_cfg, best_ho = None, -1.0
    for hidden, dropout in CNN_GRID:
        model = train_cnn(Xt_all[~mask_ho], Y[~mask_ho], n_channels=n_channels,
                           epochs=CNN_EPOCHS, hidden=hidden, dropout=dropout,
                           pos_weight=SPW, seed=SEED)
        p_ho = predict_cnn(model, Xt_all[mask_ho])
        s, auc, f1 = score_of(Y[mask_ho], p_ho)
        print(f"hidden={hidden} dropout={dropout} -> holdout AUC={auc:.4f} "
              f"F1={f1:.4f} score={s:.4f}")
        if s > best_ho:
            best_ho, best_cfg = s, (hidden, dropout)
    hidden, dropout = best_cfg
    print(f"\nArchitecture retenue : hidden={hidden}, dropout={dropout} "
          f"(holdout {best_ho:.4f})\n")

    # ------------------------------- 2) OOF conjointes GBDT et CNN
    print("=" * 70)
    print("ÉTAPE 2 -- OOF conjointes sur les mêmes folds (GBDT et CNN)")
    print("=" * 70)
    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_gbdt = np.zeros(len(X))
    oof_cnn = np.zeros(len(X))
    te_gbdt, te_cnn = [], []

    for fold, (tri, vai) in enumerate(skf.split(X, Y, groups=G)):
        m = f_lgbm()
        m.fit(X.iloc[tri], Y[tri])
        oof_gbdt[vai] = m.predict_proba(X.iloc[vai])[:, 1]
        te_gbdt.append(m.predict_proba(Xte)[:, 1])

        c = train_cnn(Xt_all[tri], Y[tri], n_channels=n_channels, epochs=CNN_EPOCHS,
                       hidden=hidden, dropout=dropout, pos_weight=SPW, seed=SEED)
        oof_cnn[vai] = predict_cnn(c, Xt_all[vai])
        te_cnn.append(predict_cnn(c, Xt_te))
        print(f"fold {fold} terminé")

    proba_te_gbdt = np.mean(te_gbdt, axis=0)
    proba_te_cnn = np.mean(te_cnn, axis=0)

    s_g, auc_g, f1_g = score_of(Y, oof_gbdt)
    s_c, auc_c, f1_c = score_of(Y, oof_cnn)
    print(f"\nGBDT OOF : AUC={auc_g:.4f} F1={f1_g:.4f} score={s_g:.4f}")
    print(f"CNN  OOF : AUC={auc_c:.4f} F1={f1_c:.4f} score={s_c:.4f}")

    # -------------------------------------- 3) LE critère : décorrélation
    print("\n" + "=" * 70)
    print("ÉTAPE 3 -- CORRÉLATION DES PRÉDICTIONS (critère décisif)")
    print("=" * 70)
    corr_oof = float(np.corrcoef(oof_gbdt, oof_cnn)[0, 1])
    corr_test = float(np.corrcoef(proba_te_gbdt, proba_te_cnn)[0, 1])
    print(f"Corrélation GBDT/CNN sur les OOF  : {corr_oof:.4f}")
    print(f"Corrélation GBDT/CNN sur le test  : {corr_test:.4f}")

    if corr_oof < 0.90:
        print(">> DÉCORRÉLÉ : le CNN voit réellement autre chose. Le blend a un "
              "potentiel réel, même si le CNN seul est plus faible.")
    elif corr_oof > 0.95:
        print(">> TROP CORRÉLÉ : le CNN réapprend la même chose que les arbres. "
              "Même impasse qu'exp017 -> le plafond du jeu de données est "
              "probablement atteint avec les approches disponibles.")
    else:
        print(">> Décorrélation partielle : le blend peut apporter un gain "
              "modeste. À trancher sur le holdout.")

    pd.DataFrame([{"corr_oof": round(corr_oof, 4), "corr_test": round(corr_test, 4)}]) \
        .to_csv(MODELS_DIR / "exp018_prediction_correlation.csv", index=False)

    # ------------------------------------------- 4) blend GBDT + CNN
    print("\n" + "=" * 70)
    print("ÉTAPE 4 -- blend GBDT + CNN (holdout figé)")
    print("=" * 70)
    # Soumissions générées pour TOUTES les configs, y compris les modèles
    # seuls : n'importe laquelle peut ressortir en tête du holdout.
    make_submission(te_ids, proba_te_gbdt, threshold=0.5,
                     filename="exp018_gbdt_alone_submission.csv")
    make_submission(te_ids, proba_te_cnn, threshold=0.5,
                     filename="exp018_cnn_alone_submission.csv")

    rows = [
        {"config": "gbdt_alone", "w_cnn": 0.0,
         "holdout_score": round(score_of(Y[mask_ho], oof_gbdt[mask_ho])[0], 4),
         "oof_score": round(s_g, 4),
         "test_pos_rate": round(float((proba_te_gbdt > 0.5).mean()), 4),
         "submission": "exp018_gbdt_alone_submission.csv"},
        {"config": "cnn_alone", "w_cnn": 1.0,
         "holdout_score": round(score_of(Y[mask_ho], oof_cnn[mask_ho])[0], 4),
         "oof_score": round(s_c, 4),
         "test_pos_rate": round(float((proba_te_cnn > 0.5).mean()), 4),
         "submission": "exp018_cnn_alone_submission.csv"},
    ]

    for w in BLEND_W_CNN:
        blend_oof = (1 - w) * oof_gbdt + w * oof_cnn
        blend_te = (1 - w) * proba_te_gbdt + w * proba_te_cnn
        s_ho, _, _ = score_of(Y[mask_ho], blend_oof[mask_ho])
        s_oof, _, _ = score_of(Y, blend_oof)
        fname = f"exp018_blend_w{str(w).replace('.', 'p')}_submission.csv"
        make_submission(te_ids, blend_te, threshold=0.5, filename=fname)
        print(f"w_cnn={w} -> holdout={s_ho:.4f}  oof={s_oof:.4f}  "
              f"taux positifs={float((blend_te > 0.5).mean()):.3f}")
        rows.append({"config": f"blend_w{w}", "w_cnn": w,
                     "holdout_score": round(s_ho, 4), "oof_score": round(s_oof, 4),
                     "test_pos_rate": round(float((blend_te > 0.5).mean()), 4),
                     "submission": fname})

    # ------------------------------------------------------------ synthèse
    df = pd.DataFrame(rows).sort_values("holdout_score", ascending=False)
    print("\n" + "=" * 70)
    print("SYNTHÈSE (triée par holdout_score)")
    print("=" * 70)
    print(df.to_string(index=False))

    ref_ho = df[df["config"] == "gbdt_alone"]["holdout_score"].iloc[0]
    best = df.iloc[0]
    print(f"\nRéférence GBDT seul (ce holdout) : {ref_ho:.4f}")
    print(f"Meilleure config : {best['config']} (holdout {best['holdout_score']:.4f}, "
          f"delta {best['holdout_score'] - ref_ho:+.4f})")
    print(f"LB de référence (exp016) : {LB_REF}")
    print(f"\nCorrélation GBDT/CNN : {corr_oof:.4f}")

    if best["config"] != "gbdt_alone" and best["holdout_score"] > ref_ho + 0.002:
        print(f"\n-> Soumettre {best['submission']} (UNE seule soumission).")
        print("   NB : ce blend n'inclut PAS le pseudo-étiquetage. S'il confirme")
        print("   au LB, l'étape suivante est de le combiner avec le")
        print("   pseudo-étiquetage d'exp016 (les deux gains sont indépendants).")
    else:
        print("\n-> Aucun gain net. Le plafond de ce jeu de données est")
        print("   probablement atteint avec les approches disponibles.")
        print("   Recommandation : figer exp016 (LB 0.8365), faire le bagging")
        print("   multi-seed pour sécuriser le classement privé, et consacrer")
        print("   l'effort restant au rapport de Phase 2.")

    df.to_csv(MODELS_DIR / "exp018_temporal_cnn.csv", index=False)
    print(f"\nRésultats écrits : {MODELS_DIR / 'exp018_temporal_cnn.csv'}")


if __name__ == "__main__":
    main()

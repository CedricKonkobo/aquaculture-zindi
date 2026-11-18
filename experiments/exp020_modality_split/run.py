"""
experiments/exp020_modality_split/run.py
---------------------------------------------
Diversité par la NATURE DE L'ENTRÉE plutôt que par l'architecture.

exp019 a livré la structure des corrélations (écart OOF -> TEST) :
    arbres vs convolution    : 0.9752 -> 0.8883  (-0.087)
    arbres vs récurrent      : 0.9635 -> 0.9016  (-0.062)
    convolution vs récurrent : 0.9813 -> 0.9492  (-0.032)
Plus les familles sont conceptuellement éloignées, plus elles se décorrèlent
sur le test. CNN et GRU partagent le MÊME tenseur : marge la plus faible.
Varier l'architecture donne des rendements décroissants.

JUSTIFICATION PHYSIQUE : Sentinel-1 (radar actif) traverse les nuages et est
peu sensible à la saison ; Sentinel-2 (optique passif) dépend fortement de la
phénologie, de l'angle solaire et de l'atmosphère. Les deux modalités ne se
dégradent PAS de la même façon quand la période change.

PRÉDICTION TESTABLE : la paire radar/optique doit montrer la décorrélation
TEST la plus forte mesurée jusqu'ici (référence : -0.087).

Le pseudo-étiquetage est ABANDONNÉ (exp019 : +0.0003 = bruit une fois le
blend en place).

Usage :
    python -m experiments.exp020_modality_split.run
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
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
from src.utils import SEED, MODELS_DIR, BANDS, S1_BANDS, S2_BANDS

N_AUG = 10
N_SPLITS = 5
SPW = 1.6
N_SEEDS_NN = 10          # exp019 : le bagging contribue nettement -> on double
CNN_EPOCHS = 30
GREEDY_ITERS = 20

HOLDOUT_PATH = MODELS_DIR / "exp017_fixed_holdout_groups.csv"
HOLDOUT_EXP019 = 0.9676
LB_EXP019 = 0.8732

# Séries relevant du radar (S1) : bandes brutes + indices dérivés
S1_SERIES_PREFIXES = ("VH_", "VV_", "VH_VV_diff_", "VH_VV_ratio_", "RVI_",
                       "tshape_VH", "tshape_VV", "tshape_RVI",
                       "selfz_VH", "selfz_VV", "selfz_RVI")
CONTEXT_FEATURES = ["n_s1", "n_s2", "frac_cloud", "water_frac", "ndvi_low_frac"]


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


def split_feature_columns(columns):
    """Sépare les colonnes en radar / optique (contexte attribué aux deux)."""
    s1_cols, s2_cols = [], []
    for c in columns:
        if c in CONTEXT_FEATURES:
            s1_cols.append(c)
            s2_cols.append(c)
        elif c.startswith(S1_SERIES_PREFIXES):
            s1_cols.append(c)
        else:
            s2_cols.append(c)
    return s1_cols, s2_cols


def subset_tensor(Xt, band_names):
    """
    Extrait les canaux correspondant à un sous-ensemble de bandes.

    cube_to_tensor produit 2*n_bands canaux : [valeurs normalisées] puis
    [masques], dans l'ordre de BANDS. On sélectionne donc l'indice de bande
    et son masque associé (décalé de len(BANDS)).
    """
    idx = [BANDS.index(b) for b in band_names]
    idx_full = idx + [i + len(BANDS) for i in idx]
    return Xt[:, idx_full, :]


def bagged_nn(model_cls, Xt_tr, y_tr, Xt_va, Xt_te, hidden=32, dropout=0.3,
               n_seeds=N_SEEDS_NN):
    """Bagging multi-seed : moyenne des probabilités sur n_seeds initialisations."""
    p_va, p_te = [], []
    for s in range(n_seeds):
        m = train_cnn(Xt_tr, y_tr, n_channels=Xt_tr.shape[1], epochs=CNN_EPOCHS,
                       hidden=hidden, dropout=dropout, pos_weight=SPW,
                       seed=SEED + s, model_cls=model_cls)
        p_va.append(predict_cnn(m, Xt_va))
        p_te.append(predict_cnn(m, Xt_te))
    return np.mean(p_va, axis=0), np.mean(p_te, axis=0)


def greedy_blend(oof, Y, mask_ho, n_iters=GREEDY_ITERS):
    """
    Sélection gloutonne ascendante avec répétition (méthode standard
    d'ensembling) : on part de rien et on ajoute itérativement le modèle qui
    améliore le plus le holdout. Les poids résultent des multiplicités.

    Préférée au grid exhaustif : avec 7 modèles, le simplexe est trop coûteux.
    """
    keys = list(oof.keys())
    selected, current, best_hist = [], None, []

    for _ in range(n_iters):
        best_key, best_s, best_blend = None, -1.0, None
        for k in keys:
            cand = oof[k] if current is None else (current * len(selected) + oof[k]) / (len(selected) + 1)
            s, _, _ = score_of(Y[mask_ho], cand[mask_ho])
            if s > best_s:
                best_key, best_s, best_blend = k, s, cand
        if best_hist and best_s <= best_hist[-1] + 1e-9:
            break     # plus d'amélioration
        selected.append(best_key)
        current = best_blend
        best_hist.append(best_s)

    weights = {k: selected.count(k) / len(selected) for k in keys if selected.count(k)}
    return weights, best_hist[-1] if best_hist else -1.0


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

    s1_cols, s2_cols = split_feature_columns(X.columns)
    print(f"Features : {X.shape[1]} total | radar {len(s1_cols)} | "
          f"optique {len(s2_cols)}")

    cube_aug = np.concatenate(cubes, axis=0)
    Xt_all, mu, sd = cube_to_tensor(cube_aug)
    Xt_te, _, _ = cube_to_tensor(te_cube, mu, sd)
    Xt_s1, Xt_te_s1 = subset_tensor(Xt_all, S1_BANDS), subset_tensor(Xt_te, S1_BANDS)
    Xt_s2, Xt_te_s2 = subset_tensor(Xt_all, S2_BANDS), subset_tensor(Xt_te, S2_BANDS)
    print(f"Tenseurs : complet {tuple(Xt_all.shape)} | radar {tuple(Xt_s1.shape)} "
          f"| optique {tuple(Xt_s2.shape)}")

    if not HOLDOUT_PATH.exists():
        raise FileNotFoundError(f"{HOLDOUT_PATH} introuvable -- lance exp017 d'abord.")
    holdout_groups = set(pd.read_csv(HOLDOUT_PATH)["group"].tolist())
    mask_ho = np.isin(G, list(holdout_groups))
    print(f"Holdout figé : {mask_ho.sum()} lignes\n")

    # ----------------------------------------- ÉTAPE 1 : les 7 modèles
    print("=" * 70)
    print("ÉTAPE 1 -- sept modèles sur les mêmes folds")
    print("=" * 70)
    names = ["gbdt_all", "gbdt_s1", "gbdt_s2", "cnn_all", "cnn_s1", "cnn_s2", "gru_all"]
    oof = {k: np.zeros(len(X)) for k in names}
    te = {k: [] for k in names}

    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (tri, vai) in enumerate(skf.split(X, Y, groups=G)):
        y_tr = Y[tri]

        for key, cols in (("gbdt_all", list(X.columns)), ("gbdt_s1", s1_cols),
                          ("gbdt_s2", s2_cols)):
            m = f_lgbm()
            m.fit(X.iloc[tri][cols], y_tr)
            oof[key][vai] = m.predict_proba(X.iloc[vai][cols])[:, 1]
            te[key].append(m.predict_proba(Xte[cols])[:, 1])

        for key, cls, Xt, Xt_t in (
            ("cnn_all", TemporalCNN, Xt_all, Xt_te),
            ("cnn_s1", TemporalCNN, Xt_s1, Xt_te_s1),
            ("cnn_s2", TemporalCNN, Xt_s2, Xt_te_s2),
            ("gru_all", TemporalGRU, Xt_all, Xt_te),
        ):
            p_va, p_te = bagged_nn(cls, Xt[tri], y_tr, Xt[vai], Xt_t)
            oof[key][vai] = p_va
            te[key].append(p_te)

        print(f"  fold {fold} terminé")

    te = {k: np.mean(v, axis=0) for k, v in te.items()}

    print("\nScores OOF individuels :")
    for k in names:
        s, auc, f1 = score_of(Y, oof[k])
        print(f"  {k:9s} AUC={auc:.4f} F1={f1:.4f} score={s:.4f}")

    # ------------------------- ÉTAPE 2 : corrélations, la prédiction testée
    print("\n" + "=" * 70)
    print("ÉTAPE 2 -- corrélations OOF vs TEST")
    print("=" * 70)
    corr_rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c_oof = float(np.corrcoef(oof[a], oof[b])[0, 1])
            c_te = float(np.corrcoef(te[a], te[b])[0, 1])
            corr_rows.append({"pair": f"{a}/{b}", "corr_oof": round(c_oof, 4),
                              "corr_test": round(c_te, 4),
                              "gap": round(c_te - c_oof, 4)})
    dfc = pd.DataFrame(corr_rows).sort_values("gap")
    print(dfc.to_string(index=False))
    dfc.to_csv(MODELS_DIR / "exp020_correlations.csv", index=False)

    print("\nRéférence exp019 : écart maximal -0.087 (gbdt/cnn)")
    modality_pairs = dfc[dfc["pair"].str.contains("s1") & dfc["pair"].str.contains("s2")]
    if len(modality_pairs):
        best_mod = modality_pairs.iloc[0]
        print(f"Meilleure paire radar/optique : {best_mod['pair']} "
              f"(écart {best_mod['gap']:+.4f})")
        if best_mod["gap"] < -0.087:
            print(">> PRÉDICTION VALIDÉE : la séparation par modalité produit la "
                  "décorrélation la plus forte mesurée jusqu'ici.")
        else:
            print(">> Prédiction NON validée : la séparation par modalité ne "
                  "décorrèle pas davantage que la séparation par architecture.")

    # ----------------------------------------- ÉTAPE 3 : blends
    print("\n" + "=" * 70)
    print("ÉTAPE 3 -- recherche des poids")
    print("=" * 70)
    w_greedy, ho_greedy = greedy_blend(oof, Y, mask_ho)
    print(f"Blend glouton : {w_greedy}")
    print(f"  holdout = {ho_greedy:.4f}  (exp019 : {HOLDOUT_EXP019:.4f})")

    blend_te_greedy = sum(w * te[k] for k, w in w_greedy.items())
    make_submission(te_ids, blend_te_greedy, threshold=0.5,
                     filename="exp020_blend_greedy_submission.csv")

    # contrôle : poids égaux sur les 7 modèles (plus robuste au surajustement
    # du holdout que la sélection gloutonne)
    blend_oof_eq = np.mean([oof[k] for k in names], axis=0)
    blend_te_eq = np.mean([te[k] for k in names], axis=0)
    ho_eq, _, _ = score_of(Y[mask_ho], blend_oof_eq[mask_ho])
    print(f"Blend à poids égaux (7 modèles) : holdout = {ho_eq:.4f}")
    make_submission(te_ids, blend_te_eq, threshold=0.5,
                     filename="exp020_blend_equal7_submission.csv")

    # ------------------------------------------------------------ synthèse
    rows = [
        {"config": "blend_greedy", "weights": str(w_greedy),
         "holdout_score": round(ho_greedy, 4),
         "test_pos_rate": round(float((blend_te_greedy > 0.5).mean()), 4),
         "submission": "exp020_blend_greedy_submission.csv"},
        {"config": "blend_equal7", "weights": "1/7 chacun",
         "holdout_score": round(ho_eq, 4),
         "test_pos_rate": round(float((blend_te_eq > 0.5).mean()), 4),
         "submission": "exp020_blend_equal7_submission.csv"},
    ]
    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("SYNTHÈSE")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nRéférence exp019 : holdout {HOLDOUT_EXP019:.4f}, LB {LB_EXP019}")

    print("\n⚠️  Le holdout sous-estime les gains de robustesse au décalage.")
    print("   Il sert de garde-fou, pas d'estimateur d'amplitude.")
    print("\n-> Soumettre les DEUX (budget 2) : le blend glouton peut sur-ajuster")
    print("   le holdout, le blend à poids égaux est plus conservateur. Leur")
    print("   comparaison au LB indiquera lequel des deux régimes privilégier")
    print("   pour la soumission finale.")

    df.to_csv(MODELS_DIR / "exp020_modality.csv", index=False)
    print(f"\nRésultats écrits : {MODELS_DIR / 'exp020_modality.csv'}")


if __name__ == "__main__":
    main()

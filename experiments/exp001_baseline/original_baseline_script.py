"""
Zindi - GeoAI Aquaculture Pond Identification
Baseline robuste au decalage temporel (train = 12 mois complets, test = fenetre de 4-6 mois).

Idees cles :
1) AUGMENTATION PAR MASQUAGE : on recree artificiellement les conditions du test dans le train
   (fenetre consecutive de 4/5/6 mois + trous nuageux sur S2).
2) FEATURES INVARIANTES A LA FENETRE : statistiques calculees uniquement sur les mois valides
   (mean/std/min/max/median/p25/p75) + indices spectraux normalises (NDWI, MNDWI, NDVI, ratios S1)
   qui resistent aux changements de saison/annee.
3) VALIDATION HONNETE : GroupKFold sur l'ID d'origine, validation faite sur des repliques masquees.

Sortie : submission.csv (ID, TargetF1, TargetRAUC), seuil fixe a 0.5.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb

SEED = 42
rng = np.random.default_rng(SEED)
 
MONTHS = [f"{i:02d}" for i in range(1, 13)]
S1_BANDS = ["VH", "VV"]
S2_BANDS = ["blue", "green", "red", "re1", "re2", "re3", "nir", "nira", "swir1", "swir2"]
BANDS = S1_BANDS + S2_BANDS
NA = -9999

# Distribution des trous S2 (nuages) observee dans le test
S2_GAP_P = np.array([0.735, 0.224, 0.036, 0.005])


# ----------------------------------------------------------------------------- data
def load(path, has_label=True):
    df = pd.read_csv(path)
    ids = df["ID"].values
    y = df["label"].values if has_label else None
    # cube (n, 12 mois, 12 bandes)
    cube = np.stack(
        [np.stack([df[f"{b}_{m}"].values for b in BANDS], axis=1) for m in MONTHS], axis=1
    ).astype(np.float32)
    cube[cube == NA] = np.nan
    return ids, y, cube


# --------------------------------------------------------------------- augmentation
def mask_window(cube, rng):
    """Applique une fenetre consecutive de 4-6 mois + trous S2, comme dans le test."""
    n = cube.shape[0]
    out = np.full_like(cube, np.nan)
    for i in range(n):
        L = rng.integers(4, 7)                      # longueur 4, 5 ou 6
        s = rng.integers(0, 12 - L + 1)             # debut de fenetre
        out[i, s:s + L] = cube[i, s:s + L]
        g = rng.choice(len(S2_GAP_P), p=S2_GAP_P)   # nb de mois sans optique
        if g > 0:
            drop = rng.choice(np.arange(s, s + L), size=min(g, L), replace=False)
            out[i, drop, len(S1_BANDS):] = np.nan   # on retire seulement S2
    return out


# ------------------------------------------------------------------------- features
def safe_ratio(a, b):
    d = a + b
    return np.where(np.abs(d) < 1e-6, np.nan, (a - b) / np.where(np.abs(d) < 1e-6, 1, d))


def build_series(cube):
    """Construit les series temporelles derivees (bandes brutes + indices)."""
    b = {name: cube[:, :, i] for i, name in enumerate(BANDS)}
    s = dict(b)
    # --- S1 (radar) : invariant aux nuages, present partout
    s["VH_VV_diff"] = b["VH"] - b["VV"]
    s["VH_VV_ratio"] = safe_ratio(b["VH"], b["VV"])
    s["RVI"] = 4 * b["VH"] / np.where(np.abs(b["VH"] + b["VV"]) < 1e-6, np.nan, b["VH"] + b["VV"])
    # --- S2 (optique) : indices normalises -> robustes au changement de periode
    s["NDVI"] = safe_ratio(b["nir"], b["red"])
    s["NDWI"] = safe_ratio(b["green"], b["nir"])      # eau libre
    s["MNDWI"] = safe_ratio(b["green"], b["swir1"])   # eau vs bati (cle pour les bassins)
    s["NDMI"] = safe_ratio(b["nir"], b["swir1"])
    s["NDWI2"] = safe_ratio(b["nir"], b["swir2"])
    s["AWEI"] = 4 * (b["green"] - b["swir1"]) - (0.25 * b["nir"] + 2.75 * b["swir2"])
    s["NDRE"] = safe_ratio(b["nira"], b["re1"])
    s["BSI"] = safe_ratio(b["swir1"] + b["red"], b["nir"] + b["blue"])
    s["brightness"] = np.nanmean(
        np.stack([b[x] for x in ["blue", "green", "red", "nir"]], -1), axis=-1
    )
    return s


def agg(x):
    """Statistiques sur les mois valides uniquement -> invariantes a la fenetre."""
    with np.errstate(all="ignore"):
        return {
            "mean": np.nanmean(x, 1), "std": np.nanstd(x, 1),
            "min": np.nanmin(x, 1), "max": np.nanmax(x, 1),
            "med": np.nanmedian(x, 1),
            "p25": np.nanpercentile(x, 25, axis=1), "p75": np.nanpercentile(x, 75, axis=1),
        }


def featurize(cube):
    s = build_series(cube)
    feats = {}
    for name, x in s.items():
        if np.all(np.isnan(x)):
            continue
        for k, v in agg(x).items():
            feats[f"{name}_{k}"] = v
        feats[f"{name}_range"] = feats[f"{name}_max"] - feats[f"{name}_min"]
        feats[f"{name}_iqr"] = feats[f"{name}_p75"] - feats[f"{name}_p25"]
        # amplitude relative (robuste au niveau absolu -> resiste au changement d'annee)
        feats[f"{name}_cv"] = feats[f"{name}_std"] / (np.abs(feats[f"{name}_mean"]) + 1e-6)
    # contexte : combien de mois observes (le modele doit savoir ce qu'il voit)
    feats["n_s1"] = np.sum(~np.isnan(s["VH"]), axis=1).astype(float)
    feats["n_s2"] = np.sum(~np.isnan(s["blue"]), axis=1).astype(float)
    feats["frac_cloud"] = 1 - feats["n_s2"] / np.maximum(feats["n_s1"], 1)
    # eau permanente : fraction de mois ou l'indice d'eau est positif
    feats["water_frac"] = np.nanmean((s["MNDWI"] > 0).astype(float), axis=1)
    feats["ndvi_low_frac"] = np.nanmean((s["NDVI"] < 0.2).astype(float), axis=1)
    df = pd.DataFrame(feats).replace([np.inf, -np.inf], np.nan)
    return df


# ------------------------------------------------------------------------------ run
def main():
    tr_ids, y, tr_cube = load("Train.csv", True)
    te_ids, _, te_cube = load("Test.csv", False)

    # --- jeu d'entrainement augmente : N_AUG repliques masquees par echantillon
    N_AUG = 10
    Xs, ys, groups = [], [], []
    for r in range(N_AUG):
        rr = np.random.default_rng(SEED + r)
        Xs.append(featurize(mask_window(tr_cube, rr)))
        ys.append(y)
        groups.append(np.arange(len(y)))
    X = pd.concat(Xs, ignore_index=True)
    Y = np.concatenate(ys)
    G = np.concatenate(groups)
    Xte = featurize(te_cube)[X.columns]
    print(f"train augmente : {X.shape} | test : {Xte.shape}")

    params = dict(
        objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=40,
        feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
        lambda_l1=0.5, lambda_l2=5.0, n_estimators=1200, verbose=-1, seed=SEED,
    )

    oof = np.zeros(len(X))
    pte = np.zeros(len(Xte))
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for f, (tri, vai) in enumerate(skf.split(X, Y, groups=G)):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tri], Y[tri], eval_set=[(X.iloc[vai], Y[vai])],
              eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[vai] = m.predict_proba(X.iloc[vai])[:, 1]
        pte += m.predict_proba(Xte)[:, 1] / skf.get_n_splits()
        print(f"fold {f}  AUC={roc_auc_score(Y[vai], oof[vai]):.4f} "
              f"F1@0.5={f1_score(Y[vai], (oof[vai] > .5).astype(int)):.4f}")

    auc = roc_auc_score(Y, oof)
    f1 = f1_score(Y, (oof > .5).astype(int))
    print(f"\nOOF (sur repliques masquees, = conditions test)  AUC={auc:.4f}  F1={f1:.4f}  "
          f"score pondere={0.6 * f1 + 0.4 * auc:.4f}")

    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 15 features :\n", imp.head(15).to_string())

    sub = pd.DataFrame({
        "ID": te_ids,
        "TargetF1": (pte > 0.5).astype(int),   # seuil par defaut 0.5, comme exige
        "TargetRAUC": pte,
    })
    sub.to_csv("submission.csv", index=False)
    print(f"\nsubmission.csv ecrit | taux de positifs predit = {sub.TargetF1.mean():.3f}")


if __name__ == "__main__":
    main()

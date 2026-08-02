"""
experiments/exp011_adversarial_validation/run.py
-----------------------------------------------------
Diagnostic : mesure le décalage de distribution train/test qui explique le
gap entre l'OOF local (0.975) et le leaderboard (0.82).

exp010 a confirmé que la STRUCTURE des mois manquants est fidèlement
reproduite par l'augmentation. Le décalage restant porte donc sur les
VALEURS -- cohérent avec l'énoncé du challenge (période temporelle
différente entre train et test).

Point méthodologique clé : le train est AUGMENTÉ (mask_window) avant
featurisation, pour que les deux côtés aient la même structure de mois
manquants. Sans ça, le classifieur adversarial détecterait trivialement
"12 mois vs 4-6 mois" au lieu du vrai décalage de valeurs.

Usage :
    python -m experiments.exp011_adversarial_validation.run
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.preprocessing import load_train, load_test
from src.augmentation import build_augmented_replicas
from src.features import featurize_many, featurize
from src.utils import SEED, MODELS_DIR, FIGURES_DIR, S1_BANDS, S2_BANDS

N_AUG_TRAIN = 3
N_SPLITS = 5
TOP_K_REPORT = 25
WEIGHT_CLIP = 10.0   # plafond des poids de propension (cf. exp014)

# Séries considérées comme "normalisées" (invariantes au niveau absolu).
# Tout le reste (stats de bandes brutes) porte des niveaux absolus, donc
# potentiellement sensibles au changement de période.
NORMALIZED_SERIES = [
    "VH_VV_diff", "VH_VV_ratio", "RVI",
    "NDVI", "NDWI", "MNDWI", "NDMI", "NDWI2", "NDRE", "BSI",
]
CONTEXT_FEATURES = ["n_s1", "n_s2", "frac_cloud", "water_frac", "ndvi_low_frac"]


def build_adversarial_dataset():
    """
    Construit (X, y_adv) où y_adv = 0 pour les échantillons train (augmentés)
    et 1 pour les échantillons test (déjà masqués nativement).
    """
    tr_ids, y, tr_cube = load_train()
    te_ids, _, te_cube = load_test()

    # Le train doit passer par la MÊME structure de masquage que le test,
    # sinon le décalage détecté serait trivial (12 mois vs 4-6 mois).
    cubes, _, _ = build_augmented_replicas(tr_cube, y, n_aug=N_AUG_TRAIN, seed=SEED)
    X_train = featurize_many(cubes)
    X_test = featurize(te_cube)[X_train.columns]

    X = pd.concat([X_train, X_test], ignore_index=True)
    y_adv = np.concatenate([
        np.zeros(len(X_train), dtype=int),
        np.ones(len(X_test), dtype=int),
    ])
    n_train_rows = len(X_train)
    print(f"Dataset adversarial : {len(X_train)} lignes train (augmentées x{N_AUG_TRAIN}), "
          f"{len(X_test)} lignes test, {X.shape[1]} features")
    return X, y_adv, n_train_rows


def select_normalized_columns(columns):
    """Colonnes issues des séries normalisées + features de contexte."""
    keep = []
    for c in columns:
        if c in CONTEXT_FEATURES:
            keep.append(c)
            continue
        for s in NORMALIZED_SERIES:
            if c.startswith(s + "_"):
                keep.append(c)
                break
    return keep


def run_adversarial(X, y_adv, label, n_splits=N_SPLITS, return_oof=False):
    """
    CV stratifiée d'un classifieur train-vs-test. Retourne l'AUC adversariale
    et les importances moyennes.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    importances = np.zeros(X.shape[1])

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_adv)):
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=300, learning_rate=0.05,
            num_leaves=31, verbose=-1, seed=SEED,
        )
        model.fit(X.iloc[tr_idx], y_adv[tr_idx])
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        importances += model.feature_importances_ / n_splits

    auc = roc_auc_score(y_adv, oof)
    print(f"\n[{label}] AUC adversariale = {auc:.4f}  ({X.shape[1]} features)")

    imp = pd.DataFrame({"feature": X.columns, "adv_importance": importances}) \
        .sort_values("adv_importance", ascending=False).reset_index(drop=True)

    if return_oof:
        return auc, imp, oof
    return auc, imp


def interpret(auc):
    if auc < 0.6:
        return ("décalage FAIBLE -- les distributions sont proches. Le gap "
                "leaderboard s'explique surtout par le seuil 0.5 mal placé "
                "-> priorité à exp012 (scale_pos_weight).")
    if auc < 0.8:
        return ("décalage MODÉRÉ -- combiner exp012 (seuil via spw) et "
                "exp013 (features invariantes au temps).")
    return ("décalage SÉVÈRE -- exp013 (features invariantes) devient la "
            "priorité absolue, et exp014 (CV repondérée) est indispensable "
            "pour retrouver une validation locale fiable.")


def main():
    X, y_adv, n_train_rows = build_adversarial_dataset()

    # ------------------------------------------- run 1 : toutes les features
    print("\n" + "=" * 70)
    print("RUN 1 -- toutes les features")
    print("=" * 70)
    auc_all, imp_all, oof_all = run_adversarial(X, y_adv, "all_features", return_oof=True)

    print(f"\nTop {TOP_K_REPORT} features qui SÉPARENT le plus train et test :")
    print(imp_all.head(TOP_K_REPORT).to_string(index=False))

    # ------------------------------ run 2 : indices normalisés uniquement
    print("\n" + "=" * 70)
    print("RUN 2 -- indices normalisés + contexte uniquement")
    print("=" * 70)
    norm_cols = select_normalized_columns(X.columns)
    auc_norm, imp_norm = run_adversarial(X[norm_cols], y_adv, "normalized_only")

    # ---------------------------------------------------------- synthèse
    print("\n" + "=" * 70)
    print("SYNTHÈSE")
    print("=" * 70)
    print(f"AUC adversariale (toutes features)      : {auc_all:.4f}")
    print(f"AUC adversariale (indices normalisés)   : {auc_norm:.4f}")
    print(f"Delta                                   : {auc_norm - auc_all:+.4f}")
    print(f"\nInterprétation (toutes features) : {interpret(auc_all)}")

    if auc_norm < auc_all - 0.05:
        print("\n>> Les indices normalisés sont NETTEMENT moins décalés que les bandes "
              "brutes : cela confirme que le décalage est porté par les NIVEAUX "
              "ABSOLUS. Piste forte pour exp013 (privilégier/ne garder que les "
              "features normalisées).")
    elif auc_norm > auc_all - 0.02:
        print("\n>> Les indices normalisés sont AUSSI décalés que le reste : le "
              "décalage ne se règle pas par simple normalisation. Il faudra "
              "envisager une normalisation par échantillon, ou repondérer la CV "
              "(exp014).")

    # part des features brutes dans le top du classement adversarial
    raw_prefixes = tuple(b + "_" for b in S1_BANDS + S2_BANDS)
    top = imp_all.head(TOP_K_REPORT)
    n_raw_in_top = top["feature"].str.startswith(raw_prefixes).sum()
    print(f"\nFeatures de bandes BRUTES dans le top {TOP_K_REPORT} : "
          f"{n_raw_in_top}/{TOP_K_REPORT}")

    # ------------------------------------------- poids de propension (exp014)
    # p(test) / (1 - p(test)) sur les lignes train : plus une ligne train
    # ressemble au test, plus son poids est élevé.
    p_train_rows = np.clip(oof_all[:n_train_rows], 1e-6, 1 - 1e-6)
    propensity = p_train_rows / (1 - p_train_rows)
    propensity = propensity / propensity.mean()   # normalisé, moyenne = 1

    # Écrêtage indispensable : quand l'AUC adversariale est élevée, quelques
    # lignes train reçoivent des poids extrêmes (x1000+) qui feraient
    # effectivement porter toute la CV d'exp014 sur une poignée d'échantillons.
    raw_max = propensity.max()
    propensity = np.clip(propensity, 0.0, WEIGHT_CLIP)
    propensity = propensity / propensity.mean()   # renormalisé après écrêtage
    n_clipped = int((propensity >= WEIGHT_CLIP - 1e-9).sum())

    pd.DataFrame({"row_index": np.arange(n_train_rows), "weight": propensity}) \
        .to_csv(MODELS_DIR / "exp011_propensity_weights.csv", index=False)
    print(f"\nPoids de propension écrits (moyenne={propensity.mean():.3f}, "
          f"max={propensity.max():.2f}, max avant écrêtage={raw_max:.1f}, "
          f"{n_clipped} lignes écrêtées à {WEIGHT_CLIP}) -- réutilisables à exp014.")
    if raw_max > 100:
        print(">> Poids bruts très étalés : signe d'un décalage fort. L'écrêtage évite "
              "qu'une poignée de lignes domine la CV repondérée d'exp014.")

    # ------------------------------------------------------------ sorties
    pd.DataFrame([
        {"run": "all_features", "n_features": X.shape[1], "adv_auc": round(auc_all, 4)},
        {"run": "normalized_only", "n_features": len(norm_cols), "adv_auc": round(auc_norm, 4)},
    ]).to_csv(MODELS_DIR / "exp011_adversarial_report.csv", index=False)

    imp_all.to_csv(MODELS_DIR / "exp011_adversarial_importance.csv", index=False)
    print(f"Rapport écrit : {MODELS_DIR / 'exp011_adversarial_report.csv'}")
    print(f"Importances écrites : {MODELS_DIR / 'exp011_adversarial_importance.csv'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))
        top_plot = imp_all.head(TOP_K_REPORT).iloc[::-1]
        ax.barh(top_plot["feature"], top_plot["adv_importance"])
        ax.set_title(f"Features séparant train/test (AUC adv = {auc_all:.3f})")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "exp011_adversarial_top_features.png", dpi=120)
        plt.close(fig)
        print(f"Figure écrite : {FIGURES_DIR / 'exp011_adversarial_top_features.png'}")
    except ImportError:
        print("matplotlib indisponible -- figure non générée.")


if __name__ == "__main__":
    main()

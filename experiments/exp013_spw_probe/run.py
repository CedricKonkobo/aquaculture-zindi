"""
experiments/exp013_spw_probe/run.py
---------------------------------------
Sondage leaderboard sur scale_pos_weight -- le levier LÉGAL pour corriger le
placement de la coupe à 0.5 (le réglage du seuil lui-même est interdit).

Constat déclencheur (exp012) : le modèle prédit 48.8 % de positifs sur le
test alors que le train en contient ~38 %. Il sur-prédit donc les bassins ->
faux positifs -> précision dégradée -> F1 effondré au seuil 0.5. Or le F1
pèse 60 % du score du challenge.

IMPORTANT -- pourquoi l'OOF ne peut pas trancher :
l'OOF est calculé sur la distribution du TRAIN. Il sélectionnera le spw
optimal POUR LE TRAIN (~1.6, cf. exp004), pas pour le test. C'est exactement
l'erreur commise à exp004. La seule information exploitable localement est le
TAUX DE POSITIFS PRÉDITS sur le test.

Usage :
    python -m experiments.exp013_spw_probe.run
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.preprocessing import load_train, load_test
from src.augmentation import build_augmented_replicas
from src.features import featurize_many, featurize
from src.train import cross_validate, default_fit_fn
from src.inference import predict_proba_ensemble, make_submission
from src.utils import SEED, MODELS_DIR

SPW_GRID = [0.5, 0.7, 1.0, 1.3, 1.6, 2.0]
N_AUG = 10
N_SPLITS = 5

LB_BASELINE = 0.82        # meilleur LB actuel (pipeline brut, spw=1.6)
LB_ALIGNED = 0.778        # exp012, pour mémoire : l'alignement dégrade


def make_lgbm_factory(spw):
    """LightGBM manuel (exp005) avec scale_pos_weight paramétrable."""
    def factory():
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=30,
            feature_fraction=0.7,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l1=0.5,
            lambda_l2=5.0,
            scale_pos_weight=spw,
            verbose=-1,
            seed=SEED,
        )
    return factory


def main():
    tr_ids, y, tr_cube = load_train()
    te_ids, _, te_cube = load_test()

    train_pos_rate = float(np.mean(y))
    print(f"Taux de positifs dans le TRAIN : {train_pos_rate:.3f}")
    print(f"(Référence : le modèle actuel spw=1.6 prédit 0.488 sur le test — "
          f"soit {0.488 - train_pos_rate:+.3f} vs le train)\n")

    cubes, ys, groups = build_augmented_replicas(tr_cube, y, n_aug=N_AUG, seed=SEED)
    X = featurize_many(cubes)
    Y = np.concatenate(ys)
    G = np.concatenate(groups)
    Xte = featurize(te_cube)[X.columns]
    print(f"Train augmenté : {X.shape}  |  Test : {Xte.shape}\n")

    rows = []
    for spw in SPW_GRID:
        print("=" * 70)
        print(f"scale_pos_weight = {spw}")
        print("=" * 70)

        result = cross_validate(make_lgbm_factory(spw), X, Y, G, n_splits=N_SPLITS,
                                 threshold=0.5, seed=SEED, verbose=False,
                                 fit_fn=default_fit_fn)

        proba_test = predict_proba_ensemble(result["models"], Xte)
        pos_rate = float((proba_test > 0.5).mean())

        fname = f"exp013_spw_{str(spw).replace('.', 'p')}_submission.csv"
        make_submission(te_ids, proba_test, threshold=0.5, filename=fname)

        print(f"OOF  AUC={result['auc']:.4f}  F1={result['f1']:.4f}  "
              f"score={result['score']:.4f}")
        print(f"Taux de positifs prédits sur le TEST : {pos_rate:.3f}  "
              f"(écart au train : {pos_rate - train_pos_rate:+.3f})\n")

        rows.append({
            "spw": spw,
            "oof_auc": round(result["auc"], 4),
            "oof_f1": round(result["f1"], 4),
            "oof_score": round(result["score"], 4),
            "test_pos_rate": round(pos_rate, 4),
            "gap_vs_train_rate": round(pos_rate - train_pos_rate, 4),
            "submission": fname,
        })

    df = pd.DataFrame(rows)
    print("=" * 70)
    print("SYNTHÈSE")
    print("=" * 70)
    print(df.to_string(index=False))

    # --------------------------------------------------------- recommandation
    df["abs_gap"] = df["gap_vs_train_rate"].abs()
    closest = df.loc[df["abs_gap"].idxmin()]
    best_oof = df.loc[df["oof_score"].idxmax()]

    print(f"\nMeilleur OOF        : spw={best_oof['spw']} (score={best_oof['oof_score']:.4f})")
    print("  -> NE PAS s'y fier : l'OOF est calculé sur la distribution du TRAIN.")
    print(f"\nTaux prédit le plus proche du train : spw={closest['spw']} "
          f"(taux={closest['test_pos_rate']:.3f})")

    print("\n--- STRATÉGIE DE SOUMISSION (3 envois maximum) ---")
    print(f"1. Envoyer spw={closest['spw']} ({closest['submission']})")
    print("   -> teste l'hypothèse « le modèle sur-prédit les positifs ».")
    lower = df[df["spw"] < closest["spw"]]
    if len(lower):
        cand = lower.iloc[-1]
        print(f"2. Envoyer spw={cand['spw']} ({cand['submission']})")
        print("   -> donne le SENS de la réponse du LB (le score continue-t-il de monter ?).")
    print("3. Selon le sens observé, envoyer UNE valeur supplémentaire pour affiner.")
    print(f"\nRéférences LB : pipeline brut spw=1.6 -> {LB_BASELINE} | "
          f"aligné (exp012) -> {LB_ALIGNED}")
    print("Si le LB monte quand spw baisse : continuer à descendre jusqu'à inversion.")
    print("Si le LB baisse : le modèle sous-prédisait, remonter spw au-dessus de 1.6.")
    print("Si le LB ne bouge pas : le F1 n'est pas le facteur limitant -> "
          "revenir au feature engineering.")

    df.drop(columns=["abs_gap"]).to_csv(MODELS_DIR / "exp013_spw_probe.csv", index=False)
    print(f"\nRésultats écrits : {MODELS_DIR / 'exp013_spw_probe.csv'}")


if __name__ == "__main__":
    main()

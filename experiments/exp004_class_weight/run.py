"""
experiments/exp004_class_weight/run.py
-----------------------------------------
Test de l'hypothèse H2 (EDA, étape 3) : le prior train (40.4% positifs) est
probablement différent du prior test réel (les organisateurs indiquent qu'il
est plus élevé, sans donner de chiffre).

IMPORTANT — contrairement à H3/H5, ce n'est PAS une hypothèse validable par
le seul score OOF : la CV compare toujours aux vrais labels train, au prior
40/60. Augmenter scale_pos_weight va donc mécaniquement FAIRE BAISSER le
score OOF (le modèle s'éloigne du prior sur lequel il est jugé) -- ça ne veut
pas dire que c'est une mauvaise idée pour le vrai test.

Ce script ne "décide" donc rien automatiquement. Il balaie une grille de
scale_pos_weight et reporte, pour chacun :
  - le score OOF (référence, pour information -- baisse attendue)
  - le taux de positifs prédit sur Test.csv (l'indicateur pertinent ici)

La décision finale (quel poids retenir, si on en retient un) est un jugement
à faire en lisant results.md, pas une règle automatique comme H3/H5.

Usage :
    python -m experiments.exp004_class_weight.run
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.preprocessing import load_train, load_test
from src.augmentation import build_augmented_replicas
from src.features import featurize_many, featurize   # référence exp001 (245 features)
from src.train import cross_validate
from src.inference import predict_proba_ensemble, make_submission, save_models
from src.utils import SEED, MODELS_DIR

N_AUG = 10
WEIGHT_GRID = [1.0, 1.3, 1.6, 2.0, 2.5]   # 1.0 = référence exp001 (pas de pondération)

BASE_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=40,
    feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
    lambda_l1=0.5, lambda_l2=5.0, n_estimators=1200, verbose=-1, seed=SEED,
)


def model_factory(weight):
    def factory():
        return lgb.LGBMClassifier(**BASE_PARAMS, scale_pos_weight=weight)
    return factory


def main():
    tr_ids, y, tr_cube = load_train()
    te_ids, _, te_cube = load_test()

    cubes, ys, groups = build_augmented_replicas(tr_cube, y, n_aug=N_AUG, seed=SEED)
    X = featurize_many(cubes)
    Y = np.concatenate(ys)
    G = np.concatenate(groups)
    Xte = featurize(te_cube)[X.columns]

    print(f"train augmenté : {X.shape} | test : {Xte.shape}")
    print(f"prior train (répliques) : {Y.mean():.3f} positifs\n")

    rows = []
    for w in WEIGHT_GRID:
        print(f"=== scale_pos_weight = {w} ===")
        result = cross_validate(model_factory(w), X, Y, G, n_splits=5, threshold=0.5,
                                 seed=SEED, verbose=False)
        proba_test = predict_proba_ensemble(result["models"], Xte)
        test_pos_rate = (proba_test > 0.5).mean()

        rows.append({
            "scale_pos_weight": w,
            "auc_oof": round(result["auc"], 4),
            "f1_oof": round(result["f1"], 4),
            "score_oof": round(result["score"], 4),
            "test_pos_rate": round(test_pos_rate, 4),
        })
        print(f"  AUC(OOF)={result['auc']:.4f}  F1(OOF)={result['f1']:.4f}  "
              f"score(OOF)={result['score']:.4f}  taux_positif(test)={test_pos_rate:.3f}")

        # on sauvegarde chaque submission pour pouvoir choisir après coup
        fname = f"exp004_class_weight_w{str(w).replace('.', '')}_submission.csv"
        make_submission(te_ids, proba_test, threshold=0.5, filename=fname)
        save_models(result["models"], name=f"exp004_class_weight_w{str(w).replace('.', '')}")

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    out = MODELS_DIR / "exp004_weight_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nRésultats écrits : {out}")
    print("\nRappel : le score OOF baisse mécaniquement avec le poids -- ce n'est pas")
    print("un signe d'échec. Juge sur l'évolution du taux de positifs test et sur")
    print("un compromis raisonnable, pas sur le score OOF maximal (voir results.md).")


if __name__ == "__main__":
    main()

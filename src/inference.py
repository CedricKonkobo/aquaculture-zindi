"""
inference.py
------------
Prédiction sur le jeu de test et génération du fichier de soumission Zindi.

Sépare volontairement la logique d'inférence de celle d'entraînement (train.py) : un
notebook ou un script d'expérience peut charger des modèles déjà entraînés (depuis
outputs/models/) et regénérer une soumission sans tout ré-entraîner.
"""
import joblib
import pandas as pd

from src.utils import SUBMISSIONS_DIR, MODELS_DIR


def predict_proba_ensemble(models, X):
    """Moyenne les probabilités prédites par une liste de modèles (ex: les 5 folds de la CV)."""
    proba = sum(m.predict_proba(X)[:, 1] for m in models) / len(models)
    return proba


def make_submission(ids, proba, threshold=0.5, filename="submission.csv"):
    """
    Construit le DataFrame de soumission au format attendu par Zindi
    (ID, TargetF1, TargetRAUC) et l'écrit dans outputs/submissions/.
    """
    sub = pd.DataFrame({
        "ID": ids,
        "TargetF1": (proba > threshold).astype(int),
        "TargetRAUC": proba,
    })
    out_path = SUBMISSIONS_DIR / filename
    sub.to_csv(out_path, index=False)
    print(f"Soumission écrite : {out_path}  (taux de positifs = {sub.TargetF1.mean():.3f})")
    return sub


def save_models(models, name="exp001_baseline"):
    """Sauvegarde une liste de modèles (un par fold) dans outputs/models/."""
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(models, path)
    print(f"Modèles sauvegardés : {path}")
    return path


def load_models(name="exp001_baseline"):
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)

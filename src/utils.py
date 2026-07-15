"""
utils.py
--------
Constantes globales et petites fonctions utilitaires partagées par tout le projet.
Aucune logique métier lourde ici : seulement ce qui est réutilisé partout
(bandes, mois, seed, chemins, helpers numériques génériques).
"""
from pathlib import Path
import numpy as np

# --------------------------------------------------------------------------- reproductibilité
SEED = 42

# --------------------------------------------------------------------------- chemins projet
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"

TRAIN_PATH = DATA_DIR / "Train.csv"
TEST_PATH = DATA_DIR / "Test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "SampleSubmission.csv"

for d in (FIGURES_DIR, MODELS_DIR, SUBMISSIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- structure des données
MONTHS = [f"{i:02d}" for i in range(1, 13)]
S1_BANDS = ["VH", "VV"]
S2_BANDS = ["blue", "green", "red", "re1", "re2", "re3", "nir", "nira", "swir1", "swir2"]
BANDS = S1_BANDS + S2_BANDS
NA_VALUE = -9999

# Distribution empirique du nombre de mois S2 manquants (nuages) observée dans le test.
# Utilisée par le module d'augmentation pour recréer les mêmes conditions dans le train.
S2_GAP_P = np.array([0.735, 0.224, 0.036, 0.005])


def safe_ratio(a, b):
    """Ratio normalisé (a-b)/(a+b), protégé contre la division par un dénominateur ~0."""
    d = a + b
    return np.where(np.abs(d) < 1e-6, np.nan, (a - b) / np.where(np.abs(d) < 1e-6, 1, d))


def get_rng(seed=SEED):
    """Générateur numpy reproductible (à préférer à np.random global)."""
    return np.random.default_rng(seed)

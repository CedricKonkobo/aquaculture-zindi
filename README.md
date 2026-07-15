# Aquaculture Pond Identification — Zindi GeoAI

Détection de bassins d'aquaculture à partir de séries temporelles Sentinel-1/2,
avec un défi central : **le train couvre 12 mois complets, le test seulement une
fenêtre consécutive de 4 à 6 mois** (+ trous nuageux sur l'optique S2).

## Statut actuel
- `exp001_baseline` : LightGBM + augmentation par masquage temporel → ~100e place Zindi.
- Prochaine étape : EDA complète (`notebooks/01_EDA.ipynb`) pour guider les
  améliorations à partir d'observations, pas d'essais aléatoires.

## Architecture

```text
aquaculture-zindi/
│
├── data/                    # Train.csv, Test.csv, SampleSubmission.csv (non versionnés)
│
├── notebooks/                # Analyse uniquement — aucune logique métier ici
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Baseline_Analysis.ipynb
│
├── src/                      # Tout le code métier, réutilisable et testable
│   ├── utils.py               # constantes (bandes, mois, seed, chemins) + helpers génériques
│   ├── preprocessing.py        # chargement CSV -> cube numpy (n, 12 mois, 12 bandes)
│   ├── augmentation.py         # masquage temporel (simule les conditions du test)
│   ├── features.py             # indices spectraux + agrégations invariantes à la fenêtre
│   ├── train.py                # CV (StratifiedGroupKFold), seuil optimal, importance
│   └── inference.py             # prédiction + génération de la soumission Zindi
│
├── experiments/               # Une expérience = un dossier, reproductible
│   ├── exp001_baseline/
│   │   ├── run.py               # script d'entraînement de l'expérience
│   │   ├── config.yaml           # hyperparamètres utilisés
│   │   └── results.md            # score obtenu, observations, conclusion
│   ├── exp002_catboost/        # (à créer à l'Étape 5)
│   └── exp003_optuna/          # (à créer à l'Étape 6)
│
├── outputs/
│   ├── figures/               # graphiques exportés par les notebooks
│   ├── models/                # modèles entraînés (.joblib)
│   └── submissions/           # fichiers CSV soumis à Zindi
│
├── requirements.txt
└── README.md
```

## Principe directeur
- **`src/`** contient toute la logique réutilisable. Un notebook ou un script
  d'expérience ne fait qu'appeler ces fonctions — jamais de copier-coller de logique.
- **`notebooks/`** sert uniquement à l'exploration et à la visualisation.
- **`experiments/`** capture un état reproductible : config + code d'appel + résultats.
  On ne modifie jamais un `run.py` d'une expérience passée après coup ; on en crée une nouvelle.

## Comment lancer la baseline actuelle
```bash
# depuis la racine du projet, avec data/Train.csv et data/Test.csv présents
pip install -r requirements.txt
python -m experiments.exp001_baseline.run
```
Le script écrit la soumission dans `outputs/submissions/` et le modèle dans `outputs/models/`.

## Prochaines étapes (voir aussi results.md de chaque expérience)
1. EDA complète → `notebooks/01_EDA.ipynb`
2. Analyse détaillée de la baseline → `notebooks/03_Baseline_Analysis.ipynb`
3. Feature engineering justifié par l'EDA
4. Comparaison de familles de modèles (LightGBM, CatBoost, XGBoost, RF, ExtraTrees, MLP)
5. Optimisation Optuna + recherche du seuil optimal
6. Documentation complète du workflow quotidien

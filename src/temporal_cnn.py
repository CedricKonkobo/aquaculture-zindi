"""
temporal_cnn.py
----------------
Réseau convolutif 1D sur les séries mensuelles brutes (12 mois x 12 bandes).

Motivation (exp017) : le projet a atteint un plateau à ~0.836. Trois
configurations différentes (pseudo-étiquetage pw=0.5, pw=2.0, blend de trois
GBDT) donnent le même score. Le blend LightGBM + XGBoost + CatBoost est
neutre (+0.0011 sur le holdout) : trois algorithmes à base d'arbres, sur les
mêmes features, produisent des prédictions trop corrélées pour se compléter.

Le motif qui a fonctionné deux fois sur ce projet est l'apport d'information
RÉELLEMENT NOUVELLE :
  - features invariantes intra-échantillon (exp015) : +0.0073
  - accès au domaine cible par pseudo-étiquetage (exp016) : +0.0041

Or il reste une information largement inexploitée : l'ORDRE TEMPOREL. Toutes
les features de src/features.py et src/features_invariant.py sont des
statistiques AGRÉGÉES sur les mois (moyenne, min, max, percentiles, écart
type). La séquence elle-même est perdue. Un bassin aquacole a pourtant
vraisemblablement une dynamique caractéristique (remplissage, exploitation,
vidange, récolte) qu'une moyenne annuelle efface.

Ce module consomme donc directement la séquence (12, 12) au lieu de
statistiques résumées. Ses erreurs devraient être décorrélées de celles des
GBDT -- ce qui donne au blend GBDT + réseau un potentiel réel, contrairement
au blend de trois arbres.

Contraintes de conception :
  - Peu de données (1821 échantillons d'origine ; l'augmentation x10 aide
    mais les répliques sont corrélées) -> architecture PETITE et fortement
    régularisée.
  - Le décalage de domaine frappe aussi ce modèle -> pas de miracle attendu
    sur le score individuel ; l'intérêt est la DÉCORRÉLATION.
  - Les mois manquants (NaN) sont fréquents et porteurs de sens (nuages) ->
    ils sont remplacés par 0 APRÈS normalisation, et un canal de MASQUE
    explicite indique au réseau quels mois sont observés.
"""
import numpy as np
import torch
import torch.nn as nn


class TemporalCNN(nn.Module):
    """
    Convolutions 1D le long de l'axe temporel.

    Entrée : (batch, n_channels, 12) où n_channels = n_bands + n_bands (masque)
    Sortie : logit unique.

    Les convolutions à noyau 3 capturent des motifs locaux (montée, descente,
    pic) ; le pooling global rend le modèle robuste à la POSITION du motif
    dans l'année -- ce qui est essentiel puisque le test provient d'une autre
    période et que la fenêtre observée n'y est pas la même.
    """

    def __init__(self, n_channels, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # moyenne + max global : invariance à la position temporelle du motif
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        h = self.net(x)
        pooled = torch.cat([h.mean(dim=2), h.max(dim=2).values], dim=1)
        return self.head(pooled).squeeze(1)


def cube_to_tensor(cube, mean=None, std=None):
    """
    Transforme un cube (n, 12 mois, n_bands) en tenseur (n, 2*n_bands, 12).

    - normalisation par bande (statistiques fournies ou calculées) ;
    - NaN -> 0 APRÈS normalisation ;
    - canal de masque explicite (1 = mois observé, 0 = manquant), pour que le
      réseau distingue "valeur nulle" de "donnée absente".

    Retourne (tensor, mean, std) : les statistiques du train doivent être
    réutilisées pour le test.
    """
    arr = np.asarray(cube, dtype=np.float32)          # (n, 12, n_bands)
    mask = (~np.isnan(arr)).astype(np.float32)

    if mean is None or std is None:
        mean = np.nanmean(arr, axis=(0, 1))
        std = np.nanstd(arr, axis=(0, 1))
        std = np.where(std < 1e-6, 1.0, std)

    normed = (arr - mean) / std
    normed = np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)

    x = np.concatenate([normed, mask], axis=2)        # (n, 12, 2*n_bands)
    x = np.transpose(x, (0, 2, 1))                    # (n, 2*n_bands, 12)
    return torch.from_numpy(np.ascontiguousarray(x)), mean, std


def train_cnn(X_tr, y_tr, w_tr=None, n_channels=None, epochs=30, batch_size=256,
               lr=1e-3, weight_decay=1e-4, pos_weight=1.6, dropout=0.3,
               hidden=64, seed=42, device=None, verbose=False):
    """
    Entraîne un TemporalCNN. Retourne le modèle en mode eval.

    w_tr : poids par échantillon (pour les pseudo-étiquettes).
    pos_weight : équivalent de scale_pos_weight (figé à 1.6, cf. exp013).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_channels = n_channels or X_tr.shape[1]
    model = TemporalCNN(n_channels, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    y_t = torch.tensor(np.asarray(y_tr, dtype=np.float32))
    w_t = torch.tensor(np.asarray(w_tr if w_tr is not None else np.ones(len(y_tr)),
                                   dtype=np.float32))
    pw = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    n = len(y_t)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_tr[idx].to(device)
            yb = y_t[idx].to(device)
            wb = w_t[idx].to(device)

            logits = model(xb)
            # pondération positive + poids d'échantillon, sans réduction
            loss_el = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, pos_weight=pw, reduction="none")
            loss = (loss_el * wb).sum() / wb.sum().clamp(min=1e-8)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}/{epochs}  loss={total/n:.4f}")

    model.eval()
    return model


@torch.no_grad()
def predict_cnn(model, X, batch_size=512, device=None):
    """Probabilités prédites (sigmoïde des logits)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)

"""
backtester/synthetic.py — Données simulées (tests, démos hors-ligne)

⚠ Ces données sont FABRIQUÉES : tout résultat obtenu dessus ne dit rien sur
les marchés réels. Elles servent à vérifier que le code tourne et que les
mécanismes (neutralité, coûts, cointégration) se comportent comme prévu.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def make_universe(
    n_assets: int = 20,
    n_days: int = 1500,
    seed: int = 42,
    drift_persistence: float = 0.995,
    drift_vol: float = 0.0001,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Univers de n_assets actions + un "marché" (SPY simulé).

        r_i,t = beta_i × r_m,t + mu_i,t + eps_i,t
        mu_i,t = persistence × mu_i,t-1 + choc      (dérive lente)

    La dérive lente mu_i,t crée un VRAI momentum (les gagnants récents
    continuent un peu), sans quoi un facteur momentum n'aurait aucun IC.
    Réglage par défaut : écart-type stationnaire de mu ≈ 0.1 %/jour (dérive lente, de signe variable),
    ce qui donne un IC cross-sectionnel de l'ordre de quelques centièmes
    (réaliste) et non un Sharpe irréaliste.

    Retourne (close DataFrame, close du marché Series).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    mkt = rng.normal(0.0003, 0.01, n_days)
    betas = np.linspace(0.5, 1.5, n_assets)

    mu = np.zeros((n_days, n_assets))
    for t in range(1, n_days):
        mu[t] = drift_persistence * mu[t - 1] + rng.normal(0, drift_vol, n_assets)

    eps = rng.normal(0, 0.012, (n_days, n_assets))
    rets = mkt[:, None] * betas[None, :] + mu + eps
    cols = [f"S{i:02d}" for i in range(n_assets)]
    close = 100 * np.cumprod(1 + rets, axis=0)
    return (pd.DataFrame(close, index=idx, columns=cols),
            pd.Series(100 * np.cumprod(1 + mkt), index=idx, name="SPY"))


def make_pair(n_days: int = 1500, beta: float = 1.2, phi: float = 0.93, seed: int = 7):
    """Deux prix cointégrés : log(Y) = beta × log(X) + c + AR(1)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    lx = 4.0 + np.cumsum(rng.normal(0.0002, 0.01, n_days))
    e = np.zeros(n_days)
    for t in range(1, n_days):
        e[t] = phi * e[t - 1] + rng.normal(0, 0.012)
    ly = beta * lx + 0.2 + e
    return pd.Series(np.exp(ly), index=idx, name="Y"), pd.Series(np.exp(lx), index=idx, name="X")

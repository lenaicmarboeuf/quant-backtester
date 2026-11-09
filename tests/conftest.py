"""
conftest.py — Fixtures partagées pour tous les tests.
Les fixtures pytest sont des fonctions réutilisables qui préparent
des données de test. Le décorateur @pytest.fixture les enregistre
auprès de pytest, qui les injecte automatiquement dans les tests
qui les demandent (par leur nom en argument).
"""
import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Séance 9 : permet d'importer `backtester` quel que soit le dossier d'où
# on lance pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Données synthétiques déterministes ──────────────────────────────────────
# np.random.seed() : fixe la graine du générateur aléatoire.
# Même graine → mêmes nombres → tests reproductibles à l'identique.

N = 500  # 500 jours de trading synthétiques (~2 ans)

@pytest.fixture
def price_series() -> pd.Series:
    """Série de prix réaliste (marche aléatoire log-normale)."""
    np.random.seed(42)
    # Simulation d'un actif avec drift +10%/an et vol 20%/an
    daily_returns = np.random.normal(loc=0.0004, scale=0.012, size=N)
    prices = 100 * np.exp(np.cumsum(daily_returns))
    idx = pd.date_range("2022-01-01", periods=N, freq="B")
    return pd.Series(prices, index=idx, name="Close")


@pytest.fixture
def returns_series(price_series) -> pd.Series:
    """Rendements journaliers dérivés de price_series."""
    return price_series.pct_change().dropna()


@pytest.fixture
def equity_series() -> pd.Series:
    """Courbe de capital croissante simple (10 000 → ~18 000 en 500j)."""
    np.random.seed(0)
    r = pd.Series(np.random.normal(0.0006, 0.010, N))
    equity = 10_000 * (1 + r).cumprod()
    return equity


@pytest.fixture
def ohlcv_df(price_series) -> pd.DataFrame:
    """DataFrame OHLCV synthétique cohérent."""
    np.random.seed(7)
    close = price_series
    noise = np.abs(np.random.normal(0, 0.005, N))
    high  = close * (1 + noise)
    low   = close * (1 - noise)
    open_ = close.shift(1).fillna(close.iloc[0])
    vol   = np.random.randint(1_000_000, 5_000_000, N)
    df = pd.DataFrame({
        "Open": open_.values, "High": high.values,
        "Low": low.values,   "Close": close.values,
        "Volume": vol,
    }, index=close.index)
    df["returns"] = df["Close"].pct_change()
    return df


# ═════════════════════════════════════════════════════════════════════════════
# SÉANCE 9 — fixtures multi-actifs (long/short, pairs, PortfolioBacktester)
# Utilisent un générateur local (default_rng) : n'interfèrent pas avec
# np.random.seed() des fixtures ci-dessus.
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def dates() -> pd.DatetimeIndex:
    """600 jours ouvrés."""
    return pd.bdate_range("2020-01-01", periods=600)


@pytest.fixture
def market(dates) -> pd.Series:
    """Rendements d'un « marché » simulé (équivalent SPY)."""
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0004, 0.01, len(dates)), index=dates, name="SPY")


@pytest.fixture
def universe(dates, market):
    """
    20 actifs : r_i = beta_i × marché + bruit idiosyncratique.
    Les betas sont étalés de 0.4 à 1.6 pour tester la neutralité beta.
    Retourne (rendements DataFrame, betas vrais Series).
    """
    rng = np.random.default_rng(1)
    betas = np.linspace(0.4, 1.6, 20)
    cols = [f"A{i:02d}" for i in range(20)]
    rets = pd.DataFrame(
        {c: b * market.values + rng.normal(0, 0.01, len(dates)) for c, b in zip(cols, betas)},
        index=dates,
    )
    return rets, pd.Series(betas, index=cols)


@pytest.fixture
def close_df(universe) -> pd.DataFrame:
    """Prix de clôture des 20 actifs de `universe`."""
    rets, _ = universe
    return 100 * (1 + rets).cumprod()


@pytest.fixture
def single_asset(dates, market) -> pd.DataFrame:
    """Un seul actif (Close + returns) pour tester le Backtester long/short."""
    close = 100 * (1 + market).cumprod()
    return pd.DataFrame({"Close": close, "returns": close.pct_change()}).dropna()
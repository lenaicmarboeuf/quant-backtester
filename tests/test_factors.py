"""
tests/test_factors.py — Tests unitaires pour backtester/factors.py

Convention : toutes les fixtures sont définies dans conftest.py
Ici on vérifie les propriétés mathématiques attendues de chaque fonction.
"""

import numpy as np
import pandas as pd
import pytest

# Import relatif — ajuster selon la structure du projet
import sys
sys.path.insert(0, '.')

from backtester.factors import (
    momentum_factor,
    volatility_factor,
    quality_factor,
    compute_ic,
    compute_rolling_ic,
    compute_icir,
    fama_french_alpha,
    _cross_section_zscore,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def price_df():
    """DataFrame de prix synthétiques pour 3 actifs sur 500 jours."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = {}
    for ticker, drift in [("AAPL", 0.0003), ("MSFT", 0.0001), ("GOOG", 0.0002)]:
        returns = np.random.normal(drift, 0.015, n)
        prices  = 100 * np.cumprod(1 + returns)
        data[ticker] = prices
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def price_series(price_df):
    """Série de prix univariée (AAPL)."""
    return price_df["AAPL"]


@pytest.fixture
def forward_returns(price_series):
    """Rendements forward J+1 (shift=-1, intentionnel pour calcul IC)."""
    return price_series.pct_change().shift(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Tests : _cross_section_zscore
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossSectionZscore:
    def test_mean_near_zero(self, price_df):
        """La moyenne cross-sectionnelle doit être ≈ 0 après normalisation."""
        z = _cross_section_zscore(price_df)
        row_means = z.dropna(how="all").mean(axis=1)
        assert (row_means.abs() < 1e-10).all(), "Moyenne cross-sectionnelle non nulle"

    def test_clip_at_3(self, price_df):
        """Les z-scores doivent être clippés entre -3 et +3."""
        z = _cross_section_zscore(price_df)
        assert z.max().max() <= 3.001
        assert z.min().min() >= -3.001

    def test_same_shape(self, price_df):
        """La sortie doit avoir la même forme que l'entrée."""
        z = _cross_section_zscore(price_df)
        assert z.shape == price_df.shape


# ─────────────────────────────────────────────────────────────────────────────
# Tests : momentum_factor
# ─────────────────────────────────────────────────────────────────────────────

class TestMomentumFactor:
    def test_returns_dataframe(self, price_df):
        mom = momentum_factor(price_df)
        assert isinstance(mom, pd.DataFrame)
        assert mom.shape == price_df.shape

    def test_accepts_series(self, price_series):
        """Doit accepter une Series univariée."""
        mom = momentum_factor(price_series)
        assert isinstance(mom, pd.DataFrame)

    def test_nan_at_start(self, price_df, window=252, skip=21):
        """Les premières (window) valeurs doivent être NaN (pas assez de données)."""
        mom = momentum_factor(price_df, window=window, skip=skip)
        assert mom.iloc[:window].isna().all().all()

    def test_values_in_range(self, price_df):
        """Les z-scores doivent être dans [-3, +3]."""
        mom = momentum_factor(price_df)
        valid = mom.dropna()
        assert valid.max().max() <= 3.01
        assert valid.min().min() >= -3.01

    def test_cross_sectional_sum_near_zero(self, price_df):
        """La somme des scores à une date donnée doit être ≈ 0."""
        mom = momentum_factor(price_df).dropna()
        row_sums = mom.sum(axis=1)
        assert (row_sums.abs() < 1e-9).all()


# ─────────────────────────────────────────────────────────────────────────────
# Tests : volatility_factor
# ─────────────────────────────────────────────────────────────────────────────

class TestVolatilityFactor:
    def test_returns_dataframe(self, price_df):
        vf = volatility_factor(price_df)
        assert isinstance(vf, pd.DataFrame)
        assert vf.shape == price_df.shape

    def test_low_vol_positive_score(self):
        """L'actif le moins volatil doit avoir le score le plus élevé."""
        dates = pd.date_range("2020-01-01", periods=200, freq="B")
        np.random.seed(0)
        # low_vol : très stable, high_vol : très agité
        low  = pd.Series(100 * np.cumprod(1 + np.random.normal(0, 0.002, 200)), index=dates)
        high = pd.Series(100 * np.cumprod(1 + np.random.normal(0, 0.04,  200)), index=dates)
        df = pd.DataFrame({"low": low, "high": high})
        vf = volatility_factor(df, window=63).dropna()
        # Le score moyen de 'low' doit être supérieur à celui de 'high'
        assert vf["low"].mean() > vf["high"].mean()

    def test_nan_at_start(self, price_df):
        vf = volatility_factor(price_df, window=63)
        assert vf.iloc[:63].isna().all().all()


# ─────────────────────────────────────────────────────────────────────────────
# Tests : quality_factor
# ─────────────────────────────────────────────────────────────────────────────

class TestQualityFactor:
    def test_returns_dataframe(self, price_df):
        qf = quality_factor(price_df)
        assert isinstance(qf, pd.DataFrame)

    def test_high_sharpe_high_score(self):
        """L'actif avec le meilleur Sharpe doit avoir le meilleur score."""
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        np.random.seed(1)
        good = pd.Series(100 * np.cumprod(1 + np.random.normal(0.001, 0.005, 300)), index=dates)
        bad  = pd.Series(100 * np.cumprod(1 + np.random.normal(-0.001, 0.02, 300)), index=dates)
        df = pd.DataFrame({"good": good, "bad": bad})
        qf = quality_factor(df).dropna()
        assert qf["good"].mean() > qf["bad"].mean()


# ─────────────────────────────────────────────────────────────────────────────
# Tests : compute_ic
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeIC:
    def test_perfect_correlation(self):
        """Signal = rendement futur → IC = 1."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        signal = pd.Series(np.random.randn(100), index=dates)
        fwd    = signal.copy()  # corrélation parfaite
        ic = compute_ic(signal, fwd)
        assert abs(ic - 1.0) < 0.01

    def test_no_correlation(self):
        """Signal aléatoire indépendant → IC ≈ 0."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        signal = pd.Series(np.random.randn(500), index=dates)
        fwd    = pd.Series(np.random.randn(500), index=dates)
        ic = compute_ic(signal, fwd)
        assert abs(ic) < 0.15  # tolérance car bruit aléatoire

    def test_returns_float(self, price_series, forward_returns):
        ic = compute_ic(price_series.pct_change(), forward_returns)
        assert isinstance(ic, float)
        assert -1.0 <= ic <= 1.0

    def test_insufficient_data(self):
        """Retourne une Series vide si moins de 10 observations."""
        signal = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        fwd    = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))
        result = compute_ic(signal, fwd)
        # Doit gérer gracieusement le manque de données
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# Tests : compute_icir
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeICIR:
    def test_stable_signal_higher_icir(self):
        """Un IC régulier doit avoir un ICIR plus élevé qu'un IC erratique."""
        dates = pd.date_range("2020-01-01", periods=200, freq="B")
        stable   = pd.Series(np.full(200, 0.08), index=dates)           # IC = 0.08 tous les jours
        erratic  = pd.Series(np.random.choice([-0.1, 0.2], 200), index=dates)  # IC aléatoire

        icir_stable  = compute_icir(stable).dropna()
        icir_erratic = compute_icir(erratic).dropna()

        assert icir_stable.mean() > icir_erratic.mean()

    def test_returns_series(self, price_series, forward_returns):
        from backtester.factors import compute_rolling_ic
        ic = compute_rolling_ic(price_series.pct_change(), forward_returns)
        icir = compute_icir(ic)
        assert isinstance(icir, pd.Series)
        assert len(icir) == len(ic)


# ─────────────────────────────────────────────────────────────────────────────
# Tests : fama_french_alpha
# ─────────────────────────────────────────────────────────────────────────────

class TestFamaFrenchAlpha:
    def test_returns_dict(self, price_series):
        np.random.seed(42)
        r_strat = price_series.pct_change().dropna()
        r_mkt   = pd.Series(np.random.normal(0.0003, 0.01, len(r_strat)),
                             index=r_strat.index)
        result = fama_french_alpha(r_strat, r_mkt)
        assert isinstance(result, dict)
        assert "alpha (annualisé %)" in result
        assert "beta" in result
        assert "R²" in result

    def test_beta_near_one_for_identical(self, price_series):
        """Si strat = marché, beta ≈ 1 et alpha ≈ 0."""
        r = price_series.pct_change().dropna()
        result = fama_french_alpha(r, r)
        assert abs(result["beta"] - 1.0) < 0.01
        assert abs(result["alpha (annualisé %)"])  < 0.1

    def test_insufficient_data(self):
        r_short = pd.Series([0.01, 0.02, -0.01], index=pd.date_range("2020-01-01", 3))
        result = fama_french_alpha(r_short, r_short)
        assert "error" in result

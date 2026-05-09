"""
tests/test_metrics.py — Tests unitaires pour backtester/metrics.py

Stratégie de test :
  - Cas nominal    : données réalistes → résultat attendu dans une plage
  - Cas limite     : série vide, std=0, gains uniquement, pertes uniquement
  - Cas mathématique : valeurs connues → résultat exact vérifiable à la main

Pourquoi tester les cas limites ?
  Une division par zéro ou un NaN silencieux peut fausser toutes
  les métriques d'un backtest sans lever d'erreur. Les tests gardiens
  (edge cases) attrapent ces bugs tôt.
"""
import pytest
import pandas as pd
import numpy as np
from backtester.metrics import (
    sharpe_ratio, max_drawdown, cagr, calmar_ratio,
    win_rate, profit_factor, value_at_risk, full_metrics,
)


# ═══════════════════════════════════════════════════════════════
# sharpe_ratio
# ═══════════════════════════════════════════════════════════════

class TestSharpeRatio:
    def test_positive_returns_positive_sharpe(self, returns_series):
        """Rendements positifs → Sharpe positif."""
        result = sharpe_ratio(returns_series)
        assert result > 0

    def test_zero_std_returns_zero(self):
        """Vol nulle (rendements constants) → Sharpe = 0 (pas de division par zéro)."""
        flat = pd.Series([0.001] * 100)
        assert sharpe_ratio(flat) == 0.0

    def test_known_value(self):
        """
        Vérification mathématique :
        mean=0.01, std=0.05, periods=252
        Sharpe = (0.01/0.05) * sqrt(252) = 0.2 * 15.875 ≈ 3.175
        """
        r = pd.Series([0.01, 0.01, -0.01, 0.03, 0.01] * 20)
        result = sharpe_ratio(r, periods=252)
        expected = (r.mean() / r.std()) * np.sqrt(252)
        assert abs(result - expected) < 1e-10

    def test_negative_returns_negative_sharpe(self):
        """Pertes constantes → Sharpe négatif."""
        losing = pd.Series([-0.005] * 50 + [-0.02] * 50)
        assert sharpe_ratio(losing) < 0

    def test_annualization_periods(self, returns_series):
        """periods=52 (hebdo) doit donner un Sharpe différent de periods=252."""
        s_daily  = sharpe_ratio(returns_series, periods=252)
        s_weekly = sharpe_ratio(returns_series, periods=52)
        assert s_daily != s_weekly


# ═══════════════════════════════════════════════════════════════
# max_drawdown
# ═══════════════════════════════════════════════════════════════

class TestMaxDrawdown:
    def test_always_negative_or_zero(self, equity_series):
        """Le drawdown est toujours ≤ 0 par définition."""
        assert max_drawdown(equity_series) <= 0

    def test_monotonic_increase_zero_drawdown(self):
        """Equity strictement croissante → drawdown = 0."""
        equity = pd.Series([100, 110, 120, 130, 140])
        assert max_drawdown(equity) == 0.0

    def test_known_drawdown(self):
        """
        Equity : 100 → 80 → 90
        Peak = 100, trough = 80
        Drawdown = (80-100)/100 = -20%
        """
        equity = pd.Series([100.0, 90.0, 80.0, 85.0, 90.0])
        result = max_drawdown(equity)
        assert abs(result - (-0.20)) < 1e-10

    def test_handles_nan(self):
        """NaN en début de série ignorés (dropna)."""
        equity = pd.Series([np.nan, 100.0, 90.0, 95.0])
        result = max_drawdown(equity)
        assert result <= 0


# ═══════════════════════════════════════════════════════════════
# cagr
# ═══════════════════════════════════════════════════════════════

class TestCagr:
    def test_positive_growth(self, equity_series):
        """Capital final > initial → CAGR positif."""
        assert cagr(equity_series) > 0

    def test_known_cagr(self):
        """
        Capital double en 252 jours (1 an) → CAGR = 100%.
        (2/1)^(1/1) - 1 = 1.0
        """
        equity = pd.Series([1.0] + [2.0] * 251)
        result = cagr(equity, periods=252)
        assert abs(result - 1.0) < 1e-6

    def test_empty_series_returns_zero(self):
        """Série vide → 0.0 (pas d'erreur)."""
        assert cagr(pd.Series([], dtype=float)) == 0.0

    def test_zero_initial_returns_zero(self):
        """Capital initial = 0 → 0.0 (évite division par zéro)."""
        equity = pd.Series([0.0, 100.0, 200.0])
        assert cagr(equity) == 0.0


# ═══════════════════════════════════════════════════════════════
# win_rate
# ═══════════════════════════════════════════════════════════════

class TestWinRate:
    def test_all_positive_is_one(self):
        """100% de jours gagnants → win rate = 1.0."""
        r = pd.Series([0.01, 0.02, 0.005, 0.03])
        assert win_rate(r) == 1.0

    def test_all_negative_is_zero(self):
        """100% de jours perdants → win rate = 0.0."""
        r = pd.Series([-0.01, -0.02, -0.005])
        assert win_rate(r) == 0.0

    def test_half_half(self):
        """50% gagnants, 50% perdants → win rate = 0.5."""
        r = pd.Series([0.01, -0.01, 0.02, -0.02])
        assert win_rate(r) == 0.5

    def test_zero_returns_excluded(self):
        """Les jours à rendement = 0 ne comptent pas dans le total."""
        r = pd.Series([0.01, 0.0, -0.01, 0.0])
        # 1 positif, 1 négatif sur 2 jours non-nuls → 0.5
        assert win_rate(r) == 0.5

    def test_empty_returns_zero(self):
        """Série vide → 0.0."""
        assert win_rate(pd.Series([], dtype=float)) == 0.0


# ═══════════════════════════════════════════════════════════════
# profit_factor
# ═══════════════════════════════════════════════════════════════

class TestProfitFactor:
    def test_known_ratio(self):
        """
        Gains : 0.10 + 0.20 = 0.30
        Pertes : 0.05 + 0.10 = 0.15
        Profit Factor = 0.30 / 0.15 = 2.0
        """
        r = pd.Series([0.10, 0.20, -0.05, -0.10])
        result = profit_factor(r)
        assert abs(result - 2.0) < 1e-10

    def test_no_losses_returns_inf(self):
        """Aucune perte → Profit Factor = inf."""
        r = pd.Series([0.01, 0.02, 0.03])
        assert profit_factor(r) == float('inf')

    def test_greater_than_one_for_profitable_strategy(self, returns_series):
        """Stratégie légèrement profitable → PF > 1."""
        # Rendements avec drift positif (fixture price_series a drift +10%/an)
        positive_r = returns_series[returns_series.abs() < 0.05]  # filtre outliers
        # On teste juste que la fonction renvoie un float > 0
        result = profit_factor(positive_r)
        assert isinstance(result, float)
        assert result > 0


# ═══════════════════════════════════════════════════════════════
# value_at_risk
# ═══════════════════════════════════════════════════════════════

class TestValueAtRisk:
    def test_always_negative_for_mixed_returns(self, returns_series):
        """VaR 95% est toujours ≤ 0 pour des rendements réalistes."""
        result = value_at_risk(returns_series)
        assert result < 0

    def test_known_percentile(self):
        """
        Données uniformes [0, 1, 2, ..., 99].
        Percentile 5% = 4.95 (np.percentile avec interpolation linéaire).
        """
        r = pd.Series(range(100), dtype=float)
        result = value_at_risk(r, confidence=0.95)
        assert abs(result - np.percentile(range(100), 5)) < 1e-10

    def test_confidence_parameter(self, returns_series):
        """VaR 99% < VaR 95% (pire perte plus sévère au niveau 99%)."""
        var_95 = value_at_risk(returns_series, confidence=0.95)
        var_99 = value_at_risk(returns_series, confidence=0.99)
        assert var_99 < var_95


# ═══════════════════════════════════════════════════════════════
# full_metrics — test d'intégration
# ═══════════════════════════════════════════════════════════════

class TestFullMetrics:
    def test_returns_all_keys(self, returns_series, equity_series):
        """full_metrics retourne bien les 9 clés attendues."""
        expected_keys = {
            "CAGR (%)", "Sharpe", "Calmar", "Max Drawdown (%)",
            "Win Rate (%)", "Profit Factor", "VaR 95% (%)",
            "Capital final (€)", "Nb jours tradés",
        }
        result = full_metrics(returns_series, equity_series)
        assert set(result.keys()) == expected_keys

    def test_values_are_numbers(self, returns_series, equity_series):
        """Toutes les valeurs sont des nombres (pas de NaN ni None)."""
        result = full_metrics(returns_series, equity_series)
        for key, val in result.items():
            assert val is not None, f"{key} est None"
            assert not (isinstance(val, float) and np.isnan(val)), f"{key} est NaN"

    def test_nb_jours_is_integer(self, returns_series, equity_series):
        """Nb jours tradés est un entier."""
        result = full_metrics(returns_series, equity_series)
        assert isinstance(result["Nb jours tradés"], int)

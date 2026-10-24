"""
tests/test_combination.py — Tests unitaires pour backtester/combination.py
"""

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, '.')

from backtester.combination import (
    combine_signals,
    rolling_ic_weights,
    rolling_icir_weights,
    signal_quality_report,
    _normalize_weights,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def dates():
    return pd.date_range("2020-01-01", periods=400, freq="B")


@pytest.fixture
def three_signals(dates):
    np.random.seed(42)
    return {
        "sma":  pd.Series(np.random.choice([-1, 1], 400).astype(float), index=dates),
        "rsi":  pd.Series(np.random.uniform(-1, 1, 400), index=dates),
        "macd": pd.Series(np.random.uniform(-0.5, 0.5, 400), index=dates),
    }


@pytest.fixture
def forward_returns(dates):
    np.random.seed(0)
    return pd.Series(np.random.normal(0.0002, 0.012, 400), index=dates)


# ─────────────────────────────────────────────────────────────────────────────
# Tests : _normalize_weights
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeWeights:
    def test_sums_to_one(self):
        w = pd.Series({"a": 0.3, "b": 0.5, "c": 0.2})
        result = _normalize_weights(w)
        assert abs(result.sum() - 1.0) < 1e-9

    def test_negative_clipped_to_zero(self):
        w = pd.Series({"a": -0.2, "b": 0.6, "c": 0.4})
        result = _normalize_weights(w)
        assert result["a"] == 0.0
        assert abs(result.sum() - 1.0) < 1e-9

    def test_all_zero_returns_equal(self):
        """Si tous les poids sont nuls → fallback poids égaux."""
        w = pd.Series({"a": 0.0, "b": 0.0, "c": 0.0})
        result = _normalize_weights(w)
        assert abs(result["a"] - 1/3) < 1e-9

    def test_all_negative_returns_equal(self):
        w = pd.Series({"a": -0.1, "b": -0.3})
        result = _normalize_weights(w)
        assert abs(result.sum() - 1.0) < 1e-9
        assert abs(result["a"] - 0.5) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Tests : combine_signals — mode 'equal'
# ─────────────────────────────────────────────────────────────────────────────

class TestCombineSignalsEqual:
    def test_returns_series(self, three_signals):
        result = combine_signals(three_signals)
        assert isinstance(result, pd.Series)

    def test_clipped_to_minus_one_one(self, three_signals):
        result = combine_signals(three_signals)
        assert result.max() <= 1.0
        assert result.min() >= -1.0

    def test_equal_weights_is_average(self, three_signals):
        """Mode equal doit être identique à la moyenne simple."""
        result = combine_signals(three_signals, weights_mode="equal")
        manual = sum(three_signals.values()) / 3
        manual = manual.clip(-1, 1)
        pd.testing.assert_series_equal(result, manual, check_names=False, rtol=1e-6)

    def test_same_index(self, three_signals):
        result = combine_signals(three_signals)
        expected_index = list(three_signals.values())[0].index
        assert result.index.equals(expected_index)


# ─────────────────────────────────────────────────────────────────────────────
# Tests : combine_signals — mode 'fixed'
# ─────────────────────────────────────────────────────────────────────────────

class TestCombineSignalsFixed:
    def test_fixed_weights_respected(self, three_signals):
        weights = {"sma": 0.6, "rsi": 0.3, "macd": 0.1}
        result = combine_signals(three_signals, weights_mode="fixed",
                                  fixed_weights=weights)
        # Calcul manuel
        manual = (three_signals["sma"] * 0.6 +
                  three_signals["rsi"] * 0.3 +
                  three_signals["macd"] * 0.1).clip(-1, 1)
        pd.testing.assert_series_equal(result, manual, check_names=False, rtol=1e-6)

    def test_raises_without_fixed_weights(self, three_signals):
        with pytest.raises(ValueError, match="fixed_weights"):
            combine_signals(three_signals, weights_mode="fixed")

    def test_weights_normalized(self, three_signals):
        """Les poids non normalisés doivent être auto-normalisés."""
        weights = {"sma": 6, "rsi": 3, "macd": 1}  # somme = 10, pas 1
        result = combine_signals(three_signals, weights_mode="fixed",
                                  fixed_weights=weights)
        assert result.max() <= 1.0
        assert result.min() >= -1.0


# ─────────────────────────────────────────────────────────────────────────────
# Tests : combine_signals — modes IC et ICIR
# ─────────────────────────────────────────────────────────────────────────────

class TestCombineSignalsIC:
    def test_ic_requires_forward_returns(self, three_signals):
        with pytest.raises(ValueError, match="forward_returns"):
            combine_signals(three_signals, weights_mode="ic")

    def test_icir_requires_forward_returns(self, three_signals):
        with pytest.raises(ValueError, match="forward_returns"):
            combine_signals(three_signals, weights_mode="icir")

    def test_ic_returns_series(self, three_signals, forward_returns):
        result = combine_signals(three_signals, weights_mode="ic",
                                  forward_returns=forward_returns, ic_window=30)
        assert isinstance(result, pd.Series)

    def test_ic_clipped(self, three_signals, forward_returns):
        result = combine_signals(three_signals, weights_mode="ic",
                                  forward_returns=forward_returns, ic_window=30)
        valid = result.dropna()
        assert valid.max() <= 1.001
        assert valid.min() >= -1.001

    def test_icir_returns_series(self, three_signals, forward_returns):
        result = combine_signals(three_signals, weights_mode="icir",
                                  forward_returns=forward_returns, ic_window=30)
        assert isinstance(result, pd.Series)

    def test_unknown_mode_raises(self, three_signals, forward_returns):
        with pytest.raises(ValueError, match="inconnu"):
            combine_signals(three_signals, weights_mode="unknown",
                             forward_returns=forward_returns)


# ─────────────────────────────────────────────────────────────────────────────
# Tests : rolling_ic_weights
# ─────────────────────────────────────────────────────────────────────────────

class TestRollingICWeights:
    def test_returns_dataframe(self, three_signals, forward_returns):
        w = rolling_ic_weights(three_signals, forward_returns, window=30)
        assert isinstance(w, pd.DataFrame)
        assert set(w.columns) == set(three_signals.keys())

    def test_weights_sum_to_one(self, three_signals, forward_returns):
        """Chaque ligne doit sommer à 1."""
        w = rolling_ic_weights(three_signals, forward_returns, window=30)
        row_sums = w.dropna(how="all").sum(axis=1)
        assert (row_sums.abs() - 1.0 < 1e-6).all()

    def test_weights_non_negative(self, three_signals, forward_returns):
        """Pas de poids négatifs (IC négatifs exclus)."""
        w = rolling_ic_weights(three_signals, forward_returns, window=30)
        assert (w.fillna(0) >= -1e-9).all().all()


# ─────────────────────────────────────────────────────────────────────────────
# Tests : signal_quality_report
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalQualityReport:
    def test_returns_dataframe(self, three_signals, forward_returns):
        report = signal_quality_report(three_signals, forward_returns, ic_window=30)
        assert isinstance(report, pd.DataFrame)

    def test_has_expected_columns(self, three_signals, forward_returns):
        report = signal_quality_report(three_signals, forward_returns, ic_window=30)
        assert "IC moyen" in report.columns
        assert "ICIR" in report.columns
        assert "Poids IC (%)" in report.columns

    def test_has_row_per_signal(self, three_signals, forward_returns):
        report = signal_quality_report(three_signals, forward_returns, ic_window=30)
        assert len(report) == len(three_signals)

    def test_poids_sum_to_100(self, three_signals, forward_returns):
        """La somme des poids IC doit être ≈ 100%."""
        report = signal_quality_report(three_signals, forward_returns, ic_window=30)
        assert abs(report["Poids IC (%)"].sum() - 100.0) < 0.1

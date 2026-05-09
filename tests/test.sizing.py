"""
tests/test_sizing.py — Tests unitaires pour backtester/sizing.py

Invariants à tester pour chaque mode de sizing :
  1. Bornes : position toujours dans [0, 1] (pas de levier, pas de short)
  2. Shift  : première valeur = 0 (shift(1).fillna(0))
  3. Direction : signal négatif → position = 0 (long-only)
  4. Cohérence : signal = 0 ou négatif → pas de position
"""
import pytest
import pandas as pd
import numpy as np
from backtester.sizing import (
    vol_target_size, kelly_size, fixed_fraction,
    signal_proportional_size, atr_stop,
)


# ═══════════════════════════════════════════════════════════════
# Helpers locaux
# ═══════════════════════════════════════════════════════════════

def _all_positive_signal(n=400):
    """Signal = +1 constant (toujours long)."""
    return pd.Series([1.0] * n)

def _all_negative_signal(n=400):
    """Signal = -1 constant (jamais long)."""
    return pd.Series([-1.0] * n)


# ═══════════════════════════════════════════════════════════════
# vol_target_size
# ═══════════════════════════════════════════════════════════════

class TestVolTargetSize:
    def test_bounded_zero_to_one(self, returns_series):
        sig = _all_positive_signal(len(returns_series))
        pos = vol_target_size(returns_series, sig)
        assert pos.between(0.0, 1.0).all()

    def test_first_value_is_zero(self, returns_series):
        """shift(1).fillna(0) → première valeur = 0."""
        sig = _all_positive_signal(len(returns_series))
        pos = vol_target_size(returns_series, sig)
        assert pos.iloc[0] == 0.0

    def test_negative_signal_gives_zero_position(self, returns_series):
        """Signal négatif → direction = 0 → position = 0."""
        sig = _all_negative_signal(len(returns_series))
        pos = vol_target_size(returns_series, sig)
        # Après le warmup (window=20), tout doit être à 0
        assert (pos.iloc[21:] == 0).all()

    def test_high_vol_reduces_position(self, returns_series):
        """
        Vol élevée → taille réduite (vol_target / vol_realized < 1).
        On compare deux périodes : une calme et une agitée.
        """
        np.random.seed(42)
        calm = pd.Series(np.random.normal(0, 0.003, 200))    # vol ~5%/an
        volatile = pd.Series(np.random.normal(0, 0.020, 200)) # vol ~32%/an
        sig = _all_positive_signal(200)

        pos_calm     = vol_target_size(calm, sig, vol_target=0.10)
        pos_volatile = vol_target_size(volatile, sig, vol_target=0.10)

        # En période calme, on investit davantage qu'en période agitée
        assert pos_calm.iloc[21:].mean() > pos_volatile.iloc[21:].mean()


# ═══════════════════════════════════════════════════════════════
# kelly_size
# ═══════════════════════════════════════════════════════════════

class TestKellySize:
    def test_bounded_zero_to_one(self, returns_series):
        sig = _all_positive_signal(len(returns_series))
        pos = kelly_size(returns_series, sig, window=100)
        assert pos.between(0.0, 1.0).all()

    def test_negative_signal_zero_position(self, returns_series):
        sig = _all_negative_signal(len(returns_series))
        pos = kelly_size(returns_series, sig, window=100)
        assert (pos == 0).all()

    def test_half_kelly_less_than_full_kelly(self, returns_series):
        """
        Demi-Kelly (fraction=0.5) doit donner des positions
        plus petites que le Kelly plein (fraction=1.0).
        """
        sig = _all_positive_signal(len(returns_series))
        half = kelly_size(returns_series, sig, window=100, fraction=0.5)
        full = kelly_size(returns_series, sig, window=100, fraction=1.0)
        # En moyenne, les positions demi-Kelly sont plus petites
        assert half.mean() <= full.mean()


# ═══════════════════════════════════════════════════════════════
# fixed_fraction
# ═══════════════════════════════════════════════════════════════

class TestFixedFraction:
    def test_fraction_applied_correctly(self):
        """Signal positif → position exactement = fraction."""
        sig = _all_positive_signal(50)
        pos = fixed_fraction(sig, fraction=0.05)
        # Toutes les valeurs (sauf la première = 0) égales à 0.05
        assert (pos.iloc[1:] == 0.05).all()

    def test_first_is_zero(self):
        sig = _all_positive_signal(50)
        pos = fixed_fraction(sig, fraction=0.05)
        assert pos.iloc[0] == 0.0

    def test_negative_signal_zero(self):
        sig = _all_negative_signal(50)
        pos = fixed_fraction(sig, fraction=0.10)
        assert (pos == 0.0).all()

    def test_default_fraction(self):
        """Fraction par défaut = 0.02."""
        sig = pd.Series([1.0] * 10)
        pos = fixed_fraction(sig)
        assert (pos.iloc[1:] == 0.02).all()


# ═══════════════════════════════════════════════════════════════
# signal_proportional_size
# ═══════════════════════════════════════════════════════════════

class TestSignalProportionalSize:
    def test_bounded(self, returns_series):
        sig = pd.Series(np.linspace(0, 1, len(returns_series)))
        pos = signal_proportional_size(sig, returns_series)
        assert pos.between(0.0, 1.0).all()

    def test_stronger_signal_bigger_position(self, returns_series):
        """
        Signal fort (0.9) → position plus grande que signal faible (0.2).
        Comparaison sur des séries constantes pour isoler l'effet du signal.
        """
        n = len(returns_series)
        sig_strong = pd.Series([0.9] * n, index=returns_series.index)
        sig_weak   = pd.Series([0.2] * n, index=returns_series.index)

        pos_strong = signal_proportional_size(sig_strong, returns_series, vol_adjust=False)
        pos_weak   = signal_proportional_size(sig_weak,   returns_series, vol_adjust=False)

        # Signal fort → position moyenne plus grande
        assert pos_strong.iloc[1:].mean() > pos_weak.iloc[1:].mean()

    def test_power_convexity(self, returns_series):
        """
        power=2 (convexe) pénalise les signaux tièdes vs power=1 (linéaire).
        Pour un signal modéré (0.5) : 0.5^2 = 0.25 < 0.5^1 = 0.5
        """
        n = len(returns_series)
        sig = pd.Series([0.5] * n, index=returns_series.index)
        pos_linear = signal_proportional_size(sig, returns_series, vol_adjust=False, power=1.0)
        pos_convex = signal_proportional_size(sig, returns_series, vol_adjust=False, power=2.0)
        assert pos_linear.iloc[1:].mean() > pos_convex.iloc[1:].mean()

    def test_vol_adjust_reduces_position_in_volatile_market(self):
        """vol_adjust=True réduit la position quand la volatilité monte."""
        np.random.seed(42)
        volatile_returns = pd.Series(np.random.normal(0, 0.03, 300))
        calm_returns     = pd.Series(np.random.normal(0, 0.005, 300))
        sig = pd.Series([0.8] * 300)

        pos_vol  = signal_proportional_size(sig, volatile_returns, vol_adjust=True)
        pos_calm = signal_proportional_size(sig, calm_returns,     vol_adjust=True)

        assert pos_calm.iloc[21:].mean() > pos_vol.iloc[21:].mean()


# ═══════════════════════════════════════════════════════════════
# atr_stop
# ═══════════════════════════════════════════════════════════════

class TestAtrStop:
    def test_returns_correct_columns(self, ohlcv_df):
        """atr_stop retourne un DataFrame avec les 3 colonnes attendues."""
        sig = pd.Series([1.0] * len(ohlcv_df), index=ohlcv_df.index)
        result = atr_stop(ohlcv_df, sig)
        assert set(result.columns) == {"atr", "stop_level", "position_atr"}

    def test_atr_positive(self, ohlcv_df):
        """ATR toujours positif (c'est une mesure de volatilité)."""
        sig = pd.Series([1.0] * len(ohlcv_df), index=ohlcv_df.index)
        result = atr_stop(ohlcv_df, sig)
        assert (result["atr"].dropna() > 0).all()

    def test_stop_level_below_close(self, ohlcv_df):
        """
        Stop = Close - ATR × multiplier.
        Le niveau de stop est toujours sous le prix de clôture.
        """
        sig = pd.Series([1.0] * len(ohlcv_df), index=ohlcv_df.index)
        result = atr_stop(ohlcv_df, sig, atr_multiplier=2.0)
        valid_idx = result["atr"].dropna().index
        assert (result.loc[valid_idx, "stop_level"] <
                ohlcv_df.loc[valid_idx, "Close"]).all()

    def test_position_bounded(self, ohlcv_df):
        """Position ATR entre 0 et 1."""
        sig = pd.Series([1.0] * len(ohlcv_df), index=ohlcv_df.index)
        result = atr_stop(ohlcv_df, sig)
        assert result["position_atr"].between(0.0, 1.0).all()

    def test_negative_signal_no_position(self, ohlcv_df):
        """Signal toujours négatif → jamais de position."""
        sig = pd.Series([-1.0] * len(ohlcv_df), index=ohlcv_df.index)
        result = atr_stop(ohlcv_df, sig)
        assert (result["position_atr"] == 0).all()

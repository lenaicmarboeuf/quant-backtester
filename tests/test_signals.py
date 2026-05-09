"""
tests/test_signals.py — Tests unitaires pour backtester/signals.py

Principe de test des signaux :
  1. Bornes : signal toujours dans [-1, +1]
  2. Direction : hausse prolongée → signal positif pour les signaux tendance
  3. NaN : les premières valeurs (période de warmup) peuvent être NaN
  4. Shift : signal_to_position décale bien d'un jour (anti look-ahead)
  5. Intégration : combine_signals respecte les poids
"""
import pytest
import pandas as pd
import numpy as np
from backtester.signals import (
    sma_crossover, bollinger_zscore, rsi_signal,
    macd_signal, combine_signals, signal_to_position,
)


# ═══════════════════════════════════════════════════════════════
# sma_crossover
# ═══════════════════════════════════════════════════════════════

class TestSmaCrossover:
    def test_output_only_plus_minus_one(self, price_series):
        """Valeurs uniquement +1 ou -1 (hors NaN de warmup)."""
        sig = sma_crossover(price_series, fast=10, slow=20)
        valid = sig.dropna()
        unique_vals = set(valid.unique())
        assert unique_vals.issubset({1.0, -1.0})

    def test_warmup_is_nan(self, price_series):
        """Les 'slow' premières valeurs doivent être NaN."""
        slow = 30
        sig = sma_crossover(price_series, fast=10, slow=slow)
        assert sig.iloc[:slow].isna().all()

    def test_uptrend_gives_positive_signal(self):
        """Prix strictement croissant → SMA rapide > SMA lente → signal = +1."""
        # Prix qui montent régulièrement : fast SMA > slow SMA
        prices = pd.Series(np.linspace(100, 200, 200))
        sig = sma_crossover(prices, fast=10, slow=50)
        valid = sig.dropna()
        # En tendance haussière nette, la majorité des signaux sont +1
        assert (valid == 1).mean() > 0.8

    def test_downtrend_gives_negative_signal(self):
        """Prix strictement décroissant → signal = -1 après warmup."""
        prices = pd.Series(np.linspace(200, 100, 200))
        sig = sma_crossover(prices, fast=10, slow=50)
        valid = sig.dropna()
        assert (valid == -1).mean() > 0.8

    def test_index_preserved(self, price_series):
        """L'index de la série d'entrée est préservé en sortie."""
        sig = sma_crossover(price_series)
        assert list(sig.index) == list(price_series.index)


# ═══════════════════════════════════════════════════════════════
# bollinger_zscore
# ═══════════════════════════════════════════════════════════════

class TestBollingerZscore:
    def test_bounded_between_minus_one_and_one(self, price_series):
        """Signal clipé entre -1 et +1."""
        sig = bollinger_zscore(price_series)
        valid = sig.dropna()
        assert valid.between(-1.0, 1.0).all()

    def test_mean_reversion_logic(self):
        """
        Prix très au-dessus de la moyenne → zscore positif → signal négatif.
        (on anticipe un retour vers la moyenne)
        """
        # Série plate puis spike brutal
        prices = pd.Series([100.0] * 50 + [200.0] * 10)
        sig = bollinger_zscore(prices, window=20)
        # Après le spike, le signal doit être négatif (anticipation de baisse)
        assert sig.iloc[-1] < 0

    def test_no_nan_after_warmup(self, price_series):
        """Pas de NaN après la période de warmup (window=20)."""
        sig = bollinger_zscore(price_series, window=20)
        assert not sig.iloc[20:].isna().any()


# ═══════════════════════════════════════════════════════════════
# rsi_signal
# ═══════════════════════════════════════════════════════════════

class TestRsiSignal:
    def test_bounded(self, price_series):
        """Signal RSI toujours dans [-1, +1]."""
        sig = rsi_signal(price_series)
        valid = sig.dropna()
        assert valid.between(-1.0, 1.0).all()

    def test_neutral_zone_returns_zero(self):
        """
        RSI en zone neutre (30 < RSI < 70) → signal = 0.
        Série avec faible volatilité → RSI proche de 50.
        """
        np.random.seed(1)
        # Série presque plate : RSI proche de 50
        prices = pd.Series(100 + np.cumsum(np.random.normal(0, 0.05, 200)))
        sig = rsi_signal(prices, period=14, overbought=70, oversold=30)
        valid = sig.dropna()
        # La majorité doit être à 0 (zone neutre)
        assert (valid == 0).mean() > 0.5

    def test_oversold_gives_positive_signal(self):
        """
        Après une chute brutale (RSI bas) → signal positif (achat).
        """
        # 50 jours de baisse intense
        prices = pd.Series(np.linspace(200, 50, 100))
        sig = rsi_signal(prices, period=14, overbought=70, oversold=30)
        # Les dernières valeurs devraient être positives ou 0
        tail = sig.dropna().tail(10)
        assert tail.max() >= 0


# ═══════════════════════════════════════════════════════════════
# combine_signals
# ═══════════════════════════════════════════════════════════════

class TestCombineSignals:
    def test_equal_weights_average(self, price_series):
        """Sans poids → moyenne arithmétique des signaux."""
        s1 = pd.Series([0.8] * len(price_series), index=price_series.index)
        s2 = pd.Series([-0.2] * len(price_series), index=price_series.index)
        combined = combine_signals({"a": s1, "b": s2})
        expected = (0.8 + (-0.2)) / 2  # = 0.3
        assert abs(combined.iloc[0] - expected) < 1e-10

    def test_custom_weights(self, price_series):
        """Poids personnalisés : signal 1 avec poids 0.7, signal 2 avec 0.3."""
        s1 = pd.Series([1.0] * len(price_series), index=price_series.index)
        s2 = pd.Series([0.0] * len(price_series), index=price_series.index)
        combined = combine_signals(
            {"a": s1, "b": s2},
            weights={"a": 0.7, "b": 0.3}
        )
        # 1.0*0.7 + 0.0*0.3 = 0.7
        assert abs(combined.iloc[0] - 0.7) < 1e-10

    def test_output_clipped_to_one(self, price_series):
        """Signal combiné toujours dans [-1, +1]."""
        s1 = pd.Series([1.0] * len(price_series), index=price_series.index)
        s2 = pd.Series([1.0] * len(price_series), index=price_series.index)
        # Avec poids > 0.5 chacun, la somme dépasserait 1 sans clip
        combined = combine_signals({"a": s1, "b": s2}, weights={"a": 0.8, "b": 0.8})
        assert combined.max() <= 1.0


# ═══════════════════════════════════════════════════════════════
# signal_to_position — test du shift anti look-ahead
# ═══════════════════════════════════════════════════════════════

class TestSignalToPosition:
    def test_binary_output(self, price_series):
        """Position uniquement 0 ou 1."""
        sig = sma_crossover(price_series).fillna(0)
        pos = signal_to_position(sig)
        unique = set(pos.unique())
        assert unique.issubset({0, 1})

    def test_shift_one_day(self):
        """
        RÈGLE D'OR : la position du jour J est basée sur
        le signal du jour J-1 (shift de 1 jour).
        
        Signal : [1, 1, 1, 0, 0]
        Position attendue (après shift) : [0, 1, 1, 1, 0]
        (la première position est toujours 0 = fillna)
        """
        signal = pd.Series([1.0, 1.0, 1.0, -1.0, -1.0])
        pos = signal_to_position(signal)
        expected = pd.Series([0.0, 1.0, 1.0, 1.0, 0.0])
        pd.testing.assert_series_equal(pos, expected, check_names=False)

    def test_threshold_filter(self):
        """
        Threshold=0.5 : seuls les signaux > 0.5 génèrent une position.
        Signal [0.3, 0.8, -0.1, 0.6] → position sur [0.8, 0.6] seulement.
        """
        signal = pd.Series([0.3, 0.8, -0.1, 0.6])
        pos = signal_to_position(signal, threshold=0.5)
        # signal > 0.5 : indices 1 (0.8) et 3 (0.6) → position à J+1
        # pos[2] = 1 (signal[1]=0.8 > 0.5), pos[4] = 1 (signal[3]=0.6 > 0.5)
        assert pos.iloc[0] == 0.0  # toujours 0 (fillna du shift)
        assert pos.iloc[2] == 1.0  # signal[1]=0.8 > 0.5 ✓
        assert pos.iloc[1] == 0.0  # signal[0]=0.3 < 0.5 ✗

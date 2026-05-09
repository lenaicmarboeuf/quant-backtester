"""
tests/test_engine.py — Tests d'intégration pour backtester/engine.py

Tests d'intégration = on teste le pipeline complet, pas les fonctions isolées.
Signal → Positions → PnL → Métriques : tout doit fonctionner ensemble.

Invariants du moteur de backtest :
  1. L'equity commence toujours à initial_capital
  2. Sans position (signal constant = -1), l'equity ne varie pas
  3. Les résultats contiennent les colonnes attendues
  4. Tous les sizing_modes s'exécutent sans erreur
"""
import pytest
import pandas as pd
import numpy as np
from backtester.engine import Backtester
from backtester.signals import sma_crossover


# ═══════════════════════════════════════════════════════════════
# Fixtures locales à engine
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def bt(ohlcv_df):
    """Instance Backtester avec données synthétiques."""
    return Backtester(ohlcv_df, initial_capital=10_000.0)

@pytest.fixture
def constant_signal(ohlcv_df):
    """Signal = +1 partout (toujours long)."""
    return pd.Series([1.0] * len(ohlcv_df), index=ohlcv_df.index)

@pytest.fixture
def no_trade_signal(ohlcv_df):
    """Signal = -1 partout (jamais long)."""
    return pd.Series([-1.0] * len(ohlcv_df), index=ohlcv_df.index)


# ═══════════════════════════════════════════════════════════════
# compute_pnl — tests sur la structure des résultats
# ═══════════════════════════════════════════════════════════════

class TestComputePnl:
    REQUIRED_COLS = {"position", "strategy_returns", "cumulative_returns", "equity_curve"}

    def test_output_columns(self, bt, constant_signal):
        """Le DataFrame résultat contient bien les 4 colonnes obligatoires."""
        position = (constant_signal > 0).astype(int).shift(1).fillna(0)
        results = bt.compute_pnl(position)
        assert self.REQUIRED_COLS.issubset(set(results.columns))

    def test_equity_starts_at_initial_capital(self, bt, no_trade_signal):
        """
        Avec position = 0 (jamais investi), l'equity reste à initial_capital.
        strategy_returns = 0 × returns = 0 → equity = 10000 constant.
        Note : on teste dropna() car returns[0] = NaN (pct_change).
        """
        position = (no_trade_signal > 0).astype(int).shift(1).fillna(0)
        results = bt.compute_pnl(position)
        assert (results["equity_curve"].dropna() == 10_000.0).all()

    def test_no_trade_equity_flat(self, bt, no_trade_signal):
        """
        Aucune position → strategy_returns = 0 → equity constante.
        Note : ligne 0 est NaN (pct_change), on ignore avec dropna().
        """
        position = pd.Series(0.0, index=bt.df.index)
        results = bt.compute_pnl(position)
        assert (results["strategy_returns"].dropna() == 0).all()
        assert (results["equity_curve"].dropna() == 10_000.0).all()


# ═══════════════════════════════════════════════════════════════
# run() — tests d'intégration complets
# ═══════════════════════════════════════════════════════════════

class TestRun:
    def test_run_returns_results_and_metrics(self, bt, ohlcv_df):
        """run() retourne un dict avec les clés 'results' et 'metrics'."""
        signal = sma_crossover(ohlcv_df["Close"], fast=10, slow=30).fillna(0)
        output = bt.run(signal)
        assert "results" in output
        assert "metrics" in output

    def test_results_stored_in_self(self, bt, ohlcv_df):
        """Après run(), bt.results est renseigné (utile pour les plots)."""
        signal = sma_crossover(ohlcv_df["Close"], fast=10, slow=30).fillna(0)
        bt.run(signal)
        assert bt.results is not None

    def test_all_sizing_modes_run_without_error(self, bt, ohlcv_df):
        """
        Test de fumée (smoke test) : tous les sizing_modes s'exécutent.
        On ne vérifie pas les valeurs, juste l'absence d'exception.
        """
        signal = sma_crossover(ohlcv_df["Close"], fast=10, slow=30).fillna(0)
        modes = [
            ("binary",      {}),
            ("vol_target",  {"vol_target": 0.10}),
            ("kelly",       {"window": 100, "fraction": 0.5}),
            ("fixed",       {"fraction": 0.05}),
            ("proportional",{"vol_adjust": False}),
        ]
        for mode, kwargs in modes:
            output = bt.run(signal, sizing_mode=mode, **kwargs)
            assert "metrics" in output, f"Mode {mode} n'a pas retourné de métriques"

    def test_metrics_keys_complete(self, bt, ohlcv_df):
        """Les métriques contiennent toutes les clés attendues."""
        signal = sma_crossover(ohlcv_df["Close"], fast=10, slow=30).fillna(0)
        output = bt.run(signal)
        expected_keys = {
            "CAGR (%)", "Sharpe", "Calmar", "Max Drawdown (%)",
            "Win Rate (%)", "Profit Factor", "VaR 95% (%)",
            "Capital final (€)", "Nb jours tradés",
        }
        assert expected_keys.issubset(set(output["metrics"].keys()))

    def test_no_look_ahead_bias(self, bt, ohlcv_df):
        """
        Test du look-ahead bias : si on utilise le signal du MÊME jour
        (sans shift), les résultats sont différents de ceux avec shift.
        
        Avec shift (correct) : on ne peut pas "voir" le signal actuel.
        Sans shift (biaisé)  : on utilise une info du futur.
        
        Les performances biaisées sont souvent meilleures — c'est le danger.
        """
        signal = sma_crossover(ohlcv_df["Close"], fast=10, slow=30).fillna(0)

        # Version correcte (shift appliqué dans compute_positions)
        output_correct = bt.run(signal, sizing_mode="binary")
        equity_correct = output_correct["results"]["equity_curve"]

        # Version biaisée : on injecte manuellement la position non-shiftée
        biased_position = (signal > 0).astype(float)  # SANS shift(1)
        results_biased = bt.compute_pnl(biased_position)
        equity_biased = results_biased["equity_curve"]

        # Les deux versions doivent donner des résultats différents
        assert not equity_correct.equals(equity_biased), (
            "DANGER : version avec et sans shift donnent les mêmes résultats. "
            "Vérifier que le shift est bien appliqué."
        )

    def test_initial_capital_respected(self, bt, ohlcv_df):
        """Capital initial de 5000€ → première equity non-NaN = 5000€."""
        bt_custom = Backtester(ohlcv_df, initial_capital=5_000.0)
        signal = sma_crossover(ohlcv_df["Close"], fast=10, slow=30).fillna(0)
        output = bt_custom.run(signal)
        first_equity = output["results"]["equity_curve"].dropna().iloc[0]
        assert abs(first_equity - 5_000.0) < 1e-6

"""
tests/test_costs.py — Tests unitaires et d'intégration pour backtester/costs.py
              et les modifications de engine.py (séance 7)

Questions clés que ces tests répondent :
  1. Les coûts réduisent-ils bien les rendements ?        (test_costs_reduce_returns)
  2. La stratégie survit-elle à 20 bps A/R ?              (test_survives_20bps)
  3. Un turnover nul génère-t-il zéro coût ?              (test_zero_turnover_zero_cost)
  4. Les modèles de coûts restent-ils dans des bornes réalistes ?
  5. L'analyse de sensibilité fonctionne-t-elle ?         (test_cost_sensitivity)
"""
import pytest
import numpy as np
import pandas as pd

from backtester.costs import (
    commission_cost,
    spread_cost,
    slippage_model,
    market_impact,
    total_transaction_cost,
    turnover_analysis,
    breakeven_cost,
)
from backtester.engine import Backtester
from backtester.signals import sma_crossover


# ═══════════════════════════════════════════════════════════════
# commission_cost
# ═══════════════════════════════════════════════════════════════

class TestCommissionCost:
    def test_always_non_positive(self, ohlcv_df):
        """Les commissions sont des coûts → jamais positives."""
        pos = pd.Series([0.0, 1.0, 1.0, 0.0, 1.0] * 100, index=ohlcv_df.index)
        cost = commission_cost(ohlcv_df["Close"], pos, rate_bps=5.0)
        assert (cost <= 0).all()

    def test_zero_turnover_zero_cost(self, ohlcv_df):
        """Position constante → aucun trade → aucune commission."""
        pos  = pd.Series(1.0, index=ohlcv_df.index)  # jamais de changement
        cost = commission_cost(ohlcv_df["Close"], pos, rate_bps=10.0)
        # diff() d'une série constante = 0 → coût = 0 (sauf première valeur)
        assert (cost.iloc[1:] == 0.0).all()

    def test_known_value(self):
        """
        Vérification mathématique :
        Turnover = 1.0 (on passe de 0 à 1), commission = 5 bps = 0.05%
        Coût = -1.0 × 0.0005 = -0.0005
        """
        pos  = pd.Series([0.0, 1.0, 1.0, 0.0])
        cost = commission_cost(pd.Series([100, 100, 100, 100]), pos, rate_bps=5.0)
        # Jour 1 : turnover = 1.0 → coût = -0.0005
        assert abs(cost.iloc[1] - (-0.0005)) < 1e-10
        # Jour 2 : turnover = 0.0 → coût = 0
        assert cost.iloc[2] == 0.0

    def test_higher_rate_higher_cost(self, ohlcv_df):
        """Commission 10 bps > Commission 5 bps en valeur absolue."""
        pos  = sma_crossover(ohlcv_df["Close"]).fillna(0).abs()
        cost_5  = commission_cost(ohlcv_df["Close"], pos, rate_bps=5.0)
        cost_10 = commission_cost(ohlcv_df["Close"], pos, rate_bps=10.0)
        assert cost_10.sum() < cost_5.sum()  # les deux sont négatifs


# ═══════════════════════════════════════════════════════════════
# spread_cost
# ═══════════════════════════════════════════════════════════════

class TestSpreadCost:
    def test_always_non_positive(self, ohlcv_df):
        pos  = pd.Series(np.random.choice([0.0, 1.0], len(ohlcv_df)), index=ohlcv_df.index)
        cost = spread_cost(ohlcv_df["Close"], pos)
        assert (cost <= 0).all()

    def test_proportional_to_spread(self, ohlcv_df):
        """Double spread → double coût."""
        pos     = pd.Series(np.random.choice([0.0, 1.0], len(ohlcv_df)), index=ohlcv_df.index)
        cost_1  = spread_cost(ohlcv_df["Close"], pos, half_spread_bps=1.0)
        cost_2  = spread_cost(ohlcv_df["Close"], pos, half_spread_bps=2.0)
        ratio   = cost_2.sum() / (cost_1.sum() + 1e-12)
        assert abs(ratio - 2.0) < 1e-6


# ═══════════════════════════════════════════════════════════════
# slippage_model
# ═══════════════════════════════════════════════════════════════

class TestSlippageModel:
    def test_always_non_positive(self, ohlcv_df):
        pos  = pd.Series(np.random.choice([0.0, 1.0], len(ohlcv_df)), index=ohlcv_df.index)
        cost = slippage_model(ohlcv_df["Close"], pos, fixed_bps=2.0)
        assert (cost <= 0).all()

    def test_higher_vol_higher_slippage(self):
        """
        Marché volatile → plus de slippage qu'un marché calme.
        Le modèle inclut une composante proportionnelle à la vol.
        """
        np.random.seed(42)
        n    = 300
        # Prix calme : vol ~5%/an
        calm_prices    = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0, 0.003, n))))
        # Prix volatile : vol ~32%/an
        volatile_prices= pd.Series(100 * np.exp(np.cumsum(np.random.normal(0, 0.020, n))))
        pos  = pd.Series(np.random.choice([0.0, 1.0], n))

        slip_calm     = slippage_model(calm_prices,     pos, fixed_bps=2.0, vol_multiplier=0.1)
        slip_volatile = slippage_model(volatile_prices, pos, fixed_bps=2.0, vol_multiplier=0.1)

        # En valeur absolue, le slippage est plus élevé en période volatile
        assert slip_volatile.abs().mean() > slip_calm.abs().mean()


# ═══════════════════════════════════════════════════════════════
# total_transaction_cost
# ═══════════════════════════════════════════════════════════════

class TestTotalTransactionCost:
    def test_always_non_positive(self, ohlcv_df):
        """Le coût total est toujours ≤ 0 (c'est une perte)."""
        pos  = sma_crossover(ohlcv_df["Close"]).fillna(0).clip(0, 1)
        cost = total_transaction_cost(ohlcv_df["Close"], pos)
        assert (cost <= 0).all()

    def test_zero_position_zero_cost(self, ohlcv_df):
        """Aucun trade → aucun coût."""
        pos  = pd.Series(0.0, index=ohlcv_df.index)
        cost = total_transaction_cost(ohlcv_df["Close"], pos)
        assert (cost == 0.0).all()

    def test_cost_in_realistic_range(self, ohlcv_df):
        """
        Coût journalier moyen sur les jours tradés doit être < 100 bps.
        Au-delà = probablement un bug de calcul.
        """
        pos  = sma_crossover(ohlcv_df["Close"]).fillna(0).clip(0, 1)
        cost = total_transaction_cost(
            ohlcv_df["Close"], pos,
            commission_bps=5.0, spread_bps=2.5, slippage_bps=2.0
        )
        trading_days_cost = cost[cost < 0]
        if len(trading_days_cost) > 0:
            avg_cost_bps = trading_days_cost.abs().mean() * 10_000
            assert avg_cost_bps < 100, f"Coût moyen trop élevé : {avg_cost_bps:.1f} bps"


# ═══════════════════════════════════════════════════════════════
# turnover_analysis
# ═══════════════════════════════════════════════════════════════

class TestTurnoverAnalysis:
    def test_returns_all_keys(self, ohlcv_df):
        """turnover_analysis retourne toutes les clés attendues."""
        sig = sma_crossover(ohlcv_df["Close"]).fillna(0)
        pos = (sig > 0).astype(float).shift(1).fillna(0)
        r   = pos * ohlcv_df["returns"]
        result = turnover_analysis(pos, r)
        expected = {
            "daily_turnover (%)", "annual_turnover (×)",
            "gross_sharpe", "net_sharpe", "sharpe_drag",
            "cost_per_year (%)", "breakeven_turnover (×)",
        }
        assert expected.issubset(set(result.keys()))

    def test_net_sharpe_leq_gross_sharpe(self, ohlcv_df):
        """Le Sharpe net doit être ≤ au Sharpe brut (les coûts drainent)."""
        sig = sma_crossover(ohlcv_df["Close"]).fillna(0)
        pos = (sig > 0).astype(float).shift(1).fillna(0)
        r   = pos * ohlcv_df["returns"]
        result = turnover_analysis(pos, r, cost_per_turnover_bps=20.0)
        assert result["net_sharpe"] <= result["gross_sharpe"]

    def test_zero_cost_no_drag(self, ohlcv_df):
        """Coût = 0 → Sharpe net = Sharpe brut."""
        sig = sma_crossover(ohlcv_df["Close"]).fillna(0)
        pos = (sig > 0).astype(float).shift(1).fillna(0)
        r   = pos * ohlcv_df["returns"]
        result = turnover_analysis(pos, r, cost_per_turnover_bps=0.0)
        assert abs(result["net_sharpe"] - result["gross_sharpe"]) < 1e-6


# ═══════════════════════════════════════════════════════════════
# Backtester avec coûts — tests d'intégration
# ═══════════════════════════════════════════════════════════════

class TestBacktesterWithCosts:
    def test_costs_reduce_returns(self, ohlcv_df):
        """
        Invariant fondamental : avec coûts > sans coûts.
        Le capital final net doit être ≤ au capital final brut.
        """
        signal = sma_crossover(ohlcv_df["Close"], fast=10, slow=30).fillna(0)

        # Sans coûts
        bt_free = Backtester(ohlcv_df, initial_capital=10_000.0, transaction_costs=None)
        out_free = bt_free.run(signal)

        # Avec coûts (10 bps A/R)
        bt_cost = Backtester(
            ohlcv_df, initial_capital=10_000.0,
            transaction_costs={"commission_bps": 5.0, "spread_bps": 2.5, "slippage_bps": 2.0}
        )
        out_cost = bt_cost.run(signal)

        capital_free = out_free["results"]["equity_curve"].iloc[-1]
        capital_cost = out_cost["results"]["equity_curve"].iloc[-1]

        assert capital_cost <= capital_free, (
            "ERREUR : les coûts n'ont pas réduit le capital final. "
            "Vérifier que cost_returns est bien soustrait des rendements."
        )

    def test_zero_costs_equals_no_costs(self, ohlcv_df):
        """
        Coûts à 0 bps → résultats identiques à sans coûts.
        Vérifie la cohérence entre les deux modes.
        """
        signal = sma_crossover(ohlcv_df["Close"], fast=10, slow=30).fillna(0)

        bt_free = Backtester(ohlcv_df, initial_capital=10_000.0, transaction_costs=None)
        out_free = bt_free.run(signal)

        bt_zero = Backtester(
            ohlcv_df, initial_capital=10_000.0,
            transaction_costs={"commission_bps": 0.0, "spread_bps": 0.0, "slippage_bps": 0.0}
        )
        out_zero = bt_zero.run(signal)

        eq_free = out_free["results"]["equity_curve"].dropna()
        eq_zero = out_zero["results"]["equity_curve"].dropna()

        pd.testing.assert_series_equal(eq_free, eq_zero, check_names=False, rtol=1e-6)

    def test_survives_20bps(self, ohlcv_df):
        """
        Test de robustesse : la stratégie SMA doit être évaluée à 20 bps A/R.
        Ce test NE vérifie PAS que la stratégie est profitable —
        il vérifie que le calcul fonctionne et que le Sharpe est un nombre.
        """
        signal = sma_crossover(ohlcv_df["Close"], fast=10, slow=30).fillna(0)
        bt = Backtester(
            ohlcv_df, initial_capital=10_000.0,
            transaction_costs={"commission_bps": 10.0, "spread_bps": 5.0, "slippage_bps": 5.0}
        )
        out = bt.run(signal)
        sharpe_net = out["metrics"]["Sharpe"]
        assert isinstance(sharpe_net, float), "Sharpe net n'est pas un float"
        assert not np.isnan(sharpe_net), "Sharpe net est NaN avec 20 bps de coûts"

    def test_cost_analysis_in_output(self, ohlcv_df):
        """Avec coûts activés → 'cost_analysis' et 'gross_metrics' dans le output."""
        signal = sma_crossover(ohlcv_df["Close"], fast=10, slow=30).fillna(0)
        bt = Backtester(
            ohlcv_df, initial_capital=10_000.0,
            transaction_costs={"commission_bps": 5.0}
        )
        out = bt.run(signal)
        assert "cost_analysis" in out, "cost_analysis manquant dans le output"
        assert "gross_metrics" in out, "gross_metrics manquant dans le output"

    def test_cost_sensitivity_shape(self, ohlcv_df):
        """cost_sensitivity retourne un DataFrame avec N lignes (une par niveau de coût)."""
        signal = sma_crossover(ohlcv_df["Close"], fast=10, slow=30).fillna(0)
        bt = Backtester(ohlcv_df, initial_capital=10_000.0, transaction_costs={})
        costs = [0, 10, 20, 30]
        result = bt.cost_sensitivity(signal, cost_range_bps=costs)
        assert len(result) == len(costs), f"Attendu {len(costs)} lignes, obtenu {len(result)}"

    def test_higher_costs_lower_net_sharpe(self, ohlcv_df):
        """
        Invariant : Sharpe(0 bps) ≥ Sharpe(10 bps) ≥ Sharpe(30 bps).
        Plus les coûts augmentent, plus le Sharpe net baisse.
        """
        signal = sma_crossover(ohlcv_df["Close"], fast=10, slow=30).fillna(0)
        bt = Backtester(ohlcv_df, initial_capital=10_000.0, transaction_costs={})
        result = bt.cost_sensitivity(signal, cost_range_bps=[0, 10, 30])

        sharpe_0  = result.loc[0,  "Sharpe"]
        sharpe_10 = result.loc[10, "Sharpe"]
        sharpe_30 = result.loc[30, "Sharpe"]

        assert sharpe_0 >= sharpe_10, "Sharpe 0bps < Sharpe 10bps → bug dans les coûts"
        assert sharpe_10 >= sharpe_30, "Sharpe 10bps < Sharpe 30bps → bug dans les coûts"


# ═══════════════════════════════════════════════════════════════
# breakeven_cost
# ═══════════════════════════════════════════════════════════════

class TestBreakevenCost:
    def test_positive_sharpe_positive_breakeven(self):
        """Stratégie profitable → breakeven positif."""
        be = breakeven_cost(gross_sharpe=1.2, annual_turnover=50)
        assert be > 0

    def test_zero_turnover_returns_inf(self):
        """Stratégie buy-and-hold (TO=0) → supporte infiniment de coûts."""
        be = breakeven_cost(gross_sharpe=1.0, annual_turnover=0)
        assert be == float("inf")

    def test_high_turnover_low_breakeven(self):
        """
        Turnover élevé → breakeven faible (moins de marge pour les coûts).
        Stratégie avec TO=500×/an vs TO=5×/an.
        """
        be_high_to = breakeven_cost(gross_sharpe=1.0, annual_turnover=500)
        be_low_to  = breakeven_cost(gross_sharpe=1.0, annual_turnover=5)
        assert be_high_to < be_low_to

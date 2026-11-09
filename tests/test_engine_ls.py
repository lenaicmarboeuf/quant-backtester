import numpy as np
import pandas as pd
import pytest

from backtester.borrow import borrow_cost, borrow_cost_panel
from backtester.engine import Backtester, PortfolioBacktester
from backtester.factors import (
    cross_sectional_ic, cross_sectional_icir, ic_summary, compute_ic,
)
from backtester.metrics import nb_trades, full_metrics
from backtester.portfolio import quantile_weights


class TestNbTrades:
    def test_signed_example_from_docstring(self):
        pos = pd.Series([0, 0, 1, 1, 0, -1, -1, 1, 1])
        assert nb_trades(pos) == 3

    def test_short_entry_counted(self):
        assert nb_trades(pd.Series([0, -1, -1, 0])) == 1

    def test_cover_not_counted_as_entry(self):
        # [0, -1, 0] : 1 entrée short ; le rachat (-1 → 0) ne doit PAS compter
        # (l'ancienne version `diff() > 0` le comptait comme une entrée)
        assert nb_trades(pd.Series([0, -1, 0])) == 1
        # une série qui démarre à -1 part implicitement de cash : 1 entrée
        assert nb_trades(pd.Series([-1, -1, 0, 0])) == 1

    def test_resize_same_direction_not_new_trade(self):
        assert nb_trades(pd.Series([0, 0.5, 0.8, 0.8, 0])) == 1

    def test_flip_counts_once(self):
        assert nb_trades(pd.Series([0, 1, -1, 0])) == 2

    def test_long_only_unchanged(self):
        assert nb_trades(pd.Series([0, 1, 1, 0, 1, 0])) == 2

    def test_full_metrics_uses_position(self):
        r = pd.Series([0.0, 0.01, -0.01, 0.02, 0.0])
        e = 100 * (1 + r).cumprod()
        m = full_metrics(r, e, pd.Series([0, 1, 1, 0, -1]))
        assert m["Nb trades"] == 2


class TestBorrow:
    def test_only_short_pays(self):
        p = pd.Series([1.0, 0.0, -0.5, -1.0])
        c = borrow_cost(p, borrow_bps=252, periods=252)
        assert c.iloc[0] == 0 and c.iloc[1] == 0
        assert np.isclose(c.iloc[2], -0.5 / 10_000)
        assert np.isclose(c.iloc[3], -1.0 / 10_000)

    def test_panel_scalar_and_dict(self):
        w = pd.DataFrame({"A": [-1.0, -1.0], "B": [1.0, 1.0]})
        assert np.isclose(borrow_cost_panel(w, 252).iloc[0], -1 / 10_000)
        assert np.isclose(borrow_cost_panel(w, {"A": 504, "B": 9999}).iloc[0], -2 / 10_000)


class TestBacktesterLongShort:
    def test_nb_trades_nonzero(self, single_asset):
        sig = np.sign(single_asset["returns"].rolling(5).mean()).fillna(0)
        out = Backtester(single_asset).run(sig, "long_short")
        assert out["metrics"]["Nb trades"] > 0

    def test_long_short_positions_values(self, single_asset):
        sig = single_asset["returns"].rolling(5).mean().fillna(0)
        pos = Backtester(single_asset).compute_positions(sig, "long_short")
        assert set(pos.unique()) <= {-1.0, 0.0, 1.0}
        assert pos.iloc[0] == 0  # shift(1)

    def test_short_profits_on_falling_market(self):
        idx = pd.bdate_range("2021-01-01", periods=60)
        close = pd.Series(100 * 0.99 ** np.arange(60), index=idx)
        df = pd.DataFrame({"Close": close, "returns": close.pct_change()}).dropna()
        sig = pd.Series(-1.0, index=df.index)
        out = Backtester(df).run(sig, "long_short")
        assert out["metrics"]["CAGR (%)"] > 0
        assert out["results"]["equity_curve"].iloc[-1] > 10_000

    def test_borrow_reduces_return_not_for_long(self, single_asset):
        sig_s = pd.Series(-1.0, index=single_asset.index)
        sig_l = pd.Series(1.0, index=single_asset.index)
        s0 = Backtester(single_asset).run(sig_s, "long_short")["results"]["equity_curve"].iloc[-1]
        s1 = Backtester(single_asset, borrow_bps=500).run(sig_s, "long_short")["results"]["equity_curve"].iloc[-1]
        l0 = Backtester(single_asset).run(sig_l, "long_short")["results"]["equity_curve"].iloc[-1]
        l1 = Backtester(single_asset, borrow_bps=500).run(sig_l, "long_short")["results"]["equity_curve"].iloc[-1]
        assert s1 < s0
        assert np.isclose(l0, l1)

    def test_signal_ls_continuous_and_vol_target(self, single_asset):
        sig = pd.Series(np.sin(np.arange(len(single_asset)) / 20), index=single_asset.index)
        bt = Backtester(single_asset)
        p = bt.compute_positions(sig, "signal_ls")
        assert p.abs().max() <= 1.0
        pv = bt.compute_positions(sig, "signal_ls", vol_target=0.05)
        assert pv.abs().max() <= 1.0 and (pv.abs() <= p.abs() + 1e-12).all()

    def test_costs_charge_flip_double(self):
        idx = pd.bdate_range("2021-01-01", periods=6)
        close = pd.Series(100.0, index=idx)
        df = pd.DataFrame({"Close": close, "returns": close.pct_change().fillna(0)})
        bt = Backtester(df, transaction_costs={"commission_bps": 10, "spread_bps": 0, "slippage_bps": 0})
        pos = pd.Series([0, 1, 1, -1, -1, 0], index=idx, dtype=float)
        res = bt.compute_pnl(pos)
        c = res["cost_returns"]
        assert abs(c.iloc[3]) > 1.5 * abs(c.iloc[1])  # flip = turnover 2 vs entrée = 1

    def test_borrow_sensitivity_monotone(self, single_asset):
        sig = pd.Series(-1.0, index=single_asset.index)
        tab = Backtester(single_asset).borrow_sensitivity(sig, [0, 100, 1000])
        assert tab["Capital final (€)"].is_monotonic_decreasing


class TestPortfolioBacktester:
    def test_matches_manual_gross(self, universe, close_df):
        rets, _ = universe
        w = quantile_weights(rets.rolling(20).mean(), 0.2, 0.2)
        out = PortfolioBacktester(rets).run(w, costs=False)
        manual = (w.shift(1).fillna(0) * rets).sum(axis=1)
        assert np.allclose(out["results"]["gross_returns"], manual)
        assert np.allclose(out["results"]["strategy_returns"], manual)

    def test_costs_reduce_return(self, universe):
        rets, _ = universe
        w = quantile_weights(rets.rolling(20).mean())
        net = PortfolioBacktester(rets).run(w)["results"]["equity_curve"].iloc[-1]
        gross = PortfolioBacktester(rets).run(w, costs=False)["results"]["equity_curve"].iloc[-1]
        assert net < gross

    def test_exposure_report(self, universe):
        rets, _ = universe
        w = quantile_weights(rets.rolling(20).mean())
        out = PortfolioBacktester(rets).run(w)
        assert abs(out["exposure"]["nette moyenne"]) < 0.05
        assert out["exposure"]["gross moyenne"] > 1.5
        assert out["exposure"]["coût emprunt annuel (%)"] > 0

    def test_borrow_cost_matches_short_leg(self, universe):
        rets, _ = universe
        w = quantile_weights(rets.rolling(20).mean())
        out = PortfolioBacktester(rets, borrow_bps=100).run(w)
        # jambe short = 100 % du capital → ~1 %/an (hors warm-up de 20 jours)
        assert 0.85 < out["exposure"]["coût emprunt annuel (%)"] < 1.05

    def test_capm_hook(self, universe, market):
        rets, _ = universe
        w = quantile_weights(rets.rolling(20).mean())
        out = PortfolioBacktester(rets).run(w, market_returns=market)
        assert "capm" in out

    def test_no_lookahead_future_weight_change(self, universe):
        rets, _ = universe
        w = quantile_weights(rets.rolling(20).mean())
        w2 = w.copy(); w2.iloc[-1] = 0.0   # on modifie seulement le dernier poids
        a = PortfolioBacktester(rets).run(w, costs=False)["results"]["strategy_returns"]
        b = PortfolioBacktester(rets).run(w2, costs=False)["results"]["strategy_returns"]
        assert np.allclose(a, b)  # le dernier poids ne peut affecter aucun rendement passé/actuel


class TestCrossSectionalIC:
    def test_perfect_factor(self, universe):
        rets, _ = universe
        fwd = rets.shift(-1)
        ic = cross_sectional_ic(fwd, fwd)  # facteur = oracle
        assert np.allclose(ic.dropna(), 1.0)

    def test_inverse_factor(self, universe):
        rets, _ = universe
        fwd = rets.shift(-1)
        assert np.allclose(cross_sectional_ic(-fwd, fwd).dropna(), -1.0)

    def test_random_factor_near_zero(self, universe):
        rets, _ = universe
        fwd = rets.shift(-1)
        rng = np.random.default_rng(4)
        f = pd.DataFrame(rng.normal(size=rets.shape), index=rets.index, columns=rets.columns)
        assert abs(cross_sectional_ic(f, fwd).mean()) < 0.03

    def test_matches_scipy_spearman(self, universe):
        from scipy.stats import spearmanr
        rets, _ = universe
        fwd = rets.shift(-1)
        f = rets.rolling(10).mean()
        ic = cross_sectional_ic(f, fwd)
        d = ic.dropna().index[100]
        expected = spearmanr(f.loc[d], fwd.loc[d])[0]
        assert np.isclose(ic.loc[d], expected)

    def test_min_assets(self):
        idx = pd.bdate_range("2021-01-01", periods=3)
        f = pd.DataFrame(np.random.rand(3, 3), index=idx)
        r = pd.DataFrame(np.random.rand(3, 3), index=idx)
        assert cross_sectional_ic(f, r, min_assets=5).isna().all()

    def test_bad_method(self, universe):
        rets, _ = universe
        with pytest.raises(ValueError):
            cross_sectional_ic(rets, rets, method="kendall")

    def test_icir_and_summary(self, universe):
        rets, _ = universe
        fwd = rets.shift(-1)
        ic = cross_sectional_ic(rets.rolling(10).mean(), fwd)
        assert np.isfinite(cross_sectional_icir(ic))
        assert isinstance(cross_sectional_icir(ic, window=60), pd.Series)
        s = ic_summary(ic)
        assert {"IC moyen", "ICIR", "t-stat", "IC > 0 (%)"} <= set(s)

    def test_compute_ic_returns_float(self, single_asset):
        sig = single_asset["returns"].rolling(5).mean()
        fwd = single_asset["returns"].shift(-1)
        assert isinstance(compute_ic(sig, fwd), float)

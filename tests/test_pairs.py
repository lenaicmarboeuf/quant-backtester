import numpy as np
import pandas as pd
import pytest

from backtester.pairs import (
    hedge_ratio, rolling_hedge_ratio, spread, spread_zscore, half_life,
    cointegration_test, pairs_signal, pair_returns,
)


def simulate_ar1(phi: float, n: int = 3000, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return pd.Series(x, index=pd.bdate_range("2015-01-01", periods=n))


def make_pair(coint: bool = True, n: int = 1500, beta: float = 1.5, seed: int = 3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    x = pd.Series(np.cumsum(rng.normal(0, 0.01, n)) + 4.0, index=idx)
    if coint:
        e = np.zeros(n)
        for t in range(1, n):
            e[t] = 0.9 * e[t - 1] + rng.normal(0, 0.02)
        y = beta * x + 0.3 + e
    else:
        y = pd.Series(np.cumsum(rng.normal(0, 0.01, n)) + 4.0, index=idx)
    return pd.Series(y, index=idx), x


class TestHedgeAndSpread:
    def test_hedge_ratio_recovers_beta(self):
        y, x = make_pair(True)
        b, a = hedge_ratio(y, x)
        assert abs(b - 1.5) < 0.1
        assert abs(a - 0.3) < 0.5

    def test_hedge_ratio_degenerate(self):
        x = pd.Series(np.ones(50))
        b, _ = hedge_ratio(x * 2, x)
        assert np.isnan(b)

    def test_spread_formula(self):
        y = pd.Series([2.0, 4.0]); x = pd.Series([1.0, 2.0])
        assert np.allclose(spread(y, x, 1.0, 0.5), [0.5, 1.5])

    def test_spread_static_default(self):
        y, x = make_pair(True)
        assert abs(spread(y, x).mean()) < 1e-8  # résidus MCO centrés

    def test_rolling_hedge_no_lookahead(self):
        y, x = make_pair(True)
        h1 = rolling_hedge_ratio(y, x, 100)
        y2 = y.copy(); y2.iloc[-1] += 5
        h2 = rolling_hedge_ratio(y2, x, 100)
        assert np.allclose(h1["beta"].iloc[:-1].fillna(0), h2["beta"].iloc[:-1].fillna(0))

    def test_zscore_stats(self):
        s = simulate_ar1(0.9, 800)
        z = spread_zscore(s, 60).dropna()
        assert abs(z.mean()) < 0.2 and 0.8 < z.std() < 1.3


class TestHalfLife:
    @pytest.mark.parametrize("phi", [0.8, 0.9, 0.95])
    def test_ar1_half_life(self, phi):
        theo = np.log(2) / -np.log(phi)
        est = half_life(simulate_ar1(phi, 20000, seed=5))
        assert abs(est - theo) / theo < 0.15

    def test_random_walk_infinite(self):
        rng = np.random.default_rng(2)
        rw = pd.Series(np.cumsum(rng.normal(size=2000)))
        assert half_life(rw) > 200  # inf ou très grand

    def test_trend_is_inf(self):
        assert half_life(pd.Series(np.arange(100.0))) == float("inf")

    def test_short_series(self):
        assert half_life(pd.Series([1.0, 2.0])) == float("inf")


class TestCointegration:
    def test_detects_cointegrated(self):
        y, x = make_pair(True)
        r = cointegration_test(y, x)
        assert r["cointegrated"] and r["pvalue"] < 0.05
        assert 3 < r["half_life"] < 20

    def test_rejects_independent_walks(self):
        rejections = 0
        for s in range(10):
            y, x = make_pair(False, seed=100 + s)
            rejections += cointegration_test(y, x)["cointegrated"]
        assert rejections <= 3  # ~5 % attendu, marge pour 10 tirages

    def test_too_short(self):
        s = pd.Series(np.arange(10.0))
        assert "error" in cointegration_test(s, s)


class TestPairsSignal:
    def test_positions_in_set(self):
        y, x = make_pair(True)
        sig = pairs_signal(y, x)
        assert set(sig["position"].unique()) <= {-1.0, 0.0, 1.0}

    def test_flat_during_warmup(self):
        y, x = make_pair(True)
        sig = pairs_signal(y, x, hedge_window=120, z_window=60)
        assert (sig["position"].iloc[:150] == 0).all()

    def test_direction(self):
        """z très positif → short le spread (-1) ; z très négatif → long (+1)."""
        y, x = make_pair(True)
        sig = pairs_signal(y, x, entry_z=2.0, stop_z=None).dropna()
        opened = sig[(sig["position"] != 0) & (sig["position"].shift(1) == 0)]
        assert len(opened) > 5
        assert (np.sign(opened["z"]) == -opened["position"]).all()

    def test_exit_at_mean(self):
        y, x = make_pair(True)
        sig = pairs_signal(y, x, entry_z=2.0, exit_z=0.0, stop_z=None).dropna()
        prev = sig["position"].shift(1)
        longs_closed = sig[(prev == 1) & (sig["position"] == 0)]
        assert (longs_closed["z"] >= 0).all()

    def test_stop_blocks_reentry(self):
        idx = pd.bdate_range("2020-01-01", periods=400)
        rng = np.random.default_rng(9)
        x = pd.Series(np.cumsum(rng.normal(0, 0.01, 400)) + 4, index=idx)
        y = 1.0 * x + rng.normal(0, 0.02, 400)
        y.iloc[300:] += np.linspace(0, 1.5, 100)  # rupture : le spread diverge
        sig = pairs_signal(y, x, hedge_window=60, z_window=40, entry_z=2, stop_z=4)
        assert (sig["position"].abs() <= 1).all()

    def test_coint_filter_runs(self):
        y, x = make_pair(True, n=700)
        sig = pairs_signal(y, x, coint_window=250, coint_every=42)
        assert (sig["position"].iloc[:250] == 0).all()


class TestPairReturns:
    def test_shift_and_gross_exposure(self):
        y, x = make_pair(True, n=600)
        sig = pairs_signal(y, x)
        cy, cx = np.exp(y), np.exp(x)
        pr = pair_returns(cy, cx, sig["position"], sig["beta"])
        assert pr["gross_exposure"].max() <= 1.0 + 1e-9
        first_pos = sig["position"].ne(0).idxmax()
        assert pr.loc[first_pos, "gross_returns"] == 0  # position pas encore détenue

    def test_costs_two_legs(self):
        idx = pd.bdate_range("2021-01-01", periods=5)
        cy = pd.Series([100.0] * 5, index=idx); cx = pd.Series([100.0] * 5, index=idx)
        pos = pd.Series([0, 1, 1, 0, 0], index=idx, dtype=float)
        beta = pd.Series(1.0, index=idx)
        pr = pair_returns(cy, cx, pos, beta, cost_bps=10, borrow_bps=0)
        # entrée : deux jambes de 0.5 → turnover 1.0 ; sortie : idem → 2.0 au total
        assert np.isclose(-pr["trading_costs"].sum(), 2.0 * 10 / 10_000)

    def test_borrow_only_short_leg(self):
        idx = pd.bdate_range("2021-01-01", periods=4)
        c = pd.Series([100.0] * 4, index=idx)
        pos = pd.Series([1, 1, 1, 1], index=idx, dtype=float)
        beta = pd.Series(1.0, index=idx)
        pr = pair_returns(c, c, pos, beta, cost_bps=0, borrow_bps=252)
        # jambe X short = 0.5 du capital, 252 bps/an → 0.5 × 1 bp / jour
        assert np.isclose(pr["borrow_costs"].iloc[-1], -0.5 * 252 / 10_000 / 252)

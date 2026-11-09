import numpy as np
import pandas as pd
import pytest

from backtester.portfolio import (
    quantile_weights, dollar_neutral, rolling_beta, portfolio_beta, beta_hedge,
    beta_neutral_weights, portfolio_returns, hedged_returns, exposure_summary, turnover,
)


@pytest.fixture
def scores(universe):
    rets, _ = universe
    return rets.rolling(20).mean()


class TestQuantileWeights:
    def test_dollar_neutral_and_gross(self, scores):
        w = quantile_weights(scores, 0.2, 0.2).dropna(how="all")
        valid = scores.notna().sum(axis=1) >= 2
        w = w[valid]
        assert np.allclose(w.sum(axis=1), 0.0)
        assert np.allclose(w.abs().sum(axis=1), 2.0)

    def test_counts(self, scores):
        w = quantile_weights(scores, 0.2, 0.2)
        last = w.iloc[-1]
        assert (last > 0).sum() == 4 and (last < 0).sum() == 4

    def test_long_leg_has_best_scores(self, scores):
        w = quantile_weights(scores, 0.2, 0.2)
        d = scores.index[-1]
        longs = scores.loc[d][w.loc[d] > 0]
        shorts = scores.loc[d][w.loc[d] < 0]
        assert longs.min() > shorts.max()

    def test_long_only(self, scores):
        w = quantile_weights(scores, 0.2, 0.2, long_only=True)
        assert (w >= 0).all().all()
        assert np.isclose(w.iloc[-1].sum(), 1.0)

    def test_rebalance_every_holds_weights(self, scores):
        w = quantile_weights(scores, 0.2, 0.2, rebalance_every=21)
        changes = (w.diff().abs().sum(axis=1) > 1e-12).sum()
        assert changes <= len(w) // 21 + 2

    def test_nan_scores_get_zero(self, scores):
        w = quantile_weights(scores, 0.2, 0.2)
        assert (w.where(scores.isna(), 0.0) == 0).all().all()

    def test_bad_rebalance(self, scores):
        with pytest.raises(ValueError):
            quantile_weights(scores, rebalance_every=0)


class TestDollarNeutral:
    def test_sum_zero_gross(self):
        w = pd.DataFrame([[0.5, 0.3, -0.2, 0.0]], columns=list("abcd"))
        out = dollar_neutral(w, gross=2.0)
        assert np.isclose(out.sum(axis=1).iloc[0], 0)
        assert np.isclose(out.abs().sum(axis=1).iloc[0], 2.0)
        assert np.allclose(out.iloc[0].values, [0.75, 0.25, -1.0, 0.0])

    def test_idempotent(self, scores):
        w = quantile_weights(scores)
        assert np.allclose(dollar_neutral(w).values, w.values)

    def test_empty_row(self):
        out = dollar_neutral(pd.DataFrame([[0.0, 0.0]]))
        assert (out == 0).all().all()


class TestBeta:
    def test_rolling_beta_recovers_true(self, universe, market):
        rets, betas = universe
        b = rolling_beta(rets, market, window=250)
        est = b.iloc[-1]
        assert np.abs(est - betas).max() < 0.25

    def test_no_lookahead(self, universe, market):
        rets, _ = universe
        b1 = rolling_beta(rets, market, 63)
        r2 = rets.copy()
        r2.iloc[-1] *= 10  # on change seulement le dernier jour
        b2 = rolling_beta(r2, market, 63)
        assert np.allclose(b1.iloc[:-1].fillna(0), b2.iloc[:-1].fillna(0))

    def test_series_input(self, universe, market):
        rets, _ = universe
        b = rolling_beta(rets["A19"], market, 120)
        assert isinstance(b, pd.Series)

    def test_portfolio_beta_example(self):
        w = pd.DataFrame({"L": [0.5], "S": [-0.5]})
        b = pd.DataFrame({"L": [1.4], "S": [0.7]})
        assert np.isclose(portfolio_beta(w, b).iloc[0], 0.35)
        assert np.isclose(beta_hedge(w, b).iloc[0], -0.35)

    def test_dollar_neutral_is_not_beta_neutral(self, universe, scores, market):
        """Net = 0 mais beta ≠ 0 si les jambes n'ont pas les mêmes betas."""
        rets, betas = universe
        w = pd.DataFrame([betas.index.map(lambda c: 0.5 if c in ("A18", "A19") else (-0.5 if c in ("A00", "A01") else 0)).values],
                         columns=betas.index)
        w.iloc[0] = w.iloc[0] * 1.0
        bdf = pd.DataFrame([betas.values], columns=betas.index)
        assert np.isclose(w.sum(axis=1).iloc[0], 0)
        assert portfolio_beta(w, bdf).iloc[0] > 0.5

    def test_beta_neutral_weights_zero_beta(self, universe):
        _, betas = universe
        w = pd.DataFrame([np.where(betas.index.isin(["A17", "A18", "A19"]), 1 / 3,
                                   np.where(betas.index.isin(["A00", "A01", "A02"]), -1 / 3, 0.0))],
                         columns=betas.index)
        bdf = pd.DataFrame([betas.values], columns=betas.index)
        out = beta_neutral_weights(w, bdf)
        assert abs(portfolio_beta(out, bdf).iloc[0]) < 1e-9
        assert np.isclose(out.abs().sum(axis=1).iloc[0], w.abs().sum(axis=1).iloc[0])

    def test_beta_neutral_fallback_on_bad_ratio(self):
        w = pd.DataFrame({"L": [1.0], "S": [-1.0]})
        b = pd.DataFrame({"L": [1.0], "S": [-1.0]})  # r < 0 → on garde
        assert np.allclose(beta_neutral_weights(w, b).values, w.values)

    def test_hedge_reduces_market_exposure(self, universe, scores, market):
        rets, _ = universe
        betas = rolling_beta(rets, market, 120)
        w = quantile_weights(scores, 0.2, 0.2)
        pr = portfolio_returns(w, rets)
        h = beta_hedge(w, betas)
        hr = hedged_returns(pr, h, market)
        after = np.corrcoef(hr.iloc[200:], market.iloc[200:])[0, 1]
        before = np.corrcoef(pr.iloc[200:], market.iloc[200:])[0, 1]
        assert abs(after) <= abs(before) + 0.02


class TestReturns:
    def test_shift_applied_once(self, universe):
        rets, _ = universe
        w = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
        w.iloc[10, 0] = 1.0  # poids décidé à la clôture du jour 10
        pr = portfolio_returns(w, rets)
        assert pr.iloc[10] == 0.0                                  # pas de gain le jour 10
        assert np.isclose(pr.iloc[11], rets.iloc[11, 0])           # appliqué le jour 11

    def test_short_gains_when_market_falls(self):
        idx = pd.bdate_range("2021-01-01", periods=4)
        r = pd.DataFrame({"X": [0.0, -0.1, -0.1, 0.0]}, index=idx)
        w = pd.DataFrame({"X": [-1.0] * 4}, index=idx)
        pr = portfolio_returns(w, r)
        assert np.isclose(pr.iloc[1], 0.1)

    def test_no_shift_flag(self, universe):
        rets, _ = universe
        w = pd.DataFrame(1.0, index=rets.index, columns=rets.columns)
        assert np.allclose(portfolio_returns(w, rets, shift=False), rets.sum(axis=1))


class TestDiagnostics:
    def test_exposure(self, scores):
        w = quantile_weights(scores)
        e = exposure_summary(w).iloc[-1]
        assert np.isclose(e["net"], 0) and np.isclose(e["gross"], 2)
        assert e["n_long"] == 4 and e["n_short"] == 4

    def test_turnover_full_rotation(self):
        idx = pd.bdate_range("2021-01-01", periods=4)
        w = pd.DataFrame({"A": [1, 1, -1, -1], "B": [-1, -1, 1, 1]}, index=idx, dtype=float)
        t = turnover(w, shift=False)
        assert np.isclose(t.iloc[2], 4.0)

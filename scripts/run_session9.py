#!/usr/bin/env python3
"""
scripts/run_session9.py — Séance 9 : long/short, neutralité de marché, pair trading

Usage :
    python scripts/run_session9.py                 # données réelles (yfinance)
    python scripts/run_session9.py --synthetic     # données simulées (hors-ligne)
    python scripts/run_session9.py --start 2015-01-01 --end 2025-12-31

Ce que fait le script :
    1. Univers de ~20 actions US + SPY (yfinance) ; avertissement biais de survie
    2. Facteurs (momentum, low-vol) → IC cross-sectionnel + ICIR
    3. Portefeuille long/short 20 %/20 % (rebalancement mensuel) vs long-only
    4. Beta glissant, couverture beta, alpha CAPM
    5. Sensibilité au coût d'emprunt
    6. Pair trading (KO/PEP, XOM/CVX, V/MA) : test de cointégration, backtest net de coûts
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtester.engine import PortfolioBacktester
from backtester.factors import (
    momentum_factor, volatility_factor, cross_sectional_ic, cross_sectional_icir,
    ic_summary, fama_french_alpha,
)
from backtester.metrics import full_metrics
from backtester.pairs import cointegration_test, pairs_signal, pair_returns
from backtester.portfolio import (
    quantile_weights, rolling_beta, beta_hedge, hedged_returns, portfolio_returns,
    exposure_summary, portfolio_beta,
)

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "BAC", "XOM", "CVX",
    "KO", "PEP", "WMT", "PG", "JNJ", "PFE", "V", "MA", "HD", "DIS",
]
PAIRS = [("KO", "PEP"), ("XOM", "CVX"), ("V", "MA")]


def load_universe(start: str, end: str, synthetic: bool):
    if synthetic:
        from backtester.synthetic import make_universe
        close, spy = make_universe()
        return close, spy, True
    import yfinance as yf
    raw = yf.download(TICKERS + ["SPY"], start=start, end=end, auto_adjust=True, progress=False)["Close"]
    raw = raw.dropna(how="all")
    if raw.empty:
        raise RuntimeError("Téléchargement vide (réseau ?). Relance avec --synthetic pour tester hors-ligne.")
    return raw[TICKERS].dropna(axis=1, thresh=int(len(raw) * 0.9)), raw["SPY"], False


def line(title: str):
    print(f"\n{'═' * 70}\n{title}\n{'═' * 70}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--borrow-bps", type=float, default=30.0)
    args = ap.parse_args()

    close, spy, synth = load_universe(args.start, args.end, args.synthetic)
    rets = close.pct_change()
    spy_r = spy.pct_change()
    fwd = rets.shift(-1)          # ← seule utilisation de shift(-1) : MESURER l'IC

    line("1. UNIVERS")
    print(f"{close.shape[1]} actifs, {len(close)} jours ({close.index[0].date()} → {close.index[-1].date()})")
    if synth:
        print("⚠ DONNÉES SIMULÉES : résultats sans valeur économique.")
    else:
        print("⚠ BIAIS DE SURVIE : univers = grandes capitalisations ACTUELLES. Les perdants "
              "exclus de l'indice (faillites, sorties) sont absents → les jambes longues sont "
              "avantagées ; l'effet sur les shorts est ambigu mais le Sharpe est probablement surestimé.")

    line("2. IC CROSS-SECTIONNEL (le vrai IC académique)")
    factors = {
        "momentum 12-1": momentum_factor(close),
        "low-vol":       volatility_factor(close),
    }
    ic_rows = {}
    for name, f in factors.items():
        ic = cross_sectional_ic(f, fwd)
        ic_rows[name] = ic_summary(ic)
    print(pd.DataFrame(ic_rows).to_string())

    line("3. PORTEFEUILLE LONG/SHORT vs LONG-ONLY (momentum, 20 %/20 %, rebal. mensuel)")
    score = factors["momentum 12-1"]
    w_ls = quantile_weights(score, 0.2, 0.2, rebalance_every=21)
    w_lo = quantile_weights(score, 0.2, 0.2, long_only=True, rebalance_every=21)
    bt = PortfolioBacktester(rets, borrow_bps=args.borrow_bps)
    out_ls = bt.run(w_ls, market_returns=spy_r)
    out_lo = bt.run(w_lo, market_returns=spy_r)
    gross_ls = bt.run(w_ls, costs=False)
    tab = pd.DataFrame({
        "L/S net":   out_ls["metrics"],
        "L/S brut":  gross_ls["metrics"],
        "Long-only": out_lo["metrics"],
    })
    print(tab.to_string())
    print("\nExpositions et coûts (L/S) :", out_ls["exposure"])
    print("CAPM L/S      :", out_ls["capm"])
    print("CAPM Long-only:", out_lo["capm"])

    line("4. BETA GLISSANT ET COUVERTURE")
    betas = rolling_beta(rets, spy_r, window=63)
    pb = portfolio_beta(w_ls.shift(1), betas.shift(1))
    print(f"Beta moyen du portefeuille L/S (dollar-neutral) : {pb.mean():+.3f}  (min {pb.min():+.2f}, max {pb.max():+.2f})")
    pr = portfolio_returns(w_ls, rets)
    h = beta_hedge(w_ls, betas)
    hr = hedged_returns(pr, h, spy_r)
    print("Beta vs SPY avant couverture :", fama_french_alpha(pr, spy_r).get("beta"))
    print("Beta vs SPY après couverture :", fama_french_alpha(hr, spy_r).get("beta"))
    print("(Couverture sans coûts : le short de SPY paierait aussi frais et slippage.)")

    line("5. SENSIBILITÉ AU COÛT D'EMPRUNT (L/S net)")
    rows = {}
    for b in [0, 30, 100, 300, 1000]:
        m = PortfolioBacktester(rets, borrow_bps=b).run(w_ls)["metrics"]
        rows[f"{b} bps"] = {k: m[k] for k in ["CAGR (%)", "Sharpe", "Max Drawdown (%)"]}
    print(pd.DataFrame(rows).T.to_string())

    line("6. PAIR TRADING")
    if synth:
        from backtester.synthetic import make_pair
        y, x = make_pair()
        pair_data = {("Y", "X"): (y, x)}
    else:
        pair_data = {p: (close[p[0]], close[p[1]]) for p in PAIRS if p[0] in close and p[1] in close}
    n_tests = len(pair_data)
    for (a, b), (cy, cx) in pair_data.items():
        ly, lx = np.log(cy), np.log(cx)
        ct = cointegration_test(ly, lx)
        print(f"\n{a}/{b} — corr rendements = {cy.pct_change().corr(cx.pct_change()):.2f}")
        print("  Engle-Granger :", ct)
        if n_tests > 1:
            print(f"  (⚠ {n_tests} paires testées : seuil Bonferroni = {0.05 / n_tests:.3f})")
        sig = pairs_signal(ly, lx, hedge_window=120, z_window=60)
        pr_ = pair_returns(cy, cx, sig["position"], sig["beta"], borrow_bps=args.borrow_bps)
        eq = 10_000 * (1 + pr_["strategy_returns"]).cumprod()
        m = full_metrics(pr_["strategy_returns"], eq, sig["position"])
        print("  Backtest net :", {k: m[k] for k in ["CAGR (%)", "Sharpe", "Max Drawdown (%)", "Nb trades"]})
        if not ct.get("cointegrated", False):
            print("  → Paire NON cointégrée sur l'échantillon : ce backtest n'est pas fiable.")

    print("\nTerminé.")


if __name__ == "__main__":
    main()

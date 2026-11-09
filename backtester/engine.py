"""
backtester/engine.py — Moteur de backtest vectorisé (v9 : long/short + multi-actifs)

Modifications séance 9 :
    - Positions SIGNÉES : modes 'long_short' et 'signal_ls' (short autorisé)
    - Coût d'emprunt (borrow_bps) sur la partie short, payé chaque jour
    - 'position' est maintenant passée à full_metrics → "Nb trades" n'est plus 0
    - Nouvelle classe PortfolioBacktester : poids multi-actifs (long/short par quantiles)

Rappel des conventions :
    - Les positions renvoyées par compute_positions() sont DÉJÀ décalées (shift(1)) :
      position[t] = ce qu'on détient PENDANT la journée t, décidé à la clôture de t-1.
    - PortfolioBacktester.run(weights) reçoit des poids "bruts" (décidés à la clôture de T)
      et applique le shift(1) lui-même, une seule fois.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtester.metrics import full_metrics
from backtester.sizing import (
    vol_target_size, kelly_size, fixed_fraction, signal_proportional_size,
)
from backtester.costs import total_transaction_cost, turnover_analysis
from backtester.borrow import borrow_cost, borrow_cost_panel


class Backtester:
    """
    Moteur de backtest vectorisé SUR UN SEUL ACTIF.

    Paramètres du constructeur
    --------------------------
    df                : DataFrame OHLCV avec colonne 'returns'
    initial_capital   : capital de départ en euros (défaut : 10 000)
    transaction_costs : dict de coûts (commission_bps, spread_bps, slippage_bps...)
                        None = sans coûts ; {} = coûts par défaut
    borrow_bps        : frais d'emprunt ANNUALISÉS (bps) sur la partie short.
                        0 = ignoré. Typique large-cap : 30 bps. Seule la position
                        négative paie ; sans effet en long only.
    """

    _DEFAULT_COSTS = {
        "commission_bps":        5.0,
        "spread_bps":            2.5,
        "slippage_bps":          2.0,
        "include_market_impact": False,
        "adv_ratio":             0.01,
    }

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10_000.0,
        transaction_costs: Optional[dict] = None,
        borrow_bps: float = 0.0,
    ):
        self.df              = df.copy()
        self.initial_capital = initial_capital
        self.borrow_bps      = borrow_bps
        self.results         = None

        if transaction_costs is None:
            self._costs_cfg = None
        else:
            self._costs_cfg = {**self._DEFAULT_COSTS, **transaction_costs}

    # ─────────────────────────────────────────────────────────────────────
    # Sizing
    # ─────────────────────────────────────────────────────────────────────

    def compute_positions(
        self,
        signal: pd.Series,
        sizing_mode: str = "binary",
        **sizing_kwargs,
    ) -> pd.Series:
        """
        Convertit un signal [-1, +1] en position (déjà décalée d'un jour).

        sizing_mode (long only, position dans [0, 1]) :
          'binary' | 'vol_target' | 'kelly' | 'fixed' | 'proportional'

        sizing_mode (long/short, position dans [-1, +1]) — NOUVEAU séance 9 :
          'long_short' : position = sign(signal) → +1 / 0 / -1
                         (signal > 0 = long, signal < 0 = short, signal = 0 = cash)
          'signal_ls'  : position = signal lui-même (continu, [-1, +1]).
                         Option vol_target=0.10 (kwarg) : divise par la vol réalisée
                         pour viser une vol annuelle cible (levier plafonné à 1).
        """
        if sizing_mode == "vol_target":
            return vol_target_size(
                returns=self.df["returns"], signal=signal, **sizing_kwargs
            )
        elif sizing_mode == "kelly":
            return kelly_size(
                returns=self.df["returns"], signal=signal, **sizing_kwargs
            )
        elif sizing_mode == "fixed":
            return fixed_fraction(signal=signal, **sizing_kwargs)
        elif sizing_mode == "proportional":
            return signal_proportional_size(
                signal=signal, returns=self.df["returns"], **sizing_kwargs
            )
        elif sizing_mode == "long_short":
            return np.sign(signal.fillna(0)).shift(1).fillna(0)
        elif sizing_mode == "signal_ls":
            vol_target = sizing_kwargs.get("vol_target", None)
            window     = sizing_kwargs.get("window", 20)
            pos = signal.fillna(0).clip(-1, 1)
            if vol_target is not None:
                vol = self.df["returns"].rolling(window).std() * np.sqrt(252)
                scale = (vol_target / vol).clip(0, 1).fillna(0)
                pos = pos * scale
            return pos.shift(1).fillna(0)
        else:  # 'binary'
            position = (signal > 0).astype(int)
            return position.shift(1).fillna(0)

    # ─────────────────────────────────────────────────────────────────────
    # PnL avec coûts de transaction et coût d'emprunt
    # ─────────────────────────────────────────────────────────────────────

    def compute_pnl(self, position: pd.Series) -> pd.DataFrame:
        """
        Calcule le PnL de la stratégie.

        Colonnes ajoutées :
            gross_returns    = position × return_marché    (short : position < 0
                               → on GAGNE quand le marché baisse)
            cost_returns     = coûts de trading (commission+spread+slippage), négatifs
            borrow_returns   = coût d'emprunt des shorts, négatif       [NOUVEAU]
            strategy_returns = gross + cost + borrow  (rendements NETS)
        """
        results = self.df.copy()
        results["position"] = position

        results["gross_returns"] = results["position"] * results["returns"]

        if self._costs_cfg is not None:
            cost_series = total_transaction_cost(
                prices    = results["Close"],
                positions = position,
                **self._costs_cfg,
            )
            results["cost_returns"] = cost_series.reindex(results.index).fillna(0)
        else:
            results["cost_returns"] = 0.0

        if self.borrow_bps > 0:
            results["borrow_returns"] = borrow_cost(position, self.borrow_bps)
        else:
            results["borrow_returns"] = 0.0

        results["strategy_returns"] = (
            results["gross_returns"] + results["cost_returns"] + results["borrow_returns"]
        )

        results["cumulative_returns"] = (
            (1 + results["strategy_returns"]).cumprod() - 1
        )
        results["equity_curve"] = (
            self.initial_capital * (1 + results["cumulative_returns"])
        )
        results["equity_gross"] = (
            self.initial_capital * (1 + results["gross_returns"]).cumprod()
        )

        return results

    # ─────────────────────────────────────────────────────────────────────
    # Métriques (position transmise → 'Nb trades' correct)
    # ─────────────────────────────────────────────────────────────────────

    def compute_metrics(self, results: pd.DataFrame) -> dict:
        """Métriques sur les rendements NETS."""
        return full_metrics(
            returns  = results["strategy_returns"],
            equity   = results["equity_curve"],
            position = results["position"],
        )

    def compute_gross_metrics(self, results: pd.DataFrame) -> dict:
        """Métriques sur les rendements BRUTS (pour mesurer le drag des coûts)."""
        return full_metrics(
            returns  = results["gross_returns"],
            equity   = results["equity_gross"],
            position = results["position"],
        )

    # ─────────────────────────────────────────────────────────────────────
    # Run complet
    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        signal: pd.Series,
        sizing_mode: str = "binary",
        **sizing_kwargs,
    ) -> dict:
        """
        Lance le backtest complet.

        Retourne un dict avec 'results', 'metrics' et, si les coûts sont
        activés, 'gross_metrics' et 'cost_analysis'.
        """
        position = self.compute_positions(signal, sizing_mode, **sizing_kwargs)
        results  = self.compute_pnl(position)
        metrics  = self.compute_metrics(results)

        output = {"results": results, "metrics": metrics}

        if self._costs_cfg is not None:
            output["gross_metrics"] = self.compute_gross_metrics(results)
            output["cost_analysis"] = turnover_analysis(
                positions = position,
                returns   = results["gross_returns"],
                cost_per_turnover_bps = (
                    self._costs_cfg["commission_bps"]
                    + self._costs_cfg["spread_bps"]
                    + self._costs_cfg["slippage_bps"]
                ) * 2,
            )

        self.results = results
        return output

    # ─────────────────────────────────────────────────────────────────────
    # Analyse de sensibilité aux coûts
    # ─────────────────────────────────────────────────────────────────────

    def cost_sensitivity(
        self,
        signal: pd.Series,
        cost_range_bps: list = None,
        sizing_mode: str = "binary",
        **sizing_kwargs,
    ) -> pd.DataFrame:
        """Teste la stratégie à différents niveaux de coûts A/R (inchangé depuis s7)."""
        if cost_range_bps is None:
            cost_range_bps = [0, 5, 10, 20, 30, 50]

        records = []
        for ar_bps in cost_range_bps:
            one_way = ar_bps / 2
            costs = {
                "commission_bps": one_way * 0.5,
                "spread_bps":     one_way * 0.25,
                "slippage_bps":   one_way * 0.25,
            }
            bt_temp = Backtester(
                self.df,
                initial_capital=self.initial_capital,
                transaction_costs=costs,
                borrow_bps=self.borrow_bps,
            )
            out = bt_temp.run(signal, sizing_mode=sizing_mode, **sizing_kwargs)
            row = {"cost_bps_AR": ar_bps}
            row.update(out["metrics"])
            records.append(row)

        return pd.DataFrame(records).set_index("cost_bps_AR")

    def borrow_sensitivity(
        self,
        signal: pd.Series,
        borrow_range_bps: list = None,
        sizing_mode: str = "long_short",
        **sizing_kwargs,
    ) -> pd.DataFrame:
        """
        Teste la stratégie à différents coûts d'emprunt (bps/an).

        Répond à : "Mon short survit-il si l'emprunt coûte 5 % par an ?"
        """
        if borrow_range_bps is None:
            borrow_range_bps = [0, 30, 100, 300, 1000]

        records = []
        for b in borrow_range_bps:
            bt_temp = Backtester(
                self.df,
                initial_capital=self.initial_capital,
                transaction_costs=None if self._costs_cfg is None else self._costs_cfg,
                borrow_bps=b,
            )
            out = bt_temp.run(signal, sizing_mode=sizing_mode, **sizing_kwargs)
            row = {"borrow_bps": b}
            row.update(out["metrics"])
            records.append(row)

        return pd.DataFrame(records).set_index("borrow_bps")


# ═════════════════════════════════════════════════════════════════════════════
# Backtest multi-actifs (portefeuilles de poids)
# ═════════════════════════════════════════════════════════════════════════════

class PortfolioBacktester:
    """
    Moteur de backtest vectorisé pour un portefeuille de N actifs.

    Entrée : un DataFrame de POIDS (dates × actifs), décidés à la clôture de T.
             Ex : sortie de portfolio.quantile_weights().

    Mécanique (par jour t) :
        w_held[t]   = weights[t-1]                       ← shift(1), appliqué ici
        gross[t]    = Σ_i w_held[i,t] × r[i,t]
        turnover[t] = Σ_i |w_held[i,t] - w_held[i,t-1]|
        trading[t]  = -turnover[t] × cost_bps / 10 000
        borrow[t]   = -Σ_i max(-w_held[i,t], 0) × borrow_bps / 10 000 / 252
        net[t]      = gross + trading + borrow

    Simplifications assumées :
        - Les poids sont ramenés à leur cible CHAQUE jour (pas de dérive entre
          rebalancements). Pour un rebalancement mensuel, l'erreur est faible
          mais réelle.
        - Coût de trading par unité de turnover = commission + spread/2 + slippage
          (bps, un sens). Pas de terme de volatilité ni d'impact marché.

    Paramètres
    ----------
    returns         : DataFrame (dates × actifs) des rendements journaliers
    initial_capital : capital de départ
    commission_bps, spread_bps, slippage_bps : coûts de trading (mêmes défauts que s7)
    borrow_bps      : float ou dict {ticker: bps/an}
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        initial_capital: float = 10_000.0,
        commission_bps: float = 5.0,
        spread_bps: float = 2.5,
        slippage_bps: float = 2.0,
        borrow_bps=30.0,
    ):
        self.returns         = returns.copy()
        self.initial_capital = initial_capital
        self.commission_bps  = commission_bps
        self.spread_bps      = spread_bps
        self.slippage_bps    = slippage_bps
        self.borrow_bps      = borrow_bps
        self.results         = None

    @property
    def one_way_cost_bps(self) -> float:
        """Coût par unité de turnover (un sens) en bps."""
        return self.commission_bps + self.spread_bps / 2 + self.slippage_bps

    def run(
        self,
        weights: pd.DataFrame,
        costs: bool = True,
        market_returns: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Lance le backtest portefeuille.

        Paramètres
        ----------
        weights        : poids décidés à la clôture de T (NON décalés)
        costs          : False = rendements bruts uniquement
        market_returns : (optionnel) rendements du marché ; si fourni, ajoute
                         beta et alpha CAPM du portefeuille dans la sortie

        Retourne
        --------
        dict : 'results' (DataFrame journalier), 'metrics', 'gross_metrics',
               'exposure' (brute/nette moyennes), 'capm' (si market_returns)
        """
        w = weights.reindex(self.returns.index).fillna(0.0)
        w = w.reindex(columns=self.returns.columns).fillna(0.0)
        w_held = w.shift(1).fillna(0.0)                # ← LA règle d'or

        r = self.returns.fillna(0.0)
        gross = (w_held * r).sum(axis=1)

        turnover = w_held.diff().abs().sum(axis=1).fillna(0.0)
        if costs:
            trading = -(turnover * self.one_way_cost_bps / 10_000)
            borrow  = borrow_cost_panel(w_held, self.borrow_bps)
        else:
            trading = pd.Series(0.0, index=w.index)
            borrow  = pd.Series(0.0, index=w.index)

        net = gross + trading + borrow

        res = pd.DataFrame({
            "gross_returns":    gross,
            "trading_costs":    trading,
            "borrow_costs":     borrow,
            "strategy_returns": net,
            "turnover":         turnover,
            "gross_exposure":   w_held.abs().sum(axis=1),
            "net_exposure":     w_held.sum(axis=1),
        })
        res["equity_curve"] = self.initial_capital * (1 + net).cumprod()
        res["equity_gross"] = self.initial_capital * (1 + gross).cumprod()

        n_rebal = int((turnover > 1e-9).sum())

        def _metrics(ret, eq):
            m = full_metrics(ret, eq)
            m.pop("Nb trades", None)
            m["Nb rebalancements"] = n_rebal
            return m

        out = {
            "results": res,
            "metrics": _metrics(res["strategy_returns"], res["equity_curve"]),
            "gross_metrics": _metrics(res["gross_returns"], res["equity_gross"]),
            "exposure": {
                "gross moyenne":  round(float(res["gross_exposure"].mean()), 3),
                "nette moyenne":  round(float(res["net_exposure"].mean()), 3),
                "turnover annuel (×)": round(float(turnover.mean() * 252), 2),
                "coût trading annuel (%)": round(float(-trading.sum() / len(trading) * 252 * 100), 3),
                "coût emprunt annuel (%)": round(float(-borrow.sum() / len(borrow) * 252 * 100), 3),
            },
        }

        if market_returns is not None:
            from backtester.factors import fama_french_alpha
            out["capm"] = fama_french_alpha(res["strategy_returns"], market_returns)

        self.results = res
        return out

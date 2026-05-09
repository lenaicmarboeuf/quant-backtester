import pandas as pd
import numpy as np

from backtester.metrics import full_metrics
from backtester.sizing import (
    vol_target_size,
    kelly_size,
    fixed_fraction,
    signal_proportional_size
)


class Backtester:
    """Moteur de backtest vectorisé. Accepte un signal externe."""

    def __init__(self, df: pd.DataFrame, initial_capital: float = 10000.0):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.results = None

    def compute_positions(self,
                          signal: pd.Series,
                          sizing_mode: str = 'binary',
                          **sizing_kwargs) -> pd.Series:
        """
        Convertit signal [-1,+1] en position avec sizing.

        sizing_mode options :
          'binary'       : 0 ou 1 (comportement séances 1-4)
          'vol_target'   : ciblage de volatilité
          'kelly'        : demi-Kelly
          'fixed'        : fraction fixe
          'proportional' : proportionnel au signal
        """
        if sizing_mode == 'vol_target':
            return vol_target_size(
                returns=self.df['returns'],
                signal=signal,
                **sizing_kwargs
            )
        elif sizing_mode == 'kelly':
            return kelly_size(
                returns=self.df['returns'],
                signal=signal,
                **sizing_kwargs
            )
        elif sizing_mode == 'fixed':
            return fixed_fraction(signal=signal, **sizing_kwargs)
        elif sizing_mode == 'proportional':
            return signal_proportional_size(
                signal=signal,
                returns=self.df['returns'],
                **sizing_kwargs
            )
        else:  # 'binary' — comportement original
            position = (signal > 0).astype(int)
            return position.shift(1).fillna(0)

    def compute_pnl(self, position: pd.Series) -> pd.DataFrame:
        results = self.df.copy()
        results['position'] = position
        results['strategy_returns'] = results['position'] * results['returns']
        results['cumulative_returns'] = (
            (1 + results['strategy_returns']).cumprod() - 1
        )
        results['equity_curve'] = (
            self.initial_capital * (1 + results['cumulative_returns'])
        )
        return results

    def compute_metrics(self, results: pd.DataFrame) -> dict:
        return full_metrics(
            returns=results['strategy_returns'],
            equity=results['equity_curve'],
            position=results['position'] 
        )

    def run(self,
            signal: pd.Series,
            sizing_mode: str = 'binary',
            **sizing_kwargs) -> dict:
        """Lance le backtest avec un signal externe et un mode de sizing."""
        position = self.compute_positions(signal, sizing_mode, **sizing_kwargs)
        results  = self.compute_pnl(position)
        metrics  = self.compute_metrics(results)
        self.results = results
        return {'results': results, 'metrics': metrics}

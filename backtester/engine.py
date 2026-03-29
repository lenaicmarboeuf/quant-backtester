import pandas as pd
import numpy as np
from data.loader import download_data, compute_returns


class Backtester:
    """
    Moteur de backtest vectorisé.
    Prend des données OHLCV et une stratégie, retourne les résultats.
    """

    def __init__(self, df: pd.DataFrame, initial_capital: float = 10000.0):
        """
        Initialise le backtester.

        Args:
            df              : DataFrame OHLCV avec colonne 'returns' déjà calculée
            initial_capital : capital de départ en euros/dollars
        """
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.results = None  # sera rempli après run()

    def compute_signal(self, window: int = 20) -> pd.Series:
        """
        Génère un signal SMA simple :
        +1 si le prix est AU-DESSUS de la moyenne mobile → tendance haussière
         0 si le prix est EN-DESSOUS                    → on reste en cash

        rolling(window).mean() : moyenne glissante sur 'window' jours
        La comparaison > retourne True/False, qu'on convertit en 1/0 avec astype(int)

        Args:
            window : nombre de jours pour la moyenne mobile (défaut 20)

        Returns:
            Series avec valeurs 0 ou 1
        """
        sma = self.df["Close"].rolling(window).mean()
        signal = (self.df["Close"] > sma).astype(int)
        return signal

    def compute_positions(self, signal: pd.Series) -> pd.Series:
        """
        Décale le signal d'un jour pour éviter le look-ahead bias.

        shift(1) : chaque valeur descend d'une ligne
            → la décision du jour J utilise le signal du jour J-1
            → on ne "voit" jamais le futur

        fillna(0) : le premier jour (NaN après shift) devient 0 = pas de position

        Args:
            signal : Series de signaux 0/1

        Returns:
            Series de positions 0/1 décalées d'un jour
        """
        position = signal.shift(1).fillna(0)
        return position

    def compute_pnl(self, position: pd.Series) -> pd.DataFrame:
        """
        Calcule le PnL (Profit and Loss) jour par jour.

        strategy_returns : rendement du portefeuille chaque jour
            = position × rendement du marché ce jour-là
            Si position = 1 et marché fait +2% → on gagne +2%
            Si position = 0 et marché fait +2% → on gagne 0% (on est en cash)
            Si position = 1 et marché fait -3% → on perd -3%

        cumsum() vs cumprod() — on utilise cumprod() :
            cumsum() additionne les rendements → approximation incorrecte
            cumprod() multiplie (1 + r) → rendement composé réel
            Exemple : +10% puis -10% avec cumprod = 0.99 (perte de 1%)
                      +10% puis -10% avec cumsum  = 0.00 (faux, dit 0%)

        equity_curve : valeur du portefeuille en euros
            = capital initial × rendement cumulé

        Args:
            position : Series de positions 0/1

        Returns:
            DataFrame avec colonnes strategy_returns, cumulative_returns, equity_curve
        """
        results = self.df.copy()
        results["position"] = position
        results["strategy_returns"] = results["position"] * results["returns"]

        # Rendement cumulé : (1 + r1) × (1 + r2) × ... - 1
        results["cumulative_returns"] = (
            (1 + results["strategy_returns"]).cumprod() - 1
        )

        # Valeur du portefeuille en euros
        results["equity_curve"] = (
            self.initial_capital * (1 + results["cumulative_returns"])
        )

        return results

    def compute_metrics(self, results: pd.DataFrame) -> dict:
        """
        Calcule les métriques clés de performance.

        CAGR (Compound Annual Growth Rate) :
            Taux de croissance annuel composé.
            Formule : (valeur_finale / valeur_initiale) ^ (1/nb_années) - 1
            252 = nombre de jours de bourse dans une année

        Sharpe ratio :
            Rendement moyen / risque (écart-type des rendements)
            Annualisé en multipliant par √252
            > 1.0 = acceptable, > 1.5 = bon, > 2.0 = excellent

        Max drawdown :
            Pire perte depuis un sommet.
            cummax() : valeur maximale atteinte jusqu'à ce jour
            drawdown = (valeur_actuelle - sommet) / sommet
            Le minimum de cette série = le pire drawdown

        Args:
            results : DataFrame retourné par compute_pnl()

        Returns:
            dict avec les métriques principales
        """
        returns = results["strategy_returns"].dropna()
        equity = results["equity_curve"]

        # Nombre d'années
        n_years = len(returns) / 252

        # CAGR
        total_return = equity.iloc[-1] / self.initial_capital
        cagr = total_return ** (1 / n_years) - 1

        # Sharpe (sans taux sans risque pour simplifier)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

        # Max drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        return {
            "CAGR": round(cagr * 100, 2),
            "Sharpe": round(sharpe, 2),
            "Max Drawdown": round(max_drawdown * 100, 2),
            "Capital final": round(equity.iloc[-1], 2),
            "Nb jours tradés": int(results["position"].sum()),
        }

    def run(self, window: int = 20) -> dict:
        """
        Lance le backtest complet en enchaînant toutes les étapes.

        C'est la seule méthode que tu appelleras de l'extérieur.
        Elle orchestre le pipeline complet :
        signal → position → pnl → métriques

        Args:
            window : paramètre de la stratégie SMA

        Returns:
            dict avec 'results' (DataFrame) et 'metrics' (dict)
        """
        signal = self.compute_signal(window)
        position = self.compute_positions(signal)
        results = self.compute_pnl(position)
        metrics = self.compute_metrics(results)

        self.results = results  # stocké dans l'objet pour y accéder après

        return {"results": results, "metrics": metrics}
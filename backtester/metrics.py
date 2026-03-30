import pandas as pd
import numpy as np


def sharpe_ratio(returns: pd.Series, periods: int = 252) -> float:
    """
    Sharpe ratio annualisé.

    returns.mean()  : rendement moyen par période
    returns.std()   : écart-type = volatilité = risque
    Le ratio mesure combien de rendement tu obtiens par unité de risque.

    periods = 252 pour données journalières (jours de bourse/an)
    periods = 52  pour données hebdomadaires
    periods = 12  pour données mensuelles

    np.sqrt(periods) : annualise le ratio.
    La volatilité s'annualise par la racine carrée du temps
    (propriété statistique du mouvement brownien).
    """
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods)


def max_drawdown(equity: pd.Series) -> float:
    equity_clean = equity.dropna()
    rolling_max = equity_clean.cummax()
    drawdown = (equity_clean - rolling_max) / rolling_max
    return float(drawdown.min())


def cagr(equity: pd.Series, periods: int = 252) -> float:
    equity_clean = equity.dropna()
    if len(equity_clean) == 0 or equity_clean.iloc[0] == 0:
        return 0.0
    n_years = len(equity_clean) / periods
    return float((equity_clean.iloc[-1] / equity_clean.iloc[0]) ** (1 / n_years) - 1)


def calmar_ratio(equity: pd.Series, periods: int = 252) -> float:
    """
    Ratio de Calmar = CAGR / |Max Drawdown|

    Mesure combien de rendement annuel tu obtiens
    par unité de drawdown maximum supporté.

    Exemple :
    CAGR = 33%, Max Drawdown = -22%
    Calmar = 0.33 / 0.22 = 1.5

    > 1.0 = acceptable
    > 2.0 = bon
    > 3.0 = excellent
    """
    equity_clean = equity.dropna()
    mdd = max_drawdown(equity_clean)
    if mdd == 0:
        return 0.0
    return float(cagr(equity_clean, periods) / abs(mdd))

def win_rate(returns: pd.Series) -> float:
    """
    Pourcentage de jours avec un rendement positif.

    Attention : un win rate de 60% ne dit rien sans
    connaître le profit factor. Une stratégie peut avoir
    40% de trades gagnants et être très rentable si les
    gains sont 3× supérieurs aux pertes.
    """
    positive_days = (returns > 0).sum()
    total_days = (returns != 0).sum()
    if total_days == 0:
        return 0.0
    return float(positive_days / total_days)


def profit_factor(returns: pd.Series) -> float:
    """
    Profit Factor = somme des gains / somme des pertes (en valeur absolue)

    Interprétation :
    < 1.0  → stratégie perdante
    1.0    → à l'équilibre
    > 1.5  → bonne stratégie
    > 2.0  → très bonne stratégie

    Exemple :
    Gains totaux : 5 000 €
    Pertes totales : 2 500 €
    Profit Factor = 5000 / 2500 = 2.0
    """
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float('inf')
    return float(gains / losses)


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Value at Risk (VaR) au niveau de confiance donné.

    Répond à la question : "Quelle est la pire perte
    journalière que je peux subir 95% du temps ?"

    np.percentile(returns, 5) : le 5e percentile des rendements.
    5% du temps, la perte sera pire que cette valeur.

    Exemple :
    VaR 95% = -2.1%
    Signification : 95% des jours, la perte ne dépassera pas 2.1%.
    Les 5% restants peuvent être pires.
    """
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def full_metrics(returns: pd.Series, equity: pd.Series) -> dict:
    """
    Calcule toutes les métriques en une seule fois.
    C'est cette fonction qu'on appelle depuis engine.py.

    Args:
        returns : Series des rendements journaliers de la stratégie
        equity  : Series de la valeur du portefeuille en euros

    Returns:
        dict avec toutes les métriques, arrondies proprement
    """
    r = returns.dropna()

    return {
        "CAGR (%)":          round(cagr(equity) * 100, 2),
        "Sharpe":            round(sharpe_ratio(r), 2),
        "Calmar":            round(calmar_ratio(equity), 2),
        "Max Drawdown (%)":  round(max_drawdown(equity) * 100, 2),
        "Win Rate (%)":      round(win_rate(r) * 100, 2),
        "Profit Factor":     round(profit_factor(r), 2),
        "VaR 95% (%)":       round(value_at_risk(r) * 100, 2),
        "Capital final (€)": round(equity.iloc[-1], 2),
        "Nb jours tradés":   int((returns != 0).sum()),
    }
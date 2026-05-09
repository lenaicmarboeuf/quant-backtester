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


def cagr(equity, periods=252):
    equity_clean = equity.dropna()
    if len(equity_clean) < 2 or equity_clean.iloc[0] == 0:
        return 0.0
    n_years = len(equity_clean) / periods
    if n_years < 0.1:          # ← moins de ~25 jours : pas fiable
        return 0.0
    return float((equity_clean.iloc[-1] / equity_clean.iloc[0]) ** (1/n_years) - 1)

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
    if abs(mdd) < 1e-10:
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

def sortino_ratio(returns: pd.Series, periods: int = 252) -> float:
    """
    Comme Sharpe mais ne pénalise que la volatilité négative.
    
    Pourquoi c'est mieux que Sharpe dans certains cas ?
    Sharpe pénalise la volatilité TOTALE : hausse ET baisse.
    Sortino dit : "une hausse volatile n'est pas un risque."
    Seule la volatilité à la baisse (downside) est du vrai risque.
    
    downside = returns[returns < 0] : garde uniquement les jours perdants
    downside.std()                  : vol des mauvais jours seulement
    
    Sortino > Sharpe → la stratégie a beaucoup de volatilité haussière
    (bonne chose). Sortino ≈ Sharpe → gains et pertes symétriques.
    """
    downside = returns[returns < 0]
    if downside.std() < 1e-10:
        return 0.0
    return (returns.mean() / downside.std()) * np.sqrt(periods)


def nb_trades(position: pd.Series) -> int:
    """
    Compte les entrées en position (passages de 0 → valeur positive).
    
    position.diff() : différence entre position[j] et position[j-1]
    > 0             : la position a augmenté = nouvelle entrée en trade
    
    Exemple :
    position = [0, 0, 1, 1, 0, 0, 1, 1, 1]
    diff     = [NaN, 0, +1, 0, -1, 0, +1, 0, 0]
    > 0      = [F, F, T, F, F, F, T, F, F]
    → 2 trades
    
    Pourquoi c'est important ?
    Nb jours tradés = 500 peut vouloir dire 1 trade de 500 jours
    ou 250 trades de 2 jours. Le turnover change tout pour les coûts.
    """
    return int((position.diff() > 0).sum())



def full_metrics(returns: pd.Series, equity: pd.Series,
                 position: pd.Series = None) -> dict:   
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
    e = equity.ffill().dropna()
    
    metrics = {
        "CAGR (%)":          round(cagr(e) * 100, 2),
        "Sharpe":            round(sharpe_ratio(r), 2),
        "Sortino":           round(sortino_ratio(r), 2),       
        "Calmar":            round(calmar_ratio(e), 2),
        "Max Drawdown (%)":  round(max_drawdown(e) * 100, 2),
        "Win Rate (%)":      round(win_rate(r) * 100, 2),
        "Profit Factor":     round(profit_factor(r), 2),
        "VaR 95% (%)":       round(value_at_risk(r) * 100, 2),
        "Capital final (€)": round(e.iloc[-1], 2),
        "Nb jours tradés":   int((returns != 0).sum()),
        "Nb trades":         nb_trades(position) if position is not None else 0,  
    }
    return metrics
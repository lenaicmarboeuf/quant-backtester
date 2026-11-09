"""
backtester/costs.py — Modèles de coûts de transaction réalistes

Pourquoi modéliser les coûts ?
-------------------------------
Un backtest sans coûts est un mensonge poli.
En réalité, chaque trade a un coût :

  1. Commission         : frais fixe ou proportionnel du broker
  2. Spread bid-ask     : on achète au Ask, on vend au Bid → perte immédiate
  3. Slippage           : entre la décision et l'exécution, le prix bouge
  4. Impact marché      : pour les gros ordres, notre propre achat fait monter le prix

Impact typique sur un backtest :
  - Stratégie HFT (100 trades/jour) → coûts = tueur absolu
  - Stratégie mensuelle (1 trade/mois) → coûts = quasi-négligeables
  - La plupart des stratégies daily → 10-30 bps A/R font la différence

Références :
  - Chan, "Algorithmic Trading" Ch.6 (coûts de transaction)
  - Almgren & Chriss (2001) : modèle d'impact marché racine carrée
  - QuantStart : "Realistic Backtesting with Transaction Costs"
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. Coûts simples
# ─────────────────────────────────────────────────────────────────────────────

def commission_cost(
    prices: pd.Series,
    positions: pd.Series,
    rate_bps: float = 5.0,
) -> pd.Series:
    """
    Coût de commission proportionnel au volume tradé.

    Paramètres
    ----------
    prices      : Series des prix de clôture
    positions   : Series des positions (entre 0 et 1)
    rate_bps    : taux de commission en points de base (défaut : 5 bps = 0.05%)

    Retourne
    --------
    Series des coûts journaliers en fraction du capital (négatifs = pertes)

    Exemple :
        5 bps = 0.05% par trade aller
        Aller-retour = 0.10% ≈ 10 bps
        Sur un capital de 10 000€ : 10€ par aller-retour
    """
    rate         = rate_bps / 10_000  # bps → fraction décimale
    # Le coût est payé quand la position CHANGE (on trade)
    turnover     = positions.diff().abs().fillna(0)
    return -(turnover * rate)


def spread_cost(
    prices: pd.Series,
    positions: pd.Series,
    half_spread_bps: float = 2.5,
) -> pd.Series:
    """
    Coût du spread bid-ask (demi-spread payé à chaque traversée).

    Explication :
        Si le spread BTC/USD est de 10$, le prix mid est par ex. 50 000$.
        On achète à 50 005$ et on vend à 49 995$ → on "perd" 5$ dès l'achat.
        C'est le demi-spread.

    Paramètres
    ----------
    prices           : Series des prix
    positions        : Series des positions [0, 1]
    half_spread_bps  : demi-spread en bps (défaut : 2.5 bps)
                       Large-cap US typique : 0.5–2 bps
                       Mid-cap : 2–5 bps
                       Small-cap / crypto : 10–50 bps

    Retourne
    --------
    Series des coûts journaliers (fraction du capital, négatifs)
    """
    half_spread = half_spread_bps / 10_000
    turnover    = positions.diff().abs().fillna(0)
    return -(turnover * half_spread)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Slippage
# ─────────────────────────────────────────────────────────────────────────────

def slippage_model(
    prices: pd.Series,
    positions: pd.Series,
    volatility: pd.Series = None,
    fixed_bps: float = 2.0,
    vol_multiplier: float = 0.1,
) -> pd.Series:
    """
    Modèle de slippage : coût lié à l'exécution imparfaite.

    Le slippage se produit entre la décision (prix de clôture J)
    et l'exécution réelle (ouverture J+1 ou VWAP J+1).

    Deux composantes :
    1. Slippage fixe       : représente la latence et les frais fixes
    2. Slippage volatilité : plus le marché bouge, plus l'exécution dévie

    slippage = fixed_bps + vol_multiplier × vol_réalisée

    Paramètres
    ----------
    prices         : Series des prix de clôture
    positions      : Series des positions [0, 1]
    volatility     : Series de vol réalisée (si None, calculée sur 20j)
    fixed_bps      : slippage fixe en bps (défaut : 2 bps)
    vol_multiplier : sensibilité à la volatilité (défaut : 0.1)
                     → vol de 20%/an = 1.26%/j → slippage ≈ 0.13 bps supp.

    Retourne
    --------
    Series des coûts de slippage (fraction du capital, négatifs)
    """
    if volatility is None:
        # Vol réalisée journalière annualisée (proxy)
        daily_vol = prices.pct_change().rolling(20).std()
    else:
        daily_vol = volatility

    # Slippage en bps selon la volatilité du moment
    slippage_bps = fixed_bps + vol_multiplier * (daily_vol * 100)  # vol en %
    slippage_rate = slippage_bps / 10_000

    turnover = positions.diff().abs().fillna(0)
    return -(turnover * slippage_rate).fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Impact marché — modèle racine carrée (Almgren-Chriss)
# ─────────────────────────────────────────────────────────────────────────────

def market_impact(
    order_size_pct: pd.Series,
    adv_ratio: float = 0.01,
    volatility: pd.Series = None,
    prices: pd.Series = None,
    eta: float = 0.1,
) -> pd.Series:
    """
    Modèle d'impact marché racine carrée (Almgren & Chriss, 2001).

    Problème : un gros ordre déplace les prix contre nous.
    Si on achète pour 1% du volume journalier d'AAPL, notre achat
    lui-même fait monter le prix avant qu'on ait fini d'exécuter.

    Formule :
        impact = η × σ × √(Q / ADV)

    Où :
        η   : coefficient d'impact (défaut 0.1 = calibration empirique)
        σ   : volatilité journalière (en fraction)
        Q   : taille de notre ordre (en % du capital)
        ADV : volume journalier moyen (en % du capital — ratio)

    Exemple :
        σ = 1.5%/j, Q = 1% du capital, ADV_ratio = 1%
        impact = 0.1 × 0.015 × √(0.01 / 0.01) = 0.1 × 0.015 × 1 = 0.15%

    Paramètres
    ----------
    order_size_pct : Series des variations de position (= turnover)
    adv_ratio      : rapport ordre/ADV (défaut : 1% → ordre = 1% du volume)
    volatility     : vol journalière (si None, calculée depuis prices)
    prices         : nécessaire si volatility est None
    eta            : coefficient d'impact Almgren-Chriss (défaut 0.1)

    Retourne
    --------
    Series d'impact marché (fraction du capital, négatifs)
    """
    if volatility is None and prices is not None:
        volatility = prices.pct_change().rolling(20).std()
    elif volatility is None:
        volatility = pd.Series(0.015, index=order_size_pct.index)

    # Participation rate = taille ordre / ADV
    participation = order_size_pct.abs() / (adv_ratio + 1e-9)

    # Modèle racine carrée : impact ∝ σ × √(participation)
    impact = eta * volatility * np.sqrt(participation.clip(0))

    return -(impact * order_size_pct.abs()).fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Coût total
# ─────────────────────────────────────────────────────────────────────────────

def total_transaction_cost(
    prices: pd.Series,
    positions: pd.Series,
    commission_bps: float = 5.0,
    spread_bps: float = 2.5,
    slippage_bps: float = 2.0,
    include_market_impact: bool = False,
    adv_ratio: float = 0.01,
) -> pd.Series:
    """
    Coût total de transaction = commission + spread + slippage (+ impact).

    C'est la fonction principale à appeler depuis engine.py.

    Coûts typiques pour une stratégie retail daily sur large-cap US :
        Commission   :  1–5 bps (IB, IBKR : ~1 bps)
        Spread       :  0.5–2 bps (large-cap liquide)
        Slippage     :  2–5 bps
        Total A/R    : ~5–15 bps aller-retour

    Paramètres
    ----------
    prices                 : Series des prix de clôture
    positions              : Series des positions [0, 1]
    commission_bps         : commission en bps (défaut : 5)
    spread_bps             : spread total bid-ask en bps (défaut : 2.5 = demi-spread × 2)
    slippage_bps           : slippage fixe en bps (défaut : 2)
    include_market_impact  : True si ordre > 1% de l'ADV
    adv_ratio              : ratio ordre/ADV (pour l'impact marché)

    Retourne
    --------
    Series des coûts totaux journaliers (fraction du capital, négatifs)
    """
    comm = commission_cost(prices, positions, rate_bps=commission_bps)
    spr  = spread_cost(prices, positions, half_spread_bps=spread_bps / 2)
    slip = (slippage_model(prices, positions, fixed_bps=slippage_bps)
        if slippage_bps > 0 else 0.0)

    total = comm + spr + slip

    if include_market_impact:
        turnover = positions.diff().abs().fillna(0)
        mi = market_impact(turnover, adv_ratio=adv_ratio, prices=prices)
        total += mi

    return total


# ─────────────────────────────────────────────────────────────────────────────
# 5. Analyse du turnover
# ─────────────────────────────────────────────────────────────────────────────

def turnover_analysis(
    positions: pd.Series,
    returns: pd.Series,
    cost_per_turnover_bps: float = 10.0,
) -> dict:
    """
    Analyse du turnover et de son coût sur la stratégie.

    Le turnover mesure à quelle fréquence on change de position.
    Un turnover élevé = coûts élevés = Sharpe net souvent décevant.

    Paramètres
    ----------
    positions             : Series des positions [0, 1]
    returns               : Series des rendements bruts de la stratégie
    cost_per_turnover_bps : coût aller-retour total en bps (défaut : 10 bps)

    Retourne
    --------
    dict avec :
        'daily_turnover'         : turnover moyen par jour
        'annual_turnover'        : turnover annualisé (× 252)
        'gross_sharpe'           : Sharpe sans coûts
        'net_sharpe'             : Sharpe après coûts
        'sharpe_drag'            : perte de Sharpe due aux coûts
        'cost_per_year_pct'      : coût annuel estimé en % du capital
        'breakeven_turnover'     : turnover max pour rester profitables
    """
    turnover_daily  = positions.diff().abs().fillna(0)
    avg_daily_to    = float(turnover_daily.mean())
    annual_to       = avg_daily_to * 252

    cost_daily      = turnover_daily * (cost_per_turnover_bps / 10_000)
    net_returns     = returns - cost_daily

    # Sharpe brut et net
    r_clean = returns.dropna()
    n_clean = net_returns.dropna()

    gross_sharpe = (r_clean.mean() / r_clean.std() * np.sqrt(252)
                    if r_clean.std() > 0 else 0.0)
    net_sharpe   = (n_clean.mean() / n_clean.std() * np.sqrt(252)
                    if n_clean.std() > 0 else 0.0)

    cost_per_year_pct = annual_to * (cost_per_turnover_bps / 10_000) * 100

    # Turnover break-even : à quel niveau de turnover le Sharpe tombe à 0 ?
    # Sharpe net = (μ - c×TO) / σ × √252 = 0
    # → TO_breakeven = μ / c
    if r_clean.std() > 0 and cost_per_turnover_bps > 0:
        cost_rate = cost_per_turnover_bps / 10_000
        breakeven_to = float(r_clean.mean() / (cost_rate / 252))
    else:
        breakeven_to = float("inf")

    return {
        "daily_turnover (%)":    round(avg_daily_to * 100, 3),
        "annual_turnover (×)":   round(annual_to, 2),
        "gross_sharpe":          round(gross_sharpe, 3),
        "net_sharpe":            round(net_sharpe, 3),
        "sharpe_drag":           round(gross_sharpe - net_sharpe, 3),
        "cost_per_year (%)":     round(cost_per_year_pct, 3),
        "breakeven_turnover (×)":round(breakeven_to, 2) if breakeven_to != float("inf") else "∞",
    }


def breakeven_cost(
    gross_sharpe: float,
    annual_turnover: float,
    periods: int = 252,
) -> float:
    """
    Quel coût aller-retour (en bps) la stratégie peut-elle absorber ?

    Autrement dit : si le Sharpe brut est X et le turnover est T fois/an,
    quel est le coût maximum avant que le Sharpe net tombe à zéro ?

    Formule :
        breakeven_cost = gross_sharpe × σ / (annual_turnover × √periods)

    Retourne le coût break-even en bps.

    Exemple :
        Sharpe brut = 1.2, turnover = 50× /an
        breakeven = 1.2 / 50 × √252 × 10_000 ≈ 381 bps → très tolérant
        Mais si turnover = 500×/an : 38 bps → plus raisonnable
    """
    if annual_turnover <= 0:
        return float("inf")
    # μ_net = μ_gross - cost × TO/252 ≥ 0
    # μ_gross ≈ Sharpe × σ / √252 ; coût break-even en fraction journalière
    # Approximation : Sharpe_net = 0 → cost_rate = μ_gross / (TO/252)
    gross_return_approx = gross_sharpe / np.sqrt(periods)  # rendement par période normalisé
    cost_rate_daily = gross_return_approx / (annual_turnover / periods)
    return round(cost_rate_daily * 10_000, 2)  # en bps

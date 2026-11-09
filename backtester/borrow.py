"""
backtester/borrow.py — Coût d'emprunt des titres (short selling)

Séance 9 : "Gagner dans toutes les directions"

Pourquoi un module dédié ?
--------------------------
Quand on vend à découvert, on EMPRUNTE les titres à un prêteur (via son broker)
et on lui paie une rémunération : le "borrow fee" (ou "stock loan fee").
Ce coût s'accumule TOUS LES JOURS où la position short est ouverte,
contrairement aux commissions/spread/slippage qui ne sont payés qu'au moment de trader.

Ordres de grandeur (annualisés) :
    Large-cap US très liquide (AAPL, MSFT, KO...) : 10 – 50 bps/an   (0.1 – 0.5 %)
    Mid-cap standard                              : 50 – 150 bps/an
    Small-cap / "hard to borrow"                  : 500 – 5 000+ bps/an (5 % à 50 %+)

Ce fichier ne modifie PAS costs.py : les coûts de trading existants utilisent
`positions.diff().abs()`, qui fonctionne déjà avec des positions signées
(passer de +1 à -1 = turnover de 2 = deux jambes à payer).
"""

import numpy as np
import pandas as pd


def borrow_cost(
    positions: pd.Series,
    borrow_bps: float = 30.0,
    periods: int = 252,
) -> pd.Series:
    """
    Coût d'emprunt journalier d'une position (seule la partie SHORT paie).

    Formule :
        coût_t = max(-position_t, 0) × borrow_bps / 10 000 / periods

    Exemple chiffré :
        position = -0.5 (50 % du capital en short), borrow = 30 bps/an
        coût par jour = 0.5 × 0.0030 / 252 = 0.00000595 (≈ 0.0006 %)
        coût annuel   = 0.5 × 0.30 % = 0.15 % du capital

    Paramètres
    ----------
    positions  : Series des positions DÉTENUES pendant la journée t
                 (donc déjà décalées de 1 jour : la position décidée à la
                 clôture de T-1 est celle qu'on porte pendant T)
    borrow_bps : frais d'emprunt ANNUALISÉS en points de base
    periods    : jours de bourse par an (252)

    Retourne
    --------
    Series des coûts journaliers (fraction du capital, NÉGATIFS = pertes)
    """
    short_exposure = (-positions).clip(lower=0).fillna(0)
    return -(short_exposure * borrow_bps / 10_000 / periods)


def borrow_cost_panel(
    weights: pd.DataFrame,
    borrow_bps=30.0,
    periods: int = 252,
) -> pd.Series:
    """
    Coût d'emprunt journalier d'un portefeuille multi-actifs.

    Paramètres
    ----------
    weights    : DataFrame (dates × actifs) des poids DÉTENUS pendant t
                 (déjà décalés de 1 jour)
    borrow_bps : float (même taux pour tous) OU dict/Series {ticker: bps}
                 pour des taux par actif (ex : small-caps plus chères)

    Retourne
    --------
    Series des coûts journaliers du portefeuille (fraction du capital, négatifs)
    """
    short_exposure = (-weights).clip(lower=0).fillna(0)
    if isinstance(borrow_bps, (dict, pd.Series)):
        rates = pd.Series(borrow_bps).reindex(weights.columns).fillna(0.0)
        daily = short_exposure.mul(rates / 10_000 / periods, axis=1).sum(axis=1)
    else:
        daily = short_exposure.sum(axis=1) * borrow_bps / 10_000 / periods
    return -daily

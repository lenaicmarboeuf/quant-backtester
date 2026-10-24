"""
backtester/combination.py — Combinaison intelligente de signaux

Séance 8 : Pondération par IC et ICIR

Problème de la pondération à poids égaux :
    On suppose implicitement que tous les signaux ont la même qualité.
    C'est rarement vrai : en période de tendance, le MACD est meilleur ;
    en période de range, le RSI surperforme.

Solution : pondérer proportionnellement à la qualité mesurée.
    La qualité d'un signal = son IC (corrélation signal → rendement futur).
    Plus l'IC est élevé, plus le signal a de poids dans le composite.

Trois modes disponibles :
    1. 'equal'  : poids égaux (1/N) — comportement original de signals.py
    2. 'ic'     : pondération par IC moyen sur ic_window jours
    3. 'icir'   : pondération par ICIR (régularité du signal)
    4. 'vol_ic' : IC pondéré par vol inverse (bonus)

Référence :
    Grinold & Kahn, "Active Portfolio Management" (2000)
    → Fundamental Law of Active Management : Sharpe ≈ IC × √(N_paris)
"""

import numpy as np
import pandas as pd
from scipy import stats

from backtester.factors import compute_rolling_ic, compute_icir


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires internes
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_weights(weights: pd.Series) -> pd.Series:
    """
    Normalise des poids pour qu'ils somment à 1, en ignorant les NaN.

    Si tous les poids sont nuls ou NaN → retourne poids égaux.
    Filtre les poids négatifs (un signal avec IC < 0 est ignoré).

    Exemple :
        weights = {'sma': 0.6, 'rsi': 0.0, 'macd': 0.4}
        → normalisé = {'sma': 0.6, 'rsi': 0.0, 'macd': 0.4} (somme = 1.0)

        weights = {'sma': -0.1, 'rsi': 0.3, 'macd': 0.2}
        → on met sma à 0 car IC négatif → {'sma': 0, 'rsi': 0.6, 'macd': 0.4}
    """
    # Les signaux avec IC négatif ne reçoivent pas de poids
    w = weights.clip(lower=0).fillna(0)

    total = w.sum()
    if total < 1e-9:
        # Fallback : poids égaux si tous les IC sont nuls/négatifs
        n = len(w)
        return pd.Series(1 / n, index=w.index)

    return w / total


# ─────────────────────────────────────────────────────────────────────────────
# Calcul des poids IC rolling par signal
# ─────────────────────────────────────────────────────────────────────────────

def rolling_ic_weights(
    signals: dict,
    forward_returns: pd.Series,
    window: int = 63,
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Calcule les poids IC dynamiques pour chaque signal sur une fenêtre glissante.

    À chaque date T :
        1. On calcule l'IC rolling de chaque signal sur les 'window' derniers jours.
        2. On normalise les IC positifs → ils deviennent les poids.
        3. Les signaux avec IC négatif (ou ≤ 0) reçoivent un poids de 0.

    Pourquoi exclure les IC négatifs ?
    Un IC < 0 signifie que le signal est INVERSEMENT corrélé aux rendements futurs.
    On ne veut pas pondérer négativement (ce serait inverser le signal).
    On préfère l'exclure et augmenter les poids des autres.

    Args:
        signals        : dict {'nom': pd.Series} des signaux
        forward_returns: rendements J+1 (close.pct_change().shift(-1))
        window         : fenêtre IC rolling (défaut 63 = 3 mois)
        method         : 'spearman' ou 'pearson'

    Returns:
        DataFrame (index=dates, columns=noms des signaux) des poids normalisés
        → chaque ligne somme à 1.0
    """
    # Calcul de l'IC rolling pour chaque signal
    ic_dict = {}
    for name, sig in signals.items():
        ic_rolling = compute_rolling_ic(sig, forward_returns, window=window, method=method)
        ic_dict[name] = ic_rolling

    ic_df = pd.DataFrame(ic_dict)

    # Normalisation ligne par ligne (chaque date)
    weights_list = []
    for date, row in ic_df.iterrows():
        w = _normalize_weights(row)
        weights_list.append(w)

    weights_df = pd.DataFrame(weights_list, index=ic_df.index)
    return weights_df


def rolling_icir_weights(
    signals: dict,
    forward_returns: pd.Series,
    ic_window: int = 63,
    icir_window: int = 63,
) -> pd.DataFrame:
    """
    Calcule les poids ICIR dynamiques (régularité du signal dans le temps).

    L'ICIR = IC_moyen / IC_std → récompense les signaux stables et fiables.
    Un signal avec IC = 0.08 tous les jours est préféré à un signal
    avec IC = 0.15 certains jours et -0.05 d'autres jours.

    Args:
        signals      : dict {'nom': pd.Series} des signaux
        forward_returns: rendements J+1
        ic_window    : fenêtre pour l'IC rolling
        icir_window  : fenêtre pour la moyenne/std de l'IC (ICIR)

    Returns:
        DataFrame (index=dates, columns=noms) des poids ICIR normalisés
    """
    icir_dict = {}
    for name, sig in signals.items():
        ic_rolling = compute_rolling_ic(sig, forward_returns, window=ic_window)
        icir = compute_icir(ic_rolling, window=icir_window)
        icir_dict[name] = icir

    icir_df = pd.DataFrame(icir_dict)

    weights_list = []
    for date, row in icir_df.iterrows():
        w = _normalize_weights(row)
        weights_list.append(w)

    return pd.DataFrame(weights_list, index=icir_df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale de combinaison
# ─────────────────────────────────────────────────────────────────────────────

def combine_signals(
    signals: dict,
    weights_mode: str = "equal",
    forward_returns: pd.Series = None,
    ic_window: int = 63,
    fixed_weights: dict = None,
) -> pd.Series:
    """
    Combine plusieurs signaux avec différents modes de pondération.

    Modes disponibles :
    ──────────────────
    'equal'  (défaut) :
        Poids égaux (1/N pour chaque signal).
        Simple, robuste, difficile à battre en pratique.
        → Idéal quand on ne sait pas lequel est meilleur.

    'fixed' :
        Poids fixes définis par l'utilisateur (fixed_weights dict).
        Ex : {'sma': 0.5, 'rsi': 0.3, 'macd': 0.2}
        → Idéal si on a une conviction a priori sur les signaux.

    'ic' :
        Pondération par IC moyen rolling sur ic_window jours.
        Les signaux avec meilleur IC prédictif récent ont plus de poids.
        ⚠️  Nécessite forward_returns.
        → Idéal pour adapter la pondération aux régimes de marché.

    'icir' :
        Pondération par ICIR (IC moyen / IC std).
        Favorise les signaux réguliers plutôt que les erratiques.
        ⚠️  Nécessite forward_returns.
        → Idéal pour sélectionner les signaux les plus fiables.

    Args:
        signals        : dict {'nom': pd.Series} des signaux entre -1 et +1
        weights_mode   : 'equal', 'fixed', 'ic', ou 'icir'
        forward_returns: rendements J+1 (obligatoire si mode != 'equal'/'fixed')
                         ⚠️  = close.pct_change().shift(-1) — look-ahead intentionnel
                         pour mesurer la qualité, PAS pour générer les positions.
        ic_window      : fenêtre IC rolling en jours (défaut 63)
        fixed_weights  : dict de poids si weights_mode='fixed'

    Returns:
        pd.Series normalisée entre -1 et +1

    Exemples :
        # Poids égaux
        sig = combine_signals({'sma': s1, 'rsi': s2, 'macd': s3})

        # Poids fixes
        sig = combine_signals({'sma': s1, 'rsi': s2},
                               weights_mode='fixed',
                               fixed_weights={'sma': 0.7, 'rsi': 0.3})

        # Pondération par IC (adaptatif)
        fwd = close.pct_change().shift(-1)
        sig = combine_signals({'sma': s1, 'rsi': s2},
                               weights_mode='ic',
                               forward_returns=fwd)
    """
    names = list(signals.keys())

    # ── Mode 1 : poids égaux ─────────────────────────────────────────────
    if weights_mode == "equal":
        w = 1 / len(signals)
        combined = sum(signals[name] * w for name in names)
        return combined.clip(-1, 1)

    # ── Mode 2 : poids fixes ─────────────────────────────────────────────
    if weights_mode == "fixed":
        if fixed_weights is None:
            raise ValueError("fixed_weights requis pour weights_mode='fixed'")
        total = sum(fixed_weights.values())
        combined = sum(signals[name] * (fixed_weights.get(name, 0) / total)
                       for name in names)
        return combined.clip(-1, 1)

    # ── Modes IC et ICIR : nécessitent forward_returns ───────────────────
    if forward_returns is None:
        raise ValueError(
            f"forward_returns requis pour weights_mode='{weights_mode}'.\n"
            "Utilise : forward_returns = close.pct_change().shift(-1)\n"
            "Ce shift(-1) est intentionnel pour MESURER la qualité du signal,\n"
            "pas pour générer des positions (pas de look-ahead bias dans le backtest)."
        )

    # Calcul des poids dynamiques
    if weights_mode == "ic":
        weights_df = rolling_ic_weights(signals, forward_returns, window=ic_window)
    elif weights_mode == "icir":
        weights_df = rolling_icir_weights(signals, forward_returns, ic_window=ic_window)
    else:
        raise ValueError(f"weights_mode inconnu : '{weights_mode}'. "
                         f"Choisir parmi : 'equal', 'fixed', 'ic', 'icir'")

    # ANTI LOOK-AHEAD : l'IC à la date T utilise forward_returns[T] = rendement
    # de T+1, inconnu à la clôture de T. On décale donc les poids d'un jour :
    # le poids de T n'utilise que l'IC calculé jusqu'à T-1.
    weights_df = weights_df.shift(1)

    # Application des poids dynamiques signal par signal
    # weights_df a le même index que les IC (moins les premières valeurs NaN)
    all_signals_df = pd.DataFrame(signals)

    # Aligner les index
    common_idx = weights_df.index.intersection(all_signals_df.index)
    weights_aligned = weights_df.loc[common_idx]
    signals_aligned = all_signals_df.loc[common_idx]

    # Combinaison pondérée : element-wise multiply + sum par ligne
    combined_values = (signals_aligned * weights_aligned).sum(axis=1)

    # Reconstruction en Series avec l'index complet (NaN pour les premières dates)
    result = pd.Series(np.nan, index=all_signals_df.index)
    result.loc[common_idx] = combined_values

    return result.clip(-1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Analyse comparative des modes de pondération
# ─────────────────────────────────────────────────────────────────────────────

def compare_weighting_modes(
    signals: dict,
    forward_returns: pd.Series,
    df: "pd.DataFrame" = None,
    initial_capital: float = 10_000.0,
    ic_window: int = 63,
) -> pd.DataFrame:
    """
    Compare les 4 modes de pondération sur la même période.

    Calcule le signal combiné pour chaque mode et, si df est fourni,
    lance un backtest simple pour comparer les performances.

    Args:
        signals        : dict {'nom': pd.Series} des signaux
        forward_returns: rendements J+1 pour les modes IC/ICIR
        df             : DataFrame OHLCV avec 'returns' (optionnel, pour backtest)
        initial_capital: capital initial si backtest lancé
        ic_window      : fenêtre IC rolling

    Returns:
        DataFrame avec les métriques de chaque mode (si df fourni)
        ou DataFrame avec les signaux combinés (si df=None)
    """
    modes = {
        "equal": combine_signals(signals, weights_mode="equal"),
        "ic":    combine_signals(signals, weights_mode="ic",
                                  forward_returns=forward_returns, ic_window=ic_window),
        "icir":  combine_signals(signals, weights_mode="icir",
                                  forward_returns=forward_returns, ic_window=ic_window),
    }

    if df is None:
        return pd.DataFrame(modes)

    # Backtest simple pour chaque mode
    from backtester.engine import Backtester
    records = []
    for mode_name, combined_signal in modes.items():
        bt = Backtester(df, initial_capital=initial_capital)
        out = bt.run(combined_signal)
        row = {"mode": mode_name}
        row.update(out["metrics"])
        records.append(row)

    return pd.DataFrame(records).set_index("mode")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic : rapport IC par signal
# ─────────────────────────────────────────────────────────────────────────────

def signal_quality_report(
    signals: dict,
    forward_returns: pd.Series,
    ic_window: int = 63,
) -> pd.DataFrame:
    """
    Rapport de qualité de chaque signal : IC moyen, ICIR, poids suggérés.

    Utile pour diagnostiquer quels signaux apportent de la valeur.

    Returns:
        DataFrame avec une ligne par signal :
            IC moyen   : corrélation moyenne signal → rendements futurs
            IC std     : volatilité de l'IC (régularité)
            ICIR       : IC moyen / IC std
            IC > 0 (%) : % du temps où l'IC est positif
            Poids IC   : poids normalisé si mode='ic'
            Poids ICIR : poids normalisé si mode='icir'
    """
    records = []
    ic_means = {}

    for name, sig in signals.items():
        ic_rolling = compute_rolling_ic(sig, forward_returns, window=ic_window)
        ic_clean = ic_rolling.dropna()

        if len(ic_clean) == 0:
            continue

        ic_mean = float(ic_clean.mean())
        ic_std  = float(ic_clean.std())
        icir    = ic_mean / ic_std if ic_std > 0 else 0.0
        ic_positive_pct = float((ic_clean > 0).mean() * 100)

        records.append({
            "Signal":      name,
            "IC moyen":    round(ic_mean, 4),
            "IC std":      round(ic_std, 4),
            "ICIR":        round(icir, 4),
            "IC > 0 (%)":  round(ic_positive_pct, 1),
        })
        ic_means[name] = ic_mean

    df_report = pd.DataFrame(records).set_index("Signal")

    # Poids suggérés
    ic_series = pd.Series(ic_means)
    w_ic   = _normalize_weights(ic_series)
    w_icir = _normalize_weights(pd.Series({
        r["Signal"]: r["ICIR"] for r in records
    }))

    df_report["Poids IC (%)"]   = (w_ic * 100).round(1)
    df_report["Poids ICIR (%)"] = (w_icir * 100).round(1)

    return df_report
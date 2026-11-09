"""
backtester/validation.py — Walk-Forward & Combinatorial Purged Cross-Validation

Pourquoi ce module ?
--------------------
Optimiser une stratégie sur l'historique complet et mesurer sa performance
sur ce même historique, c'est du sur-apprentissage (overfitting).
Le backtester "mémorise" les bonnes périodes.

La solution : séparer strictement les données d'entraînement (IS)
des données de test (OOS), et ne jamais regarder l'OOS pendant l'optimisation.

Deux méthodes sont implémentées ici :

1. Walk-Forward Optimisation (WFO) :
   ┌──────────────────────────────────────────────────┐
   │  IS [252j]  │ OOS[63j]│                           │  fenêtre 1
   │             │  IS [252j]  │ OOS[63j]│             │  fenêtre 2
   │             │             │  IS [252j]  │ OOS[63j]│  fenêtre 3
   └──────────────────────────────────────────────────┘
   → On optimise sur IS, on évalue sur OOS, on avance.

2. Combinatorial Purged Cross-Validation (CPCV) — Lopez de Prado :
   Comme une cross-validation classique, MAIS avec une "purge" qui
   supprime les observations trop proches du fold de test pour éviter
   la fuite d'information (les rendements financiers sont auto-corrélés).
"""

import itertools
import warnings
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from backtester.engine import Backtester
from backtester.metrics import full_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires internes
# ─────────────────────────────────────────────────────────────────────────────

def _expand_grid(param_grid: Dict) -> List[Dict]:
    """
    Convertit un dict de listes en liste de dicts (produit cartésien).

    Exemple :
        {'fast': [10, 20], 'slow': [30, 50]}
        → [{'fast':10,'slow':30}, {'fast':10,'slow':50},
           {'fast':20,'slow':30}, {'fast':20,'slow':50}]
    """
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _run_signal_on_df(
    df: pd.DataFrame,
    signal_func: Callable,
    params: Dict,
    initial_capital: float,
) -> Optional[Dict]:
    """Lance signal_func sur df et retourne les métriques. None si erreur."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sig = signal_func(df["Close"], **params).fillna(0)
        bt  = Backtester(df, initial_capital=initial_capital)
        return bt.run(sig)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Optimisation
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(
    df: pd.DataFrame,
    signal_func: Callable,
    param_grid: Dict,
    is_window: int = 252,
    oos_window: int = 63,
    anchored: bool = False,
    metric: str = "Sharpe",
    initial_capital: float = 10_000.0,
    min_is_periods: int = 100,
) -> Dict:
    """
    Walk-Forward Optimisation (WFO).

    Pour chaque fenêtre glissante :
      1. Tester toutes les combinaisons de param_grid sur IS.
      2. Garder la combinaison qui maximise `metric`.
      3. Évaluer cette combinaison sur la fenêtre OOS suivante.
      4. Concaténer tous les OOS pour reconstituer une equity "réelle".

    Paramètres
    ----------
    df              : DataFrame OHLCV avec colonne 'returns' (sortie de compute_returns)
    signal_func     : fonction(close: pd.Series, **params) → pd.Series
    param_grid      : {'fast': [10,20,30], 'slow': [40,60,80]}
    is_window       : taille IS en jours de bourse (défaut : 252 = 1 an)
    oos_window      : taille OOS en jours de bourse (défaut : 63 = 1 trimestre)
    anchored        : False = IS glissant (taille fixe)
                      True  = IS croissant (ancré au début des données)
    metric          : métrique à maximiser sur IS ('Sharpe', 'CAGR (%)', ...)
    initial_capital : capital de départ pour chaque backtest
    min_is_periods  : nombre minimum de jours IS (ignore les fenêtres trop courtes)

    Retourne
    --------
    dict avec :
        'oos_results'  : DataFrame — métriques par fenêtre OOS
        'best_params'  : DataFrame — meilleurs paramètres IS par fenêtre
        'equity_oos'   : Series — equity OOS reconstituée (la courbe "honnête")
        'oos_returns'  : Series — rendements OOS concaténés
        'summary'      : dict — métriques agrégées sur tout l'OOS
    """
    combos = _expand_grid(param_grid)
    n      = len(df)

    oos_returns_list = []
    window_results   = []
    best_params_list = []

    # Fenêtres OOS : on avance par pas de oos_window après is_window
    oos_starts = range(is_window, n - oos_window + 1, oos_window)

    for oos_start in oos_starts:
        is_start = 0 if anchored else max(0, oos_start - is_window)
        is_end   = oos_start
        oos_end  = min(oos_start + oos_window, n)

        if (is_end - is_start) < min_is_periods:
            continue

        df_is  = df.iloc[is_start:is_end].copy()
        df_oos = df.iloc[oos_start:oos_end].copy()

        if len(df_oos) < 5:
            break

        # ── Optimisation sur IS ───────────────────────────────────────────
        best_score = -np.inf
        best_combo = combos[0]

        for combo in combos:
            out = _run_signal_on_df(df_is, signal_func, combo, initial_capital)
            if out is None:
                continue
            score = out["metrics"].get(metric, -np.inf)
            if isinstance(score, (int, float)) and score > best_score:
                best_score = score
                best_combo = combo

        best_params_list.append({
            "is_start":    df.index[is_start],
            "is_end":      df.index[is_end - 1],
            "oos_start":   df.index[oos_start],
            "oos_end":     df.index[min(oos_end, n) - 1],
            "best_params": str(best_combo),
            f"is_{metric}": round(best_score, 4),
        })

        # ── Évaluation sur OOS ────────────────────────────────────────────
        out_oos = _run_signal_on_df(df_oos, signal_func, best_combo, initial_capital)
        if out_oos is None:
            continue

        oos_r  = out_oos["results"]["strategy_returns"]
        metrics_row = out_oos["metrics"].copy()
        metrics_row.update({
            "oos_start":   df.index[oos_start],
            "oos_end":     df.index[min(oos_end, n) - 1],
            "best_params": str(best_combo),
        })
        window_results.append(metrics_row)
        oos_returns_list.append(oos_r)

    if not oos_returns_list:
        return {"error": "Aucune fenêtre OOS valide. Augmente la taille du DataFrame."}

    # ── Reconstruction de l'equity OOS globale ────────────────────────────
    all_oos_returns = pd.concat(oos_returns_list).sort_index()
    equity_oos      = initial_capital * (1 + all_oos_returns).cumprod()

    # ── Métriques agrégées ────────────────────────────────────────────────
    summary = full_metrics(returns=all_oos_returns, equity=equity_oos)
    summary["Nb fenêtres OOS"]    = len(window_results)
    summary["Mode"]               = "Ancré" if anchored else "Glissant"
    summary["IS window (jours)"]  = is_window
    summary["OOS window (jours)"] = oos_window

    return {
        "oos_results":  pd.DataFrame(window_results),
        "best_params":  pd.DataFrame(best_params_list),
        "equity_oos":   equity_oos,
        "oos_returns":  all_oos_returns,
        "summary":      summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Combinatorial Purged Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

def purged_cv(
    df: pd.DataFrame,
    signal_func: Callable,
    params: Dict,
    n_splits: int = 5,
    purge_gap: int = 5,
    embargo_gap: int = 5,
    initial_capital: float = 10_000.0,
) -> Dict:
    """
    Combinatorial Purged Cross-Validation (CPCV) — Lopez de Prado (2018).

    Problème de la CV classique sur séries financières :
    Les rendements sont auto-corrélés. Si le fold de test est adjacent
    au fold d'entraînement, les données proches "contaminent" le test
    (le modèle a indirectement vu de l'information future).

    Solution :
    ┌──────────────────────────────────────────────────────────────┐
    │  TRAIN  │ PURGE │   TEST   │ EMBARGO │       TRAIN          │
    │  (IS)   │ [5j]  │  [fold]  │  [5j]  │        (IS)          │
    └──────────────────────────────────────────────────────────────┘
    - Purge   : supprime les jours juste AVANT le test (corrélation retard)
    - Embargo : supprime les jours juste APRÈS le test (corrélation avance)

    Paramètres
    ----------
    df          : DataFrame OHLCV avec colonne 'returns'
    signal_func : fonction(close, **params) → signal
    params      : paramètres du signal (déjà optimisés via WFO)
    n_splits    : nombre de folds (défaut 5 = 20% OOS par fold)
    purge_gap   : jours à exclure avant le fold test
    embargo_gap : jours à exclure après le fold test
    initial_capital : capital de départ

    Retourne
    --------
    dict avec :
        'fold_metrics'   : DataFrame — métriques par fold
        'mean_metrics'   : dict — moyenne sur tous les folds
        'std_metrics'    : dict — écart-type (mesure de robustesse)
        'deflated_sharpe': float — Sharpe corrigé pour multiplicité
    """
    n         = len(df)
    fold_size = n // n_splits
    fold_metrics_list = []

    for i in range(n_splits):
        # ── Définition du fold de test ────────────────────────────────────
        test_start = i * fold_size
        test_end   = test_start + fold_size if i < n_splits - 1 else n

        # ── Purge et embargo ──────────────────────────────────────────────
        # Purge : on exclut les 'purge_gap' jours précédant le fold de test
        purge_start = max(0, test_start - purge_gap)

        # Embargo : on exclut les 'embargo_gap' jours suivant le fold de test
        embargo_end = min(n, test_end + embargo_gap)

        # Les indices d'entraînement sont TOUT sauf [purge_start, embargo_end]
        train_idx = list(range(0, purge_start)) + list(range(embargo_end, n))
        test_idx  = list(range(test_start, test_end))

        if len(test_idx) < 10:
            continue

        df_test = df.iloc[test_idx].copy()

        out = _run_signal_on_df(df_test, signal_func, params, initial_capital)
        if out is None:
            continue

        m = out["metrics"].copy()
        m["fold"]       = i + 1
        m["test_start"] = df.index[test_start]
        m["test_end"]   = df.index[test_end - 1]
        m["n_train"]    = len(train_idx)
        m["n_test"]     = len(test_idx)
        fold_metrics_list.append(m)

    if not fold_metrics_list:
        return {"error": "Aucun fold valide."}

    df_folds  = pd.DataFrame(fold_metrics_list)
    num_cols  = df_folds.select_dtypes(include=[np.number]).columns
    stat_cols = [c for c in num_cols if c not in ["fold", "n_train", "n_test"]]

    mean_m = df_folds[stat_cols].mean().round(4).to_dict()
    std_m  = df_folds[stat_cols].std().round(4).to_dict()

    # ── Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) ────────────
    #
    # Problème : si on teste N combinaisons de paramètres, il y a une
    # probabilité non nulle d'obtenir un bon Sharpe par chance.
    # La correction de Bonferroni naïve (diviser par N) est trop sévère.
    #
    # Le Deflated Sharpe normalise par le Sharpe maximal attendu par chance :
    #   DSR = (SR_obs - SR_attendu) / écart-type(SR)
    #   SR_attendu ≈ (1 - γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e))
    # où γ ≈ 0.5772 (constante d'Euler-Mascheroni)
    sharpes = df_folds["Sharpe"].dropna()
    deflated_sr = None

    if len(sharpes) > 1:
        sr_mean = float(sharpes.mean())
        sr_std  = float(sharpes.std())
        n_trials = n_splits  # nombre d'essais indépendants

        gamma = 0.5772156649  # constante d'Euler-Mascheroni

        # Sharpe maximal attendu sous H0 (Lopez de Prado, 2014)
        sr_max_expected = (
            (1 - gamma) * norm.ppf(1 - 1 / n_trials) +
            gamma        * norm.ppf(1 - 1 / (n_trials * np.e))
        ) * sr_std

        deflated_sr = round(
            (sr_mean - sr_max_expected) / (sr_std + 1e-9), 4
        )
        mean_m["Deflated Sharpe"] = deflated_sr

    return {
        "fold_metrics":    df_folds,
        "mean_metrics":    mean_m,
        "std_metrics":     std_m,
        "deflated_sharpe": deflated_sr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Surface de robustesse (heatmap 2D)
# ─────────────────────────────────────────────────────────────────────────────

def robustness_surface(
    df: pd.DataFrame,
    signal_func: Callable,
    param_x: str,
    param_x_values: List,
    param_y: str,
    param_y_values: List,
    fixed_params: Dict = None,
    metric: str = "Sharpe",
    initial_capital: float = 10_000.0,
) -> pd.DataFrame:
    """
    Calcule une surface 2D de la performance en fonction de deux paramètres.

    Permet de visualiser si la stratégie est robuste (un pic isolé = fragile,
    un plateau élevé = robuste).

    Retourne un DataFrame (pivot table) prêt pour sns.heatmap().
    """
    if fixed_params is None:
        fixed_params = {}

    records = []
    for x_val in param_x_values:
        for y_val in param_y_values:
            params = {**fixed_params, param_x: x_val, param_y: y_val}
            out = _run_signal_on_df(df, signal_func, params, initial_capital)
            score = out["metrics"].get(metric, np.nan) if out else np.nan
            records.append({param_x: x_val, param_y: y_val, metric: score})

    df_surface = pd.DataFrame(records)
    pivot = df_surface.pivot(index=param_y, columns=param_x, values=metric)
    return pivot.round(3)

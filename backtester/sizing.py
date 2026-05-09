import pandas as pd
import numpy as np

# ============================================================
# VERSION OPTIMALE — backtester/sizing.py — Séance 5
# ============================================================
# 4 fonctions de sizing + 1 stop-loss ATR + 1 utilitaire
# Toutes retournent une Series entre [0, 1] avec shift(1).


def vol_target_size(
    returns: pd.Series,
    signal: pd.Series,
    vol_target: float = 0.10,
    window: int = 20
) -> pd.Series:
    """
    Taille de position basée sur le ciblage de volatilité.

    Paramètres
    ----------
    returns    : rendements journaliers (colonne 'returns' de ton df)
    signal     : signal brut [-1, +1] (ex: sma_crossover)
    vol_target : volatilité annuelle cible (défaut 10%)
    window     : fenêtre pour estimer la vol réalisée (défaut 20 jours)

    Retourne
    --------
    position : pd.Series entre [0, 1]

    Logique :
        vol_realized = std(returns, 20j) × √252   → vol annualisée
        raw_size     = vol_target / vol_realized   → taille brute
        size         = clip(raw_size, 0, 1)        → pas de levier
        position     = size × direction             → 0 si signal négatif
    """
    vol_realized = returns.rolling(window).std() * np.sqrt(252)
    raw_size = vol_target / vol_realized
    size = raw_size.clip(0, 1)
    direction = (signal > 0).astype(float)
    position = size * direction
    return position.shift(1).fillna(0)
  
    
def kelly_size(returns, signal, window=252, fraction=0.5):
    """
    Taille de position basée sur le critère de Kelly fractionné.

    Paramètres
    ----------
    returns  : rendements journaliers
    signal   : signal brut [-1, +1]
    window   : fenêtre d'estimation des stats (défaut 252 = 1 an)
    fraction : multiplicateur Kelly (0.5 = demi-Kelly, recommandé)

    Formule de Kelly :
        p  = taux de gain (% de jours positifs sur window jours)
        q  = 1 - p = taux de perte
        b  = gain_moyen / perte_moyenne
        f* = (p × b - q) / b

    Pourquoi demi-Kelly (fraction=0.5) ?
        Kelly plein maximise le log-capital mais est très agressif.
        En pratique, on utilise 50% de Kelly pour réduire le risque
        d'une estimation incorrecte des paramètres (p, b).

    Retourne
    --------
    position : pd.Series entre [0, 1]
    """

    win_indicator  = (returns > 0).astype(float)
    loss_indicator = (returns <= 0).astype(float)

    win_rate  = win_indicator.rolling(window, min_periods=30).mean()
    loss_rate = 1 - win_rate

    win_sum   = returns.clip(lower=0).rolling(window, min_periods=30).sum()
    win_count = win_indicator.rolling(window, min_periods=30).sum()
    avg_win   = win_sum / win_count.replace(0, np.nan)

    loss_sum   = returns.clip(upper=0).abs().rolling(window, min_periods=30).sum()
    loss_count = loss_indicator.rolling(window, min_periods=30).sum()
    avg_loss   = loss_sum / loss_count.replace(0, np.nan)

    b       = avg_win / avg_loss.replace(0, np.nan)
    kelly_f = ((win_rate * b - loss_rate) / b) * fraction
    size    = kelly_f.clip(0, 1)

    direction = (signal > 0).astype(float)
    return (size * direction).shift(1).fillna(0)

def fixed_fraction(signal: pd.Series, fraction: float = 0.02) -> pd.Series:
    """
    Taille de position fixe : fraction constante du capital.

    Paramètres
    ----------
    signal   : signal brut [-1, +1]
    fraction : fraction du capital par trade (défaut 2%)

    Retourne
    --------
    position : pd.Series entre [0, fraction]

    Note : contrairement aux autres fonctions, le max ici est
    'fraction' et non 1. Pour des comparaisons, utilise fraction=1.0.
    """
    direction = (signal > 0).astype(float)
    position  = direction * fraction
    return position.shift(1).fillna(0)


def signal_proportional_size(
    signal: pd.Series,
    returns: pd.Series,
    vol_adjust: bool = True,
    vol_target: float = 0.10,
    vol_window: int = 20,
    min_position: float = 0.0,
    power: float = 1.0
) -> pd.Series:
    """
    Taille proportionnelle à la force du signal.

    Plus le signal est fort → plus on investit. Moins le marché
    est volatil → plus on peut investir.

    Paramètres
    ----------
    signal       : signal continu entre [-1, +1]
    returns      : rendements journaliers (pour vol_adjust)
    vol_adjust   : si True, pondère par ciblage de volatilité
    vol_target   : vol annuelle cible si vol_adjust=True
    vol_window   : fenêtre de calcul de la vol
    min_position : position minimale quand signal > 0 (défaut 0)
                   Ex : 0.1 → au moins 10% même si signal faible
    power        : exposant du signal
                   1.0 = linéaire (défaut)
                   > 1 = convexe → récompense les signaux très forts
                   < 1 = concave → atténue les signaux forts

    Exemples avec power :
        signal=0.5, power=1.0 → taille = 0.50
        signal=0.5, power=2.0 → taille = 0.25 (pénalise l'hésitation)
        signal=0.5, power=0.5 → taille = 0.71 (favorise les petits signaux)

    Retourne
    --------
    position : pd.Series entre [0, 1]
    """
    # 1. Garder uniquement signal positif
    raw = signal.clip(0, 1)

    # 2. Appliquer l'exposant
    raw = raw ** power

    # 3. Position minimale quand signal positif
    direction = (signal > 0).astype(float)
    size = raw + min_position * direction
    size = size.clip(0, 1)

    # 4. Pondération par la volatilité
    if vol_adjust:
        vol_realized = returns.rolling(vol_window).std() * np.sqrt(252)
        vol_multiplier = (vol_target / vol_realized).clip(0, 1)
        size = size * vol_multiplier

    return size.clip(0, 1).shift(1).fillna(0)


def atr_stop(
    df: pd.DataFrame,
    signal: pd.Series,
    atr_window: int = 14,
    atr_multiplier: float = 2.0
) -> pd.DataFrame:
    """
    Stop-loss dynamique basé sur l'ATR (Average True Range).

    L'ATR mesure la volatilité "vraie" d'un actif en tenant compte
    des gaps (écarts entre la clôture de la veille et l'ouverture
    d'aujourd'hui). Plus l'ATR est grand → marché plus agité.

    True Range = max(
        High - Low,           # amplitude du jour
        |High - Close_prev|,  # gap haussier
        |Low  - Close_prev|   # gap baissier
    )

    ATR = moyenne mobile du True Range sur atr_window jours.

    Stop loss = Close - ATR × multiplier
        → si le prix tombe en dessous du stop, on sort.

    Exemple :
        Close = 150 €, ATR = 3 €, multiplier = 2
        Stop = 150 - 2 × 3 = 144 €
        → si le prix descend sous 144 €, position = 0

    Paramètres
    ----------
    df             : DataFrame OHLCV (colonnes High, Low, Close)
    signal         : signal brut [-1, +1]
    atr_window     : période ATR (défaut 14 — standard Wilder)
    atr_multiplier : distance du stop en multiples d'ATR (défaut 2)

    Retourne
    --------
    DataFrame avec 3 colonnes :
        'atr'          : ATR calculé
        'stop_level'   : niveau de stop en euros
        'position_atr' : position finale [0, 1]
    """
    result = df.copy()

    # --- Calcul du True Range ---
    hl = df['High'] - df['Low']                          # amplitude du jour
    hc = (df['High'] - df['Close'].shift(1)).abs()       # gap haussier
    lc = (df['Low']  - df['Close'].shift(1)).abs()       # gap baissier

    # max des 3 composantes, jour par jour
    true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    # ATR = moyenne simple du True Range
    result['atr'] = true_range.rolling(atr_window).mean()

    # --- Niveau de stop ---
    result['stop_level'] = df['Close'] - atr_multiplier * result['atr']

    # --- Application du stop ---
    direction      = (signal > 0).astype(float)
    stop_triggered = df['Close'] < result['stop_level']

    # Si stop déclenché → position = 0, sinon → direction normale
    result['position_atr'] = direction.where(~stop_triggered, 0)
    result['position_atr'] = result['position_atr'].shift(1).fillna(0)

    return result[['atr', 'stop_level', 'position_atr']]

def compare_sizing_modes(
    df: pd.DataFrame,
    signal: pd.Series,
    initial_capital: float = 10_000.0
) -> pd.DataFrame:
    """
    Compare les 5 modes de sizing sur la même période.

    Calcule la courbe de capital pour chaque mode et retourne
    un DataFrame avec les equity curves côte à côte.

    Args:
        df             : DataFrame avec colonnes 'returns' et OHLCV
        signal         : signal brut [-1, +1]
        initial_capital: capital de départ (défaut 10 000 €)

    Returns:
        DataFrame avec colonnes :
            'equity_binary', 'equity_vol', 'equity_kelly',
            'equity_fixed', 'equity_proportional', 'equity_atr'
    """
    results = pd.DataFrame(index=df.index)

    def equity_from_position(pos: pd.Series) -> pd.Series:
        """Calcule la courbe de capital depuis une position."""
        strat_returns = (pos * df['returns']).fillna(0)
        cum_returns   = (1 + strat_returns).cumprod() - 1
        return initial_capital * (1 + cum_returns)

    # 1. Binaire
    pos_binary = (signal > 0).astype(int).shift(1).fillna(0)
    results['equity_binary'] = equity_from_position(pos_binary)

    # 2. Vol-targeting
    pos_vol = vol_target_size(df['returns'], signal)
    results['equity_vol'] = equity_from_position(pos_vol)

    # 3. Kelly (demi)
    pos_kelly = kelly_size(df['returns'], signal)
    results['equity_kelly'] = equity_from_position(pos_kelly)

    # 4. Fixed fraction (fraction=1.0 pour comparer à iso-échelle)
    pos_fixed = fixed_fraction(signal, fraction=1.0)
    results['equity_fixed'] = equity_from_position(pos_fixed)

    # 5. Signal proportionnel
    pos_prop = signal_proportional_size(signal, df['returns'])
    results['equity_proportional'] = equity_from_position(pos_prop)

    # 6. ATR stop (direction du signal comme base)
    atr_result = atr_stop(df, signal)
    pos_atr    = atr_result['position_atr']
    results['equity_atr'] = equity_from_position(pos_atr)

    return results

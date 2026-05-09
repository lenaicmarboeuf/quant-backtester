import pandas as pd
import numpy as np
import pandas_ta as ta


def sma_crossover(close: pd.Series,
                  fast: int = 20,
                  slow: int = 50) -> pd.Series:
    """
    Signal de croisement de deux moyennes mobiles.

    Logique :
    - SMA rapide (20j) au-dessus de SMA lente (50j) → tendance haussière → signal +1
    - SMA rapide en dessous de SMA lente            → tendance baissière → signal -1

    Pourquoi deux SMA plutôt qu'une ?
    Une seule SMA compare le prix à sa moyenne — sensible au bruit.
    Deux SMA comparent deux tendances de durées différentes — plus fiable.

    Le croisement (fast passe au-dessus de slow) est un signal
    de changement de tendance — c'est le "Golden Cross" (haussier)
    et le "Death Cross" (baissier) en analyse technique.

    Args:
        close : Series des prix de clôture
        fast  : fenêtre de la SMA rapide (défaut 20 jours)
        slow  : fenêtre de la SMA lente  (défaut 50 jours)

    Returns:
        Series avec valeurs +1 (haussier) ou -1 (baissier)
    """
    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean()

    # np.where(condition, valeur_si_vrai, valeur_si_faux)
    signal = pd.Series(
        np.where(sma_fast > sma_slow, 1, -1),
        index=close.index
    )

    # Les premières 'slow' valeurs sont invalides (pas assez de données)
    signal.iloc[:slow] = np.nan

    return signal


def bollinger_zscore(close: pd.Series,
                     window: int = 20,
                     std_dev: float = 2.0) -> pd.Series:
    """
    Signal mean-reversion basé sur le z-score des Bandes de Bollinger.

    Les Bandes de Bollinger définissent une zone "normale" autour
    du prix. Quand le prix s'en éloigne trop, il tend à revenir
    vers la moyenne (mean-reversion).

    Calcul du z-score :
        z = (prix - moyenne) / écart-type
        z = 0   → prix exactement sur la moyenne
        z = +2  → prix 2 écarts-types au-dessus (survente potentielle)
        z = -2  → prix 2 écarts-types en dessous (survente potentielle)

    On normalise ensuite entre -1 et +1 en divisant par std_dev.
    Signal négatif → prix trop haut → on anticipe une baisse
    Signal positif → prix trop bas  → on anticipe une hausse

    C'est l'opposé du signal de tendance : ici on va CONTRE
    le mouvement récent en pariant sur un retour à la moyenne.

    Args:
        close   : Series des prix de clôture
        window  : fenêtre de calcul (défaut 20 jours)
        std_dev : nombre d'écarts-types pour les bandes (défaut 2.0)

    Returns:
        Series normalisée entre -1 et +1
    """
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()

    # Z-score : combien d'écarts-types est-on au-dessus de la moyenne ?
    zscore = (close - sma) / std

    # Normalisation : diviser par std_dev pour ramener entre -1 et +1
    # On clip pour forcer les valeurs dans [-1, 1] même en cas d'extrêmes
    signal = (-zscore / std_dev).clip(-1, 1)

    return signal


def rsi_signal(close: pd.Series,
               period: int = 14,
               overbought: int = 70,
               oversold: int = 30) -> pd.Series:
    """
    Signal basé sur le RSI (Relative Strength Index).

    Le RSI mesure la force relative des hausses vs les baisses
    sur une période donnée. Il oscille entre 0 et 100.

    Interprétation classique :
        RSI > 70 → surachat   (le prix a trop monté) → signal de vente
        RSI < 30 → survente   (le prix a trop baissé) → signal d'achat
        30 < RSI < 70 → zone neutre → pas de signal

    On normalise entre -1 et +1 :
        RSI = 100 → signal = -1  (surachat extrême, forte conviction de vente)
        RSI = 50  → signal =  0  (neutre)
        RSI = 0   → signal = +1  (survente extrême, forte conviction d'achat)

    Formule : signal = -(RSI - 50) / 50
    À RSI=70 : -(70-50)/50 = -0.4 (signal de vente modéré)
    À RSI=20 : -(20-50)/50 = +0.6 (signal d'achat modéré)

    Args:
        close      : Series des prix de clôture
        period     : période du RSI (défaut 14 jours)
        overbought : seuil de surachat (défaut 70)
        oversold   : seuil de survente (défaut 30)

    Returns:
        Series normalisée entre -1 et +1
    """
    # pandas_ta calcule le RSI en une ligne
    rsi = ta.rsi(close, length=period)

    # Normalisation centrée sur 50 (neutre)
    signal = -(rsi - 50) / 50

    # Zone neutre : si RSI entre 30 et 70, signal = 0
    signal = signal.where(rsi > overbought, 0)   # met 0 si RSI <= 70
    signal = signal.where(rsi < oversold, signal) # garde le signal si RSI < 30

    return signal.clip(-1, 1)


def macd_signal(close: pd.Series,
                fast: int = 12,
                slow: int = 26,
                signal_period: int = 9) -> pd.Series:
    """
    Signal basé sur le MACD (Moving Average Convergence Divergence).

    Le MACD mesure la différence entre deux moyennes exponentielles
    (EMA) de vitesses différentes.

    EMA (Exponential Moving Average) vs SMA :
        SMA : chaque jour a le même poids
        EMA : les jours récents ont un poids plus élevé
              → réagit plus vite aux changements de tendance

    Composantes du MACD :
        MACD line    = EMA(12) - EMA(26)   → tendance court terme vs long terme
        Signal line  = EMA(9) de la MACD   → lissage du MACD
        Histogramme  = MACD - Signal line  → force du signal

    Logique du signal :
        Histogramme > 0 → MACD au-dessus de sa ligne de signal
                       → momentum haussier → signal positif
        Histogramme < 0 → momentum baissier → signal négatif

    On normalise par la valeur maximale glissante sur 252 jours
    pour que le signal reste entre -1 et +1 quelle que soit
    l'échelle du prix.

    Args:
        close         : Series des prix de clôture
        fast          : période EMA rapide (défaut 12)
        slow          : période EMA lente (défaut 26)
        signal_period : période EMA du signal (défaut 9)

    Returns:
        Series normalisée entre -1 et +1
    """
    # pandas_ta retourne un DataFrame avec 3 colonnes
    macd_df = ta.macd(close, fast=fast, slow=slow, signal=signal_period)

    # On récupère l'histogramme (MACD - Signal line)
    # Le nom de colonne suit le format : MACDh_12_26_9
    hist_col = [c for c in macd_df.columns if c.startswith("MACDh")][0]
    histogram = macd_df[hist_col]

    # Normalisation par le max glissant sur 252 jours
    rolling_max = histogram.abs().rolling(252, min_periods=1).max()
    signal = (histogram / rolling_max).clip(-1, 1)

    return signal


def combine_signals(signals: dict,
                    weights: dict = None) -> pd.Series:
    """
    Combine plusieurs signaux en un signal composite.

    Pourquoi combiner ?
    Chaque signal a des forces et des faiblesses selon le régime
    de marché. En combinant des signaux décorrélés, on lisse
    les périodes où un signal individuel se trompe.

    Méthode : moyenne pondérée des signaux.
    Si aucun poids fourni, on utilise des poids égaux.

    Exemple avec 3 signaux :
        sma_signal  = +0.8  (fort signal haussier)
        rsi_signal  = -0.2  (léger signal baissier)
        macd_signal = +0.5  (signal haussier modéré)
        → combiné = (0.8 - 0.2 + 0.5) / 3 = +0.37

    Args:
        signals : dict {'nom': Series} des signaux à combiner
        weights : dict {'nom': float} des poids (somme = 1)
                  Si None, poids égaux

    Returns:
        Series normalisée entre -1 et +1
    """
    if weights is None:
        # Poids égaux : 1/N pour chaque signal
        weights = {name: 1 / len(signals) for name in signals}

    combined = sum(
        signals[name] * weights[name]
        for name in signals
    )

    return combined.clip(-1, 1)


def signal_to_position(signal: pd.Series,
                        threshold: float = 0.0) -> pd.Series:
    """
    Convertit un signal continu [-1, +1] en position binaire [0, 1].

    Un signal de -0.1 à +0.1 peut être du bruit — pas assez fort
    pour justifier une position. Le seuil (threshold) filtre ce bruit.

    Logique :
        signal > threshold  → position = 1 (on achète)
        signal <= threshold → position = 0 (on reste en cash)

    On n'implémente pas les positions short (vente à découvert)
    pour l'instant — ça viendra à la séance 9.

    Args:
        signal    : Series de signaux entre -1 et +1
        threshold : seuil minimum pour prendre position (défaut 0)

    Returns:
        Series avec valeurs 0 ou 1, décalée d'un jour (shift(1))
    """
    position = (signal > threshold).astype(int)
    # shift(1) appliqué ici pour garantir qu'on ne l'oublie jamais
    return position.shift(1).fillna(0)
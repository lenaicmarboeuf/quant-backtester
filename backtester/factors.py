"""
backtester/factors.py — Facteurs quantitatifs cross-sectionnels et time-series

Séance 8 : Combiner sans se tromper soi-même

Deux types de signaux coexistent dans ce projet :

  1. Signaux time-series (signals.py) :
     → Compare un actif à son propre passé.
     → Ex : SMA crossover sur AAPL seul.

  2. Facteurs cross-sectionnels (ce fichier) :
     → Compare plusieurs actifs ENTRE EUX à un instant T.
     → Ex : "AAPL a le meilleur momentum parmi ces 10 actifs."
     → On classe, on normalise, on choisit les meilleurs.

Pourquoi la distinction est importante ?
  Un signal time-series peut donner +1 à tous les actifs en même temps
  (tout monte). Un facteur cross-sectionnel est RELATIF : il y a toujours
  des gagnants et des perdants, la somme des scores est ≈ 0.

Références académiques :
  - Jegadeesh & Titman (1993) — Momentum
  - Ang, Hodrick, Xing, Zhang (2006) — Low volatility anomaly
  - Novy-Marx (2013) — Qualité / rentabilité
  - Grinold & Kahn (2000) — IC et Fundamental Law of Active Management
"""

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaire : normalisation cross-sectionnelle en z-scores
# ─────────────────────────────────────────────────────────────────────────────

def _cross_section_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise chaque ligne (= chaque date) en z-scores cross-sectionnels.

    Pour chaque date T :
        z_i = (signal_i - mean(signals)) / std(signals)

    Permet de comparer des actifs ayant des échelles différentes
    (ex : prix de 10 € vs prix de 1 000 €).

    Après normalisation :
        - moyenne = 0  (la moitié des actifs au-dessus, moitié en dessous)
        - std = 1      (un actif à +2 est vraiment exceptionnel)
        - clip(-3, 3)  : élimine les valeurs extrêmes (outliers)
    """
    z = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)
    return z.clip(-3, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Facteur Momentum
# ─────────────────────────────────────────────────────────────────────────────

def momentum_factor(
    close: pd.DataFrame,
    window: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """
    Facteur momentum cross-sectionnel (Jegadeesh & Titman, 1993).

    Idée : les actifs qui ont le mieux performé sur les 12 derniers mois
    (hors le dernier mois) continuent de surperformer à court terme.

    Pourquoi exclure le dernier mois (skip=21) ?
    Le dernier mois présente un effet de retournement à court terme
    (mean-reversion sur 1–4 semaines). En l'excluant, on capture
    le "vrai" momentum intermédiaire (1–12 mois).

    Formule :
        rendement_12_1 = close[T - skip] / close[T - window] - 1
        → rendement sur [T-252 ; T-21]

    Args:
        close  : DataFrame (index=dates, columns=tickers) des prix de clôture
        window : fenêtre totale de calcul (défaut 252 = 12 mois)
        skip   : jours récents à exclure (défaut 21 = 1 mois)

    Returns:
        DataFrame de z-scores cross-sectionnels (même index/colonnes que close)

    Exemple :
        close = pd.DataFrame({'AAPL': ..., 'MSFT': ..., 'GOOG': ...})
        mom = momentum_factor(close)
        # mom.iloc[-1] → score de momentum de chaque actif aujourd'hui
        # AAPL = +1.5 → momentum fort, GOOG = -0.8 → momentum faible
    """
    if isinstance(close, pd.Series):
        # Cas univarié : on wrappe en DataFrame pour cohérence
        close = close.to_frame()

    # Rendement de T-window à T-skip
    # pct_change(window-skip) : rendement sur (window-skip) jours
    # shift(skip)             : recule de 'skip' jours → exclut le mois récent
    raw_momentum = close.pct_change(window - skip).shift(skip)

    # Normalisation cross-sectionnelle
    return _cross_section_zscore(raw_momentum)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Facteur Low-Volatility
# ─────────────────────────────────────────────────────────────────────────────

def volatility_factor(
    close: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """
    Facteur low-volatility / low-beta (Black, 1972 ; Ang et al., 2006).

    Anomalie paradoxale : les actifs peu risqués (faible volatilité)
    surperforment les actifs risqués sur le long terme — ce qui contredit
    le CAPM classique (plus de risque = plus de rendement).

    Pourquoi cette anomalie existe ?
    - Contraintes institutionnelles : les gérants benchmarkés évitent
      les actifs peu volatils (trop ennuyeux), les sous-évaluant.
    - Biais comportemental : les investisseurs surpaient pour les actifs
      "excitants" (forte volatilité, potentiel de jackpot).

    Signal : -vol_réalisée
    → Signe négatif car on VEUT les actifs peu volatils.
    → Un actif avec vol = 15%/an a un meilleur score qu'un actif à 40%/an.

    Args:
        close  : DataFrame (index=dates, columns=tickers)
        window : fenêtre de calcul de la vol (défaut 63 = 3 mois)

    Returns:
        DataFrame de z-scores cross-sectionnels (même index/colonnes que close)
    """
    if isinstance(close, pd.Series):
        close = close.to_frame()

    # Vol journalière annualisée
    returns = close.pct_change()
    vol_realized = returns.rolling(window).std() * np.sqrt(252)

    # On inverse : faible vol → score positif
    raw_signal = -vol_realized

    return _cross_section_zscore(raw_signal)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Facteur Qualité (proxy prix)
# ─────────────────────────────────────────────────────────────────────────────

def quality_factor(
    close: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """
    Facteur qualité — proxy basé sur le Sharpe ratio rolling.

    En théorie, la qualité se mesure via les fondamentaux :
        ROE (Return on Equity), marge brute, endettement...
    Mais ces données nécessitent une API financière payante.

    Proxy disponible avec les prix seuls :
        Sharpe rolling = mean(returns) / std(returns) × √252

    Rationale : une entreprise de "qualité" tend à avoir des rendements
    réguliers et positifs — ce que capture le Sharpe ratio.
    C'est imparfait mais corrélé avec les facteurs de qualité académiques.

    Args:
        close  : DataFrame (index=dates, columns=tickers)
        window : fenêtre rolling pour le Sharpe (défaut 252 = 1 an)

    Returns:
        DataFrame de z-scores cross-sectionnels
    """
    if isinstance(close, pd.Series):
        close = close.to_frame()

    returns = close.pct_change()

    # Sharpe rolling
    rolling_mean = returns.rolling(window).mean()
    rolling_std  = returns.rolling(window).std().replace(0, np.nan)
    sharpe_rolling = (rolling_mean / rolling_std) * np.sqrt(window)

    return _cross_section_zscore(sharpe_rolling)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Information Coefficient (IC)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ic(
    factor: pd.Series,
    forward_returns: pd.Series,
    method: str = "spearman",
) -> pd.Series:
    """
    Information Coefficient (IC) — mesure de qualité d'un signal.

    L'IC mesure la corrélation entre le signal au jour J et
    le rendement RÉALISÉ au jour J+1.

    IC = Spearman(signal[J], returns[J+1])

    Pourquoi Spearman plutôt que Pearson ?
    - Spearman est basé sur les rangs → robuste aux outliers.
    - Pearson suppose une relation linéaire → trop sensible aux extrêmes.
    - En finance, les distributions sont "fat-tailed" → Spearman préféré.

    Interprétation :
        IC > 0.05  : signal avec de la valeur prédictive
        IC > 0.10  : signal fort
        IC < 0.0   : signal inversement corrélé (ou bruit)

    ⚠️  Attention au look-ahead bias :
        forward_returns doit être close.pct_change().shift(-1)
        C'est le SEUL cas où on utilise shift(-1) — pour mesurer
        la qualité du signal a posteriori, PAS pour trader.

    Args:
        factor          : signal au jour J (pd.Series)
        forward_returns : rendement au jour J+1 (= close.pct_change().shift(-1))
        method          : 'spearman' (défaut) ou 'pearson'

    Returns:
        pd.Series des IC journaliers (un float par date)

    Exemple :
        close = download_data('AAPL', ...)['Close']
        signal = sma_crossover(close)
        fwd = close.pct_change().shift(-1)  # ← seul shift(-1) autorisé
        ic_series = compute_ic(signal, fwd)
        print(f"IC moyen : {ic_series.mean():.4f}")
    """
    # Alignement des deux séries sur les mêmes dates
    combined = pd.DataFrame({"factor": factor, "fwd": forward_returns}).dropna()

    if len(combined) < 10:
        return pd.Series(dtype=float)

    if method == "spearman":
        # Corrélation de Spearman point par point sur une fenêtre
        # → on calcule l'IC comme la corrélation globale (standard)
        corr, _ = stats.spearmanr(combined["factor"], combined["fwd"])
    else:
        corr = combined["factor"].corr(combined["fwd"])

    # Retourne un scalaire pour compatibilité avec compute_rolling_ic
    return float(corr)


def compute_rolling_ic(
    factor: pd.Series,
    forward_returns: pd.Series,
    window: int = 63,
    method: str = "spearman",
) -> pd.Series:
    """
    IC rolling — évolution de la qualité du signal dans le temps.

    Calcule l'IC sur une fenêtre glissante de 'window' jours.
    Permet de détecter si le signal est plus fort en certaines périodes
    (ex : momentum fonctionne mieux en tendance, moins en range).

    Args:
        factor          : signal au jour J
        forward_returns : rendement au jour J+1
        window          : fenêtre rolling (défaut 63 = 3 mois)
        method          : 'spearman' ou 'pearson'

    Returns:
        pd.Series des IC rolling (index = dates)
    """
    combined = pd.DataFrame({"factor": factor, "fwd": forward_returns}).dropna()

    if method == "spearman":
        # Spearman sur fenêtre glissante : boucle numpy (simple et fiable)
        f = combined["factor"].to_numpy()
        r = combined["fwd"].to_numpy()
        ic_values = np.full(len(combined), np.nan)
        for i in range(window - 1, len(combined)):
            corr, _ = stats.spearmanr(f[i - window + 1:i + 1],
                                      r[i - window + 1:i + 1])
            ic_values[i] = corr  # NaN si la fenêtre est constante
        return pd.Series(ic_values, index=combined.index)

    else:
        return combined["factor"].rolling(window).corr(combined["fwd"])


# ─────────────────────────────────────────────────────────────────────────────
# 5. ICIR — Information Coefficient Information Ratio
# ─────────────────────────────────────────────────────────────────────────────

def compute_icir(
    ic_series: pd.Series,
    window: int = 63,
) -> pd.Series:
    """
    ICIR = IC.rolling(window).mean() / IC.rolling(window).std()

    L'ICIR mesure la RÉGULARITÉ du signal, pas juste sa force moyenne.

    Analogie : c'est le Sharpe ratio du signal.
        IC moyen = rendement moyen du signal
        IC std   = volatilité du signal (à quel point il est erratique)

    Exemple :
        Signal A : IC = [0.08, 0.07, 0.09, 0.08] → ICIR élevé (régulier)
        Signal B : IC = [0.20, -0.05, 0.15, -0.10] → ICIR faible (erratique)
        Signal A est préférable malgré un IC moyen plus faible.

    Interprétation :
        ICIR > 0.5 : signal exploitable, mérite d'être dans le composite
        ICIR > 1.0 : signal excellent
        ICIR < 0.3 : signal peu fiable

    Loi fondamentale du management actif (Grinold & Kahn) :
        Sharpe ≈ ICIR × √(N_paris)
        → Un ICIR de 0.5 sur 252 paris/an → Sharpe ≈ 0.5 × √252 ≈ 7.9
        (théorique, avant coûts et contraintes)

    Args:
        ic_series : Series des IC journaliers (sortie de compute_rolling_ic)
        window    : fenêtre rolling (défaut 63 = 3 mois)

    Returns:
        pd.Series des ICIR rolling (même index que ic_series)
    """
    ic_mean = ic_series.rolling(window).mean()
    ic_std  = ic_series.rolling(window).std().replace(0, np.nan)
    return ic_mean / ic_std


# ─────────────────────────────────────────────────────────────────────────────
# 6. Régression Fama-French (alpha isolé)
# ─────────────────────────────────────────────────────────────────────────────

def fama_french_alpha(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    risk_free: float = 0.0,
) -> dict:
    """
    Régression CAPM (1 facteur) pour isoler l'alpha de la stratégie.

    Modèle :
        R_strat - Rf = alpha + beta × (R_mkt - Rf) + epsilon

    Le test de significativité porte sur ALPHA (l'intercept) :
        t_alpha = alpha / se(alpha)
        se(alpha) = sqrt( sigma² × (1/n + x_moy² / Sxx) )
    La p-value est bilatérale, loi de Student à n-2 degrés de liberté.

    Si alpha > 0 et p < 0.05 : valeur ajoutée au-delà de l'exposition au marché.
    Si alpha ≈ 0             : la stratégie ne fait que capter le marché (beta).

    Note : CAPM seulement. Le vrai Fama-French (SMB, HML...) demande les
    facteurs de Kenneth French et statsmodels (voir séance suivante).

    Args:
        strategy_returns : rendements journaliers de la stratégie
        market_returns   : rendements du benchmark (ex : SPY)
        risk_free        : taux sans risque journalier (défaut 0)

    Returns:
        dict avec alpha, beta, R², p-value et t-stat de l'alpha
    """
    combined = pd.DataFrame({
        "strat": strategy_returns - risk_free,
        "mkt":   market_returns - risk_free,
    }).dropna()

    n = len(combined)
    if n < 30:
        return {"error": "Pas assez de données (minimum 30 observations)"}

    y = combined["strat"].to_numpy(dtype=float)
    x = combined["mkt"].to_numpy(dtype=float)

    x_mean, y_mean = x.mean(), y.mean()
    sxx = ((x - x_mean) ** 2).sum()
    if sxx < 1e-18:
        return {"error": "Variance du marché nulle"}

    beta      = ((x - x_mean) * (y - y_mean)).sum() / sxx
    alpha     = y_mean - beta * x_mean
    residuals = y - (alpha + beta * x)
    ss_res    = (residuals ** 2).sum()
    ss_tot    = ((y - y_mean) ** 2).sum()
    r2        = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    sigma2   = ss_res / (n - 2)
    se_alpha = np.sqrt(sigma2 * (1 / n + x_mean ** 2 / sxx))

    if se_alpha < 1e-15:
        t_stat = 0.0 if abs(alpha) < 1e-15 else np.inf
    else:
        t_stat = alpha / se_alpha
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2)))

    return {
        "alpha (annualisé %)":  round(alpha * 252 * 100, 2),
        "beta":                 round(float(beta), 3),
        "R²":                   round(float(r2), 4),
        "p-value alpha":        round(p_value, 4),
        "t-stat alpha":         round(float(t_stat), 3),
        "alpha significatif":   p_value < 0.05,
        "nb observations":      n,
    }
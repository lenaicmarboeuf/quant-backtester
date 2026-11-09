"""
backtester/pairs.py — Pair trading et arbitrage statistique

Séance 9 : "Gagner dans toutes les directions"

Le principe
-----------
Deux actifs Y et X (ex : Coca-Cola et Pepsi) sont économiquement liés. Leurs
prix peuvent chacun dériver, mais si une combinaison linéaire reste stable :

        spread_t = log(Y_t) - β · log(X_t) - α

alors le spread oscille autour de sa moyenne (processus mean-reverting).
Quand le spread est anormalement HAUT  → Y est "trop cher" vs X
    → on SHORT Y et on ACHÈTE β de X  (position = -1)
Quand il est anormalement BAS          → Y est "trop bon marché" vs X
    → on ACHÈTE Y et on SHORT β de X  (position = +1)
On sort quand le spread revient vers sa moyenne.

CORRÉLATION ≠ COINTÉGRATION
---------------------------
Deux actifs peuvent être très corrélés (leurs rendements montent et baissent
ensemble) sans être cointégrés : leur écart peut diverger indéfiniment
(deux marches aléatoires qui montent). Seule la cointégration garantit
(statistiquement) que l'écart REVIENT. On la TESTE (Engle-Granger), on ne la
suppose jamais. Et même après un test réussi, la relation peut se casser
(fusion, changement de modèle économique, régime) : d'où le stop de cointégration.

Convention anti look-ahead
--------------------------
hedge ratio, moyenne et écart-type du spread sont tous estimés sur des
fenêtres GLISSANTES ≤ T. La position décidée à la clôture de T est détenue
pendant T+1 (shift(1) appliqué dans pair_returns).
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. Hedge ratio, spread, z-score
# ─────────────────────────────────────────────────────────────────────────────

def hedge_ratio(y: pd.Series, x: pd.Series) -> Tuple[float, float]:
    """
    Ratio de couverture statique par MCO : y = α + β·x + ε.

    β = Cov(x, y) / Var(x)   — "pour 1 € de variation de X, Y bouge de β €".
    α = ȳ - β·x̄

    On travaille en général sur les LOG-prix (le spread est alors un écart de
    rendements cumulés, comparable entre actifs de prix différents).

    ⚠ Une estimation sur TOUT l'échantillon utilise le futur : OK pour
    analyser/visualiser une paire, PAS pour backtester. Pour le backtest,
    utiliser rolling_hedge_ratio.

    Retourne (beta, alpha).
    """
    d = pd.concat([y, x], axis=1).dropna()
    yy, xx = d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float)
    var_x = xx.var(ddof=1)
    if len(d) < 3 or var_x < 1e-18:
        return float("nan"), float("nan")
    beta = np.cov(xx, yy, ddof=1)[0, 1] / var_x
    alpha = yy.mean() - beta * xx.mean()
    return float(beta), float(alpha)


def rolling_hedge_ratio(y: pd.Series, x: pd.Series, window: int = 120) -> pd.DataFrame:
    """
    Hedge ratio MCO sur fenêtre glissante (sans look-ahead).

    Retourne un DataFrame ['beta', 'alpha'] indexé par date, avec
        beta_t  = Cov(x,y)/Var(x) sur [t-window+1, t]
        alpha_t = moyenne(y) - beta_t × moyenne(x) sur la même fenêtre.
    """
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var().replace(0, np.nan)
    beta = cov / var
    alpha = y.rolling(window).mean() - beta * x.rolling(window).mean()
    return pd.DataFrame({"beta": beta, "alpha": alpha})


def spread(
    y: pd.Series,
    x: pd.Series,
    beta: Optional[object] = None,
    alpha: object = 0.0,
) -> pd.Series:
    """
    Spread = y - β·x - α.

    beta peut être un float (hedge ratio fixe) ou une Series (rolling).
    Si beta est None : hedge ratio statique estimé sur tout l'échantillon
    (analyse seulement, voir hedge_ratio).
    """
    if beta is None:
        beta, alpha = hedge_ratio(y, x)
    return y - beta * x - alpha


def spread_zscore(s: pd.Series, window: int = 60) -> pd.Series:
    """
    Z-score glissant du spread : (s_t - moyenne_window) / écart-type_window.

        z = 0   → spread à sa moyenne récente
        z = +2  → spread exceptionnellement haut (2 σ) → Y trop cher
        z = -2  → spread exceptionnellement bas        → Y trop bon marché

    Choix de la fenêtre : idéalement quelques fois la half-life. Trop courte →
    le z-score "suit" le spread et ne déclenche presque jamais ; trop longue →
    il réagit trop tard aux changements de niveau d'équilibre.
    """
    mean = s.rolling(window).mean()
    std = s.rolling(window).std().replace(0, np.nan)
    return (s - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# 2. Half-life (Ornstein-Uhlenbeck) et test de cointégration
# ─────────────────────────────────────────────────────────────────────────────

def half_life(s: pd.Series) -> float:
    """
    Half-life de retour à la moyenne d'un spread (modèle d'Ornstein-Uhlenbeck).

    Modèle continu :  ds = -θ (s - μ) dt + σ dW
    Version discrète (AR(1)) : s_t = c + φ s_{t-1} + ε_t   avec φ = e^{-θ}

    On estime par MCO la régression  Δs_t = a + b·s_{t-1} + ε  (b = φ - 1) :
        φ = 1 + b ;   θ = -ln(φ) ;   half_life = ln(2) / θ  (en jours)

    Interprétation : temps moyen pour que l'écart à la moyenne soit divisé par 2.
        Exemple : b = -0.05 → φ = 0.95 → half-life = ln2 / (-ln 0.95) ≈ 13.5 jours.
        Rendez-vous 1/2 jour → trop rapide pour du daily ; > 100 jours → trop lent.

    Retourne float('inf') si b ≥ 0 (pas de retour à la moyenne : marche
    aléatoire ou tendance) — signal que la paire n'est PAS exploitable.
    """
    s = s.dropna()
    if len(s) < 10:
        return float("inf")
    lag = s.shift(1).iloc[1:].to_numpy(float)
    delta = s.diff().iloc[1:].to_numpy(float)
    X = np.column_stack([np.ones(len(lag)), lag])
    coef, *_ = np.linalg.lstsq(X, delta, rcond=None)
    b = coef[1]
    phi = 1 + b
    # b ≈ 0 (à l'erreur numérique près) = pas de retour à la moyenne : marche
    # aléatoire ou tendance. Sans ce seuil, b = -1e-16 donnerait θ ≈ 0 → -inf/NaN.
    if b >= -1e-8 or phi <= 0:
        return float("inf")
    return float(np.log(2) / -np.log(phi))


def cointegration_test(y: pd.Series, x: pd.Series, pvalue_max: float = 0.05) -> Dict:
    """
    Test de cointégration d'Engle-Granger (statsmodels.tsa.stattools.coint).

    Procédure :
        1. Régression MCO  y = α + β·x + ε
        2. Test ADF (Dickey-Fuller augmenté) sur les résidus ε :
           H0 : les résidus ont une racine unitaire (PAS de cointégration)
           H1 : les résidus sont stationnaires (cointégration)
        3. p-value < 0.05 → on rejette H0 → paire cointégrée (au seuil de 5 %).

    ⚠ Pièges :
        - Les valeurs critiques ne sont PAS celles d'un ADF classique (les
          résidus sont estimés) : on utilise coint() qui les corrige.
        - Tester beaucoup de paires et garder les "significatives" = data
          snooping : sur 100 paires indépendantes, ~5 passent le test par pur
          hasard à 5 %. Il faut corriger (Bonferroni, FDR) ou valider hors échantillon.
        - Le test est asymétrique : coint(y, x) ≠ coint(x, y) en pratique.
        - On l'applique sur des LOG-prix, pas des rendements.

    Retourne dict : t_stat, pvalue, crit_5pct, cointegrated (bool), half_life,
    beta (hedge ratio statique), n.
    """
    from statsmodels.tsa.stattools import coint

    d = pd.concat([y, x], axis=1).dropna()
    if len(d) < 30:
        return {"error": "Pas assez de données (minimum 30 observations)"}
    t_stat, pvalue, crit = coint(d.iloc[:, 0], d.iloc[:, 1])
    beta, alpha = hedge_ratio(d.iloc[:, 0], d.iloc[:, 1])
    s = spread(d.iloc[:, 0], d.iloc[:, 1], beta, alpha)
    return {
        "t_stat":       round(float(t_stat), 3),
        "pvalue":       round(float(pvalue), 4),
        "crit_5pct":    round(float(crit[1]), 3),
        "cointegrated": bool(pvalue < pvalue_max),
        "half_life":    round(half_life(s), 1),
        "beta":         round(beta, 4),
        "n":            len(d),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Signal de trading
# ─────────────────────────────────────────────────────────────────────────────

def pairs_signal(
    y: pd.Series,
    x: pd.Series,
    hedge_window: int = 120,
    z_window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    stop_z: Optional[float] = 4.0,
    coint_window: Optional[int] = None,
    coint_every: int = 21,
    coint_pvalue: float = 0.10,
) -> pd.DataFrame:
    """
    Machine à états d'un pair trading z-score (positions décidées à la clôture de T).

    Entrées : y, x = LOG-prix des deux actifs.

    États : position ∈ {-1, 0, +1}
        +1 = long le spread  (long Y, short β·X)  — ouvert quand z < -entry_z
        -1 = short le spread (short Y, long β·X)  — ouvert quand z > +entry_z
         0 = flat

    Règles :
        ENTRÉE  : |z| > entry_z  (défaut ±2σ)
        SORTIE  : le z-score revient à exit_z (défaut 0 = la moyenne)
                  long  spread : on sort quand z >= exit_z
                  short spread : on sort quand z <= -exit_z
        STOP    : |z| > stop_z (défaut 4σ) → on coupe : le spread diverge au
                  lieu de revenir, c'est peut-être une rupture de relation.
                  On ne rouvre pas tant que |z| n'est pas repassé sous entry_z.
        STOP DE COINTÉGRATION (optionnel, coint_window ≠ None) :
                  tous les `coint_every` jours, on re-teste la cointégration
                  sur les `coint_window` derniers jours (≤ T). Si p-value >
                  coint_pvalue → relation jugée cassée → on passe/reste flat
                  jusqu'au prochain test réussi.

    Paramètres temporels : le hedge ratio utilise `hedge_window` jours, le
    z-score `z_window` jours. Les premières lignes (fenêtres incomplètes) sont flat.

    Retourne DataFrame ['beta', 'spread', 'z', 'position'] (position NON décalée :
    c'est pair_returns qui applique shift(1)).
    """
    hr = rolling_hedge_ratio(y, x, hedge_window)
    s = spread(y, x, hr["beta"], hr["alpha"])
    z = spread_zscore(s, z_window)

    # Test de cointégration périodique (autorisation de trader)
    allowed = pd.Series(True, index=y.index)
    if coint_window is not None:
        from statsmodels.tsa.stattools import coint
        ok = True
        flags = []
        for i in range(len(y)):
            if i >= coint_window and (i - coint_window) % coint_every == 0:
                yy = y.iloc[i - coint_window + 1:i + 1]
                xx = x.iloc[i - coint_window + 1:i + 1]
                d = pd.concat([yy, xx], axis=1).dropna()
                if len(d) >= 30:
                    try:
                        ok = bool(coint(d.iloc[:, 0], d.iloc[:, 1])[1] < coint_pvalue)
                    except Exception:
                        ok = False
            elif i < coint_window:
                ok = False  # pas encore de test possible → on ne trade pas
            flags.append(ok)
        allowed = pd.Series(flags, index=y.index)

    pos = np.zeros(len(y))
    state = 0
    z_vals = z.to_numpy()
    allow = allowed.to_numpy()
    blocked_after_stop = False

    for i in range(len(y)):
        zi = z_vals[i]
        if np.isnan(zi) or not allow[i]:
            state = 0
            pos[i] = 0
            continue

        if blocked_after_stop and abs(zi) < entry_z:
            blocked_after_stop = False

        if state == 0:
            if not blocked_after_stop:
                if zi > entry_z and (stop_z is None or zi <= stop_z):
                    state = -1
                elif zi < -entry_z and (stop_z is None or zi >= -stop_z):
                    state = 1
        elif state == 1:
            if zi >= exit_z:
                state = 0
            elif stop_z is not None and zi < -stop_z:
                state = 0
                blocked_after_stop = True
        elif state == -1:
            if zi <= -exit_z:
                state = 0
            elif stop_z is not None and zi > stop_z:
                state = 0
                blocked_after_stop = True
        pos[i] = state

    return pd.DataFrame({
        "beta":     hr["beta"],
        "spread":   s,
        "z":        z,
        "position": pd.Series(pos, index=y.index),
    })


# ─────────────────────────────────────────────────────────────────────────────
# 4. P&L d'une paire
# ─────────────────────────────────────────────────────────────────────────────

def pair_returns(
    close_y: pd.Series,
    close_x: pd.Series,
    position: pd.Series,
    beta: pd.Series,
    cost_bps: float = 8.5,
    borrow_bps: float = 30.0,
) -> pd.DataFrame:
    """
    Rendements d'une paire, en fraction du capital BRUT engagé.

    Pour une position p ∈ {-1,0,+1} et un hedge ratio β (décidés à la clôture
    de T, appliqués pendant T+1) :
        poids Y :  p      / (1 + |β|)
        poids X : -p·β    / (1 + |β|)       (somme des |poids| = 1)
        rendement brut = w_Y · r_Y + w_X · r_X

    Normaliser par (1+|β|) fixe l'exposition brute à 100 % du capital alloué
    à la paire, pour des comparaisons équitables.

    Coûts (le piège : DEUX jambes) :
        trading = Σ_jambes |Δpoids| × cost_bps / 10 000
                  (cost_bps par défaut 8.5 = commission 5 + spread/2 1.25 + slippage 2,
                  cohérent avec la séance 7)
        emprunt = Σ_jambes short  |poids| × borrow_bps / 10 000 / 252
    Une entrée + une sortie = 4 ordres (2 jambes × 2), donc ~4 × cost_bps.

    Colonnes retournées :
        gross_returns, trading_costs, borrow_costs, strategy_returns,
        w_y, w_x (poids détenus), gross_exposure.
    """
    ry = close_y.pct_change().fillna(0.0)
    rx = close_x.pct_change().fillna(0.0)

    p = position.shift(1).fillna(0.0)
    b = beta.shift(1).fillna(0.0)          # beta connu à la clôture précédente
    norm = 1 + b.abs()
    w_y = p / norm
    w_x = -p * b / norm

    gross = w_y * ry + w_x * rx

    turn = w_y.diff().abs().fillna(0.0) + w_x.diff().abs().fillna(0.0)
    trading = -(turn * cost_bps / 10_000)

    short_exp = (-w_y).clip(lower=0) + (-w_x).clip(lower=0)
    borrow = -(short_exp * borrow_bps / 10_000 / 252)

    out = pd.DataFrame({
        "gross_returns":    gross,
        "trading_costs":    trading,
        "borrow_costs":     borrow,
        "strategy_returns": gross + trading + borrow,
        "w_y":              w_y,
        "w_x":              w_x,
        "gross_exposure":   w_y.abs() + w_x.abs(),
    })
    return out

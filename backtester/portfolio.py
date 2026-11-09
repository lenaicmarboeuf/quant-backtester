"""
backtester/portfolio.py — Portefeuilles long/short, neutralité dollar et beta

Séance 9 : "Gagner dans toutes les directions"

Idée centrale
-------------
Un facteur cross-sectionnel (séance 8) donne à chaque date un SCORE par actif.
On en fait un portefeuille :
    - LONG  les actifs aux meilleurs scores (quantile du haut)
    - SHORT les actifs aux pires scores     (quantile du bas)
Si les deux jambes ont la même taille, le portefeuille est DOLLAR-NEUTRAL :
    Σ w_i = 0   (net = 0)    et    Σ |w_i| = 2   (brut = 200 %)
Quand le marché monte de 10 %, la jambe longue gagne ~10 % et la jambe short
perd ~10 % : le marché s'annule, il reste la DIFFÉRENCE de performance entre
gagnants et perdants (le "spread" du facteur).

Attention : dollar-neutral ≠ beta-neutral.
Si la jambe longue est faite d'actions à beta 1.4 et la jambe short d'actions
à beta 0.7, le portefeuille est long de marché même avec net = 0 : il perdra
quand le marché chute. D'où rolling_beta / beta_hedge / beta_neutral_weights.

Convention anti look-ahead
--------------------------
Toutes les fonctions qui fabriquent des POIDS (quantile_weights, dollar_neutral,
beta_neutral_weights...) travaillent avec les données ≤ T et renvoient des poids
"décidés à la clôture de T". C'est portfolio_returns() (ou PortfolioBacktester)
qui applique shift(1) UNE SEULE FOIS pour obtenir les poids détenus pendant T+1.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. Poids par quantiles
# ─────────────────────────────────────────────────────────────────────────────

def quantile_weights(
    scores: pd.DataFrame,
    top: float = 0.2,
    bottom: float = 0.2,
    long_only: bool = False,
    rebalance_every: int = 1,
    gross: float = 2.0,
) -> pd.DataFrame:
    """
    Construit un portefeuille long/short à pondération égale par quantiles.

    À chaque date de rebalancement :
        1. On classe les actifs par score (rang 1 = pire, N = meilleur).
        2. LONG  : les k_long  meilleurs  → poids +(gross/2) / k_long  chacun
        3. SHORT : les k_short moins bons → poids -(gross/2) / k_short chacun
        4. Le reste → poids 0.
    avec k_long = round(top × N), k_short = round(bottom × N), au moins 1.

    Exemple chiffré (N = 20 actifs, top = bottom = 20 %, gross = 2) :
        k_long = k_short = 4
        chaque long  : +1/4 = +0.25    (somme jambe longue = +1)
        chaque short : -1/4 = -0.25    (somme jambe courte = -1)
        net = 0, brut = 2  → dollar-neutral, 100 % long + 100 % short du capital.

    Paramètres
    ----------
    scores          : DataFrame (dates × actifs) — plus haut = plus attractif
                      (ex : momentum_factor(close))
    top, bottom     : fraction d'actifs en long / en short (défaut 20 % / 20 %)
    long_only       : True = on ne garde que la jambe longue (somme = gross/2 = 1 par défaut)
                      → sert de benchmark "long-only" pour comparer
    rebalance_every : rebalancer tous les k jours (21 ≈ mensuel). Entre deux
                      rebalancements les poids sont maintenus (ffill).
    gross           : exposition brute cible (2.0 = 100 % long + 100 % short)

    Retourne
    --------
    DataFrame (dates × actifs) de poids. Dates sans assez de scores valides
    (< 2 actifs) → poids nuls.

    Limites :
        - Pondération égale : ne tient pas compte de la volatilité de chaque
          titre (une action très volatile contribue plus au risque).
        - Les égalités de scores sont départagées par l'ordre des colonnes
          (rank method='first') : sans conséquence sur des scores continus.
    """
    if rebalance_every < 1:
        raise ValueError("rebalance_every doit être >= 1")

    n_valid = scores.notna().sum(axis=1)
    ranks   = scores.rank(axis=1, method="first", ascending=True)  # 1 = pire

    k_long  = np.maximum(1, np.round(n_valid * top)).astype(int)
    k_short = np.maximum(1, np.round(n_valid * bottom)).astype(int)

    # Masque long : rang > N - k_long ; masque short : rang <= k_short
    long_mask  = ranks.gt(n_valid - k_long, axis=0) & scores.notna()
    short_mask = ranks.le(k_short, axis=0) & scores.notna()
    # Un actif ne peut pas être des deux côtés (si top+bottom > 1)
    short_mask = short_mask & ~long_mask

    leg = gross / 2   # taille de chaque jambe (1.0 par défaut)
    w_long  = long_mask.astype(float).div(k_long, axis=0) * leg
    w_short = -short_mask.astype(float).div(k_short, axis=0) * leg

    weights = w_long if long_only else (w_long + w_short)

    # Pas assez d'actifs → pas de position
    weights[n_valid < 2] = 0.0

    if rebalance_every > 1:
        rebal_dates = weights.index[::rebalance_every]
        weights = weights.loc[rebal_dates].reindex(weights.index).ffill()

    return weights.fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Neutralité dollar
# ─────────────────────────────────────────────────────────────────────────────

def dollar_neutral(weights: pd.DataFrame, gross: float = 2.0) -> pd.DataFrame:
    """
    Force Σ w_i = 0 à chaque date, puis ramène l'exposition brute à `gross`.

    Méthode :
        1. On retire la moyenne des poids ACTIFS (non nuls) de chaque ligne :
              w'_i = w_i - mean(w_actifs)      pour les i actifs
           → la somme devient exactement 0.
        2. On rééchelonne : w'' = w' × gross / Σ|w'|.

    Utile si on part de poids non équilibrés (ex : scores bruts, ou
    quantile_weights avec top ≠ bottom). Si les poids sont déjà neutres,
    la fonction ne change rien (idempotente).

    Exemple : w = [+0.5, +0.3, -0.2, 0]  (net = +0.6)
        actifs = 3, moyenne = 0.2  → [+0.3, +0.1, -0.4, 0]  (net = 0)
        brut = 0.8 → on multiplie par 2/0.8 = 2.5 → [+0.75, +0.25, -1.0, 0]

    Limite : dollar-neutral n'implique PAS beta-neutral (voir beta_neutral_weights).
    """
    active = (weights != 0)
    n_active = active.sum(axis=1).replace(0, np.nan)
    mean_active = weights.where(active).sum(axis=1) / n_active
    w = weights.sub(mean_active.fillna(0), axis=0).where(active, 0.0)

    gross_now = w.abs().sum(axis=1).replace(0, np.nan)
    w = w.mul(gross / gross_now, axis=0)
    return w.fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Beta glissant et couverture
# ─────────────────────────────────────────────────────────────────────────────

def rolling_beta(
    returns: Union[pd.Series, pd.DataFrame],
    market: pd.Series,
    window: int = 63,
    min_periods: Optional[int] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Beta glissant vs le marché (ex : SPY).

        beta_t = Cov(r_i, r_m) / Var(r_m)    sur les `window` derniers jours

    Interprétation :
        beta = 1.0 → l'actif bouge comme le marché
        beta = 1.5 → il amplifie de 50 % (marché -10 % → actif -15 %)
        beta = 0.5 → il amortit
        beta < 0   → il monte quand le marché baisse (rare, ex : or, VIX)

    Anti look-ahead : le beta à la date T n'utilise que les rendements ≤ T.
    (Les poids construits avec ce beta à T s'appliquent ensuite à T+1.)

    Limites : le beta est ESTIMÉ avec du bruit (63 jours = peu de données) et
    instable dans le temps (il grimpe en crise). Fenêtre courte = réactif mais
    bruité ; fenêtre longue = stable mais en retard.

    Paramètres
    ----------
    returns : Series (un actif) ou DataFrame (dates × actifs)
    market  : Series des rendements du marché
    window  : fenêtre en jours (défaut 63 ≈ 3 mois)
    """
    mp = min_periods or max(20, window // 2)
    m = market.reindex(returns.index)
    var_m = m.rolling(window, min_periods=mp).var()

    if isinstance(returns, pd.Series):
        cov = returns.rolling(window, min_periods=mp).cov(m)
        return cov / var_m.replace(0, np.nan)

    cov = returns.rolling(window, min_periods=mp).cov(m)  # DataFrame vs Series
    return cov.div(var_m.replace(0, np.nan), axis=0)


def portfolio_beta(weights: pd.DataFrame, betas: pd.DataFrame) -> pd.Series:
    """
    Beta du portefeuille à chaque date : β_p = Σ_i w_i × β_i.

    Exemple : long 0.5 d'un actif à beta 1.4, short 0.5 d'un actif à beta 0.7
        β_p = 0.5×1.4 - 0.5×0.7 = +0.35  → portefeuille encore long de marché !
    """
    b = betas.reindex(index=weights.index, columns=weights.columns)
    return (weights * b).sum(axis=1, min_count=1)


def beta_hedge(weights: pd.DataFrame, betas: pd.DataFrame) -> pd.Series:
    """
    Poids de la couverture de marché (ex : position sur SPY) qui annule le beta.

        h_t = -β_p,t = -Σ_i w_i,t × β_i,t

    Si β_p = +0.35, on SHORT 0.35 de SPY : le rendement couvert est
        r_couvert = r_portefeuille + h × r_marché.

    Avantage : simple, peu de turnover sur le portefeuille d'actions.
    Limite : le hedge ratio est lui-même estimé avec erreur → il reste un
    beta résiduel ; et couvrir coûte (le short de SPY paie aussi des frais).

    Retourne la Series h (poids sur le marché, décidé à la clôture de T ;
    à décaler d'un jour comme les autres poids → utilisé par hedged_returns).
    """
    return -portfolio_beta(weights, betas).fillna(0.0)


def beta_neutral_weights(
    weights: pd.DataFrame,
    betas: pd.DataFrame,
    max_ratio: float = 5.0,
) -> pd.DataFrame:
    """
    Rééchelonne la jambe longue et la jambe courte pour que β_p = 0,
    en gardant l'exposition brute inchangée.

    Notation (par date) :
        gL = Σ poids longs, gS = Σ |poids courts|
        βL = Σ_{longs} w_i β_i  (> 0 typiquement)
        βS = Σ_{courts} w_i β_i (< 0 typiquement)
        On cherche a (jambe longue) et b (jambe courte) tels que
            a·βL + b·βS = 0         (beta nul)
            a·gL + b·gS = G         (brut conservé, G = gL + gS d'origine)
        Avec r = -βL/βS :  b = a·r  et  a = G / (gL + r·gS).

    Conséquence : le portefeuille n'est généralement plus dollar-neutral
    (net ≠ 0). Beta-neutral et dollar-neutral sont deux contraintes
    DIFFÉRENTES ; on ne peut pas toujours avoir les deux avec 2 paramètres.

    Si r n'est pas fini, négatif, ou hors de [1/max_ratio, max_ratio],
    on garde les poids d'origine à cette date (sécurité contre un beta mal estimé).
    """
    b = betas.reindex(index=weights.index, columns=weights.columns)
    lng = weights.clip(lower=0)
    sht = weights.clip(upper=0)

    gL = lng.sum(axis=1)
    gS = -sht.sum(axis=1)
    bL = (lng * b).sum(axis=1, min_count=1)
    bS = (sht * b).sum(axis=1, min_count=1)

    r = -bL / bS.replace(0, np.nan)
    valid = np.isfinite(r) & (r > 1 / max_ratio) & (r < max_ratio) & (gL > 0) & (gS > 0)

    G = gL + gS
    a = G / (gL + r * gS)
    a = a.where(valid, 1.0)
    bb = (a * r).where(valid, 1.0)

    out = lng.mul(a, axis=0) + sht.mul(bb, axis=0)
    return out.fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Rendements du portefeuille & diagnostics d'exposition
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    shift: bool = True,
) -> pd.Series:
    """
    Rendements bruts (sans coûts) du portefeuille.

        r_p,t = Σ_i w_i,t-1 × r_i,t

    shift=True (défaut) : les poids décidés à la clôture de T sont appliqués
    aux rendements de T+1 → AUCUN look-ahead. Ne mettre shift=False que si
    les poids ont DÉJÀ été décalés (sinon on triche).

    Un rendement manquant (NaN) compte pour 0 (l'actif n'était pas tradable).
    """
    w = weights.reindex(returns.index).fillna(0.0)
    w = w.shift(1).fillna(0.0) if shift else w
    return (w * returns.fillna(0.0)).sum(axis=1)


def hedged_returns(
    port_returns: pd.Series,
    hedge_weight: pd.Series,
    market_returns: pd.Series,
    shift: bool = True,
) -> pd.Series:
    """
    Rendement du portefeuille + couverture de marché :
        r_couvert,t = r_p,t + h_{t-1} × r_m,t
    """
    h = hedge_weight.reindex(port_returns.index).fillna(0.0)
    h = h.shift(1).fillna(0.0) if shift else h
    return port_returns + h * market_returns.reindex(port_returns.index).fillna(0.0)


def exposure_summary(weights: pd.DataFrame) -> pd.DataFrame:
    """
    Expositions du portefeuille à chaque date :
        gross  = Σ|w|          (levier : 2.0 = 100 % long + 100 % short)
        net    = Σ w           (direction : 0 = dollar-neutral)
        long   = Σ max(w, 0),  short = Σ min(w, 0)
        n_long, n_short = nombre de lignes
    """
    return pd.DataFrame({
        "gross":   weights.abs().sum(axis=1),
        "net":     weights.sum(axis=1),
        "long":    weights.clip(lower=0).sum(axis=1),
        "short":   weights.clip(upper=0).sum(axis=1),
        "n_long":  (weights > 0).sum(axis=1),
        "n_short": (weights < 0).sum(axis=1),
    })


def turnover(weights: pd.DataFrame, shift: bool = True) -> pd.Series:
    """
    Turnover journalier = Σ_i |w_i,t - w_i,t-1| (poids détenus).

    Un portefeuille 100 % long / 100 % short entièrement renouvelé a un
    turnover de 4 (on vend toute la jambe longue : 1, on rachète toute
    la jambe courte : 1, puis on rachète/vend les nouvelles : 2).
    """
    w = weights.shift(1).fillna(0.0) if shift else weights
    return w.diff().abs().sum(axis=1).fillna(0.0)

# Séance 8 — Facteurs quantitatifs & Combinaison intelligente de signaux
# =========================================================================
# Ce script reproduit un notebook Jupyter sous forme de script Python exécutable.
# Lance-le avec : python scripts/run_session8.py
#
# Plan :
#   Partie 1 : Facteurs cross-sectionnels (momentum, low-vol, qualité)
#   Partie 2 : Information Coefficient — mesurer la qualité d'un signal
#   Partie 3 : Combinaison pondérée par IC vs poids égaux
#   Partie 4 : Alpha de Fama-French — est-ce vraiment du skill ?
#   Partie 5 : Rapport HTML de synthèse

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── Imports du backtester ────────────────────────────────────────────────────
from backtester.data      import download_data, compute_returns
from backtester.signals   import sma_crossover, rsi_signal, macd_signal, bollinger_zscore
from backtester.factors   import (
    momentum_factor, volatility_factor, quality_factor,
    compute_rolling_ic, compute_icir, fama_french_alpha,
)
from backtester.combination import (
    combine_signals, signal_quality_report, compare_weighting_modes,
)
from backtester.engine    import Backtester
from backtester.metrics   import full_metrics


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 1 — Téléchargement des données
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("PARTIE 1 — Téléchargement des données")
print("═"*65)

TICKERS  = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "JNJ"]
START    = "2018-01-01"
END      = "2024-01-01"

# Téléchargement univarié (AAPL) pour les stratégies time-series
df_aapl = compute_returns(download_data("AAPL", START, END))
close   = df_aapl["Close"]
print(f"\n✓ AAPL chargé : {len(df_aapl)} jours | {START} → {END}")

# Téléchargement multi-actifs pour les facteurs cross-sectionnels
print(f"\nTéléchargement de {len(TICKERS)} actifs pour l'analyse cross-sectionelle...")
prices_dict = {}
for ticker in TICKERS:
    try:
        df_tmp = download_data(ticker, START, END)
        prices_dict[ticker] = df_tmp["Close"]
        print(f"  ✓ {ticker}")
    except Exception as e:
        print(f"  ✗ {ticker} : {e}")

close_multi = pd.DataFrame(prices_dict).dropna()
print(f"\n✓ Univers multi-actifs : {close_multi.shape[1]} actifs × {len(close_multi)} jours")


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 2 — Facteurs cross-sectionnels
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("PARTIE 2 — Facteurs cross-sectionnels")
print("═"*65)

# ── 2.1 Facteur Momentum ────────────────────────────────────────────────────
print("\n── 2.1  Facteur Momentum (12-1 mois) ──")
mom = momentum_factor(close_multi, window=252, skip=21)
last_mom = mom.dropna().iloc[-1].sort_values(ascending=False)
print("\nScores de momentum — dernière date disponible :")
for ticker, score in last_mom.items():
    bar = "█" * int(abs(score) * 10)
    sign = "+" if score > 0 else ""
    print(f"  {ticker:6s}  {sign}{score:+.3f}  {'▲' if score > 0 else '▼'} {bar}")

# ── 2.2 Facteur Low-Volatility ──────────────────────────────────────────────
print("\n── 2.2  Facteur Low-Volatility ──")
vol_factor = volatility_factor(close_multi, window=63)
last_vol = vol_factor.dropna().iloc[-1].sort_values(ascending=False)
print("\nScores low-volatility — dernière date (positif = peu volatil) :")
for ticker, score in last_vol.items():
    bar = "█" * int(abs(score) * 5)
    sign = "+" if score > 0 else ""
    print(f"  {ticker:6s}  {sign}{score:+.3f}  {bar}")

# ── 2.3 Facteur Qualité ─────────────────────────────────────────────────────
print("\n── 2.3  Facteur Qualité (Sharpe rolling) ──")
qual = quality_factor(close_multi, window=252)
last_qual = qual.dropna().iloc[-1].sort_values(ascending=False)
print("\nScores qualité — dernière date :")
for ticker, score in last_qual.items():
    bar = "█" * int(abs(score) * 5)
    sign = "+" if score > 0 else ""
    print(f"  {ticker:6s}  {sign}{score:+.3f}  {bar}")

# ── 2.4 Signaux composite cross-sectionnel → time-series ────────────────────
print("\n── 2.4  Signal composite cross-sectionnel → stratégie sur AAPL ──")
print("""
Passage cross-sectionnel → time-series pour AAPL :
  - On extrait la colonne AAPL de chaque facteur normalisé
  - On combine les 3 scores pour obtenir un signal unique
  - Interprétation : score élevé = AAPL est attractif RELATIVEMENT aux autres
""")

mom_aapl  = mom["AAPL"].rename("momentum")
vol_aapl  = vol_factor["AAPL"].rename("low_vol")
qual_aapl = qual["AAPL"].rename("quality")

# Combinaison simple à poids égaux des 3 facteurs
factor_signal = (mom_aapl + vol_aapl + qual_aapl) / 3
factor_signal = factor_signal.clip(-1, 1)

print(f"Signal factoriel AAPL — dernière valeur : {factor_signal.dropna().iloc[-1]:+.4f}")
print(f"Signal factoriel AAPL — moyenne : {factor_signal.dropna().mean():+.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 3 — Information Coefficient (IC)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("PARTIE 3 — Information Coefficient (IC)")
print("═"*65)

print("""
L'IC mesure à quel point un signal prédit VRAIMENT les rendements futurs.
  IC = Spearman(signal[J], rendement[J+1])
  IC > 0.05 = signal avec de la valeur prédictive
  IC > 0.10 = signal fort
  IC < 0.0  = signal sans valeur (ou inversé)

⚠️  Le shift(-1) dans forward_returns est intentionnel :
    on regarde EN ARRIÈRE pour mesurer la qualité historique,
    jamais pour générer des positions futures.
""")

# Construction des signaux time-series pour AAPL
sig_sma  = sma_crossover(close, fast=20, slow=50)
sig_rsi  = rsi_signal(close, period=14)
sig_macd = macd_signal(close, fast=12, slow=26)
sig_boll = bollinger_zscore(close, window=20)

signals = {
    "SMA 20/50":    sig_sma,
    "RSI 14":       sig_rsi,
    "MACD 12/26/9": sig_macd,
    "Bollinger":    sig_boll,
}

# Forward returns : le seul shift(-1) autorisé dans tout le projet
forward_returns = close.pct_change().shift(-1)

# IC rolling (fenêtre 63 jours = 3 mois)
print("── IC rolling sur 63 jours ──\n")
ic_by_signal = {}
for name, sig in signals.items():
    ic_rolling = compute_rolling_ic(sig.fillna(0), forward_returns, window=63)
    ic_by_signal[name] = ic_rolling

    ic_clean = ic_rolling.dropna()
    ic_mean  = ic_clean.mean()
    ic_std   = ic_clean.std()
    icir     = ic_mean / ic_std if ic_std > 0 else 0.0
    pct_pos  = (ic_clean > 0).mean() * 100

    quality = "🟢 Bon" if ic_mean > 0.05 else ("🟡 Faible" if ic_mean > 0 else "🔴 Négatif")
    print(f"  {name:15s}  IC={ic_mean:+.4f}  std={ic_std:.4f}  ICIR={icir:+.3f}  "
          f"IC>0: {pct_pos:.0f}%  {quality}")

# ICIR par signal
print("\n── ICIR rolling sur 63 jours ──\n")
for name, ic_rolling in ic_by_signal.items():
    icir_series = compute_icir(ic_rolling, window=63)
    icir_clean  = icir_series.dropna()
    print(f"  {name:15s}  ICIR moyen={icir_clean.mean():+.3f}  "
          f"ICIR std={icir_clean.std():.3f}  "
          f"ICIR > 0.5 : {(icir_clean > 0.5).mean()*100:.0f}% du temps")

# Rapport de qualité
print("\n── Rapport de qualité complet ──\n")
report = signal_quality_report(signals, forward_returns, ic_window=63)
print(report.to_string())


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 4 — Combinaison pondérée vs poids égaux
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("PARTIE 4 — Combinaison : poids égaux vs IC vs ICIR")
print("═"*65)

print("\nBacktest des 3 modes de pondération sur AAPL 2018-2024...")
initial_capital = 10_000.0

results_by_mode = {}
for mode in ["equal", "ic", "icir"]:
    try:
        sig_combined = combine_signals(
            signals,
            weights_mode=mode,
            forward_returns=forward_returns,
            ic_window=63,
        )

        bt  = Backtester(df_aapl, initial_capital=initial_capital)
        out = bt.run(sig_combined.fillna(0))
        results_by_mode[mode] = out

    except Exception as e:
        print(f"  ✗ Mode '{mode}' : {e}")
        continue

# Tableau comparatif
print("\n┌─────────────────────────────────────────────────────────────────┐")
print("│  Comparaison des modes de pondération — AAPL 2018-2024          │")
print("├────────────┬──────────┬──────────┬──────────┬──────────┬────────┤")
print("│ Mode       │ CAGR (%) │  Sharpe  │  Calmar  │ MDD (%)  │ Cap.   │")
print("├────────────┼──────────┼──────────┼──────────┼──────────┼────────┤")

for mode, out in results_by_mode.items():
    m = out["metrics"]
    print(f"│ {mode:10s} │ {m['CAGR (%)']:8.2f} │ {m['Sharpe']:8.2f} │ "
          f"{m['Calmar']:8.2f} │ {m['Max Drawdown (%)']:8.2f} │ "
          f"{m['Capital final (€)']:6.0f} │")

print("└────────────┴──────────┴──────────┴──────────┴──────────┴────────┘")

# Buy & Hold comme benchmark
bah_returns  = df_aapl["returns"].fillna(0)
bah_equity   = initial_capital * (1 + bah_returns).cumprod()
bah_metrics  = full_metrics(bah_returns, bah_equity)
print(f"\n  Buy & Hold AAPL : CAGR={bah_metrics['CAGR (%)']:.2f}%  "
      f"Sharpe={bah_metrics['Sharpe']:.2f}  MDD={bah_metrics['Max Drawdown (%)']:.2f}%")

# Signal composite factoriel (momentum + low-vol + qualité)
print("\n── Signal composite factoriel vs signaux techniques ──\n")
bt_factor = Backtester(df_aapl, initial_capital=initial_capital)
out_factor = bt_factor.run(factor_signal.fillna(0))
mf = out_factor["metrics"]
print(f"  Facteurs cross-sectionnel : CAGR={mf['CAGR (%)']:.2f}%  "
      f"Sharpe={mf['Sharpe']:.2f}  MDD={mf['Max Drawdown (%)']:.2f}%")


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 5 — Alpha de Fama-French
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("PARTIE 5 — Analyse alpha (CAPM simplifié)")
print("═"*65)

print("""
Régression : R_stratégie = α + β × R_marché + ε

  α > 0 et p < 0.05 : la stratégie génère de la valeur RÉELLE
                      au-delà de la simple exposition au marché
  α ≈ 0             : la stratégie fait du beta-déguisé
  β > 1             : plus risqué que le marché
  β < 1             : défensif
""")

# Téléchargement du benchmark (SPY = proxy S&P 500)
try:
    df_spy  = compute_returns(download_data("SPY", START, END))
    mkt_ret = df_spy["returns"].fillna(0)

    for mode, out in results_by_mode.items():
        strat_ret = out["results"]["strategy_returns"]
        fa = fama_french_alpha(strat_ret, mkt_ret)
        sig_str = "✓ SIGNIFICATIF" if fa.get("alpha significatif") else "✗ non significatif"
        print(f"\n  Mode '{mode}' :")
        print(f"    α = {fa['alpha (annualisé %)']:+.2f}%/an  "
              f"β = {fa['beta']:.3f}  "
              f"R² = {fa['R²']:.3f}  "
              f"p = {fa['p-value alpha']:.4f}  {sig_str}")

except Exception as e:
    print(f"  ✗ Impossible de télécharger SPY : {e}")
    print("  → Passe cette partie si pas de connexion internet")


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 6 — Rapport HTML de synthèse
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("PARTIE 6 — Génération du rapport HTML")
print("═"*65)

def generate_html_report(results_by_mode, report_df, bah_metrics):
    """Génère un rapport HTML autonome pour la séance 8."""

    rows_modes = ""
    for mode, out in results_by_mode.items():
        m = out["metrics"]
        rows_modes += f"""
        <tr>
            <td><strong>{mode}</strong></td>
            <td class="{'pos' if m['CAGR (%)'] > 0 else 'neg'}">{m['CAGR (%)']:.2f}%</td>
            <td class="{'pos' if m['Sharpe'] > 0.5 else ''}">{m['Sharpe']:.2f}</td>
            <td>{m['Calmar']:.2f}</td>
            <td class="neg">{m['Max Drawdown (%)']:.2f}%</td>
            <td>{m['Win Rate (%)']:.1f}%</td>
            <td>{m['Capital final (€)']:.0f} €</td>
        </tr>"""

    rows_bah = f"""
        <tr style="background:#f0f4ff">
            <td><strong>Buy &amp; Hold</strong></td>
            <td class="{'pos' if bah_metrics['CAGR (%)'] > 0 else 'neg'}">{bah_metrics['CAGR (%)']:.2f}%</td>
            <td>{bah_metrics['Sharpe']:.2f}</td>
            <td>{bah_metrics['Calmar']:.2f}</td>
            <td class="neg">{bah_metrics['Max Drawdown (%)']:.2f}%</td>
            <td>{bah_metrics['Win Rate (%)']:.1f}%</td>
            <td>—</td>
        </tr>"""

    # Tableau IC
    rows_ic = ""
    for sig_name, row in report_df.iterrows():
        quality_class = "pos" if row["IC moyen"] > 0.05 else ("" if row["IC moyen"] > 0 else "neg")
        rows_ic += f"""
        <tr>
            <td>{sig_name}</td>
            <td class="{quality_class}">{row['IC moyen']:+.4f}</td>
            <td>{row['IC std']:.4f}</td>
            <td class="{'pos' if row['ICIR'] > 0.5 else ''}">{row['ICIR']:+.4f}</td>
            <td>{row['IC > 0 (%)']:.1f}%</td>
            <td>{row['Poids IC (%)']:.1f}%</td>
            <td>{row['Poids ICIR (%)']:.1f}%</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>Backtester — Séance 8 : Facteurs & Combinaison</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f7fa;
            color: #1a1a2e; line-height: 1.6; }}
    header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
              color: white; padding: 40px; text-align: center; }}
    header h1 {{ font-size: 2em; letter-spacing: 2px; }}
    header p  {{ opacity: 0.8; margin-top: 8px; }}
    .container {{ max-width: 1100px; margin: 30px auto; padding: 0 20px; }}
    .card {{ background: white; border-radius: 12px; padding: 28px;
             margin-bottom: 24px; box-shadow: 0 2px 12px rgba(0,0,0,.08); }}
    .card h2 {{ font-size: 1.2em; color: #0f3460; border-left: 4px solid #e94560;
                padding-left: 12px; margin-bottom: 18px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.92em; }}
    th {{ background: #0f3460; color: white; padding: 10px 14px;
          text-align: left; font-weight: 500; }}
    td {{ padding: 9px 14px; border-bottom: 1px solid #eef0f5; }}
    tr:hover {{ background: #f8f9ff; }}
    .pos {{ color: #22c55e; font-weight: 600; }}
    .neg {{ color: #ef4444; font-weight: 600; }}
    .concept {{ background: #f0f4ff; border-left: 4px solid #6366f1;
                padding: 16px 20px; border-radius: 8px; margin-bottom: 18px;
                font-size: 0.93em; line-height: 1.7; }}
    .concept code {{ background: #e0e7ff; padding: 2px 6px; border-radius: 4px;
                     font-family: monospace; font-size: 0.9em; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .formula {{ background: #1a1a2e; color: #e2e8f0; padding: 14px 18px;
                border-radius: 8px; font-family: monospace; font-size: 0.88em;
                line-height: 1.8; margin: 12px 0; }}
    footer {{ text-align: center; padding: 30px; color: #888; font-size: 0.85em; }}
    @media(max-width:700px){{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<header>
  <h1>📊 Backtester — Séance 8</h1>
  <p>Facteurs quantitatifs &amp; Combinaison intelligente de signaux</p>
  <p style="margin-top:6px;font-size:.85em;opacity:.6">AAPL · 2018–2024 · Capital initial : 10 000 €</p>
</header>

<div class="container">

  <!-- Concepts clés -->
  <div class="card">
    <h2>🧠 Concepts clés de la séance 8</h2>
    <div class="grid">
      <div class="concept">
        <strong>Facteurs cross-sectionnels vs Time-series</strong><br>
        Un signal time-series (<code>SMA</code>, <code>RSI</code>) compare un actif à son propre passé.<br>
        Un facteur cross-sectionnel (<code>Momentum</code>, <code>Low-Vol</code>) compare plusieurs actifs
        entre eux à un instant T → il y a toujours des gagnants et des perdants.
      </div>
      <div class="concept">
        <strong>Information Coefficient (IC)</strong><br>
        <code>IC = Spearman(signal[J], rendement[J+1])</code><br>
        Mesure si le signal prédit vraiment les rendements futurs.<br>
        IC &gt; 0.05 = signal utile · IC &gt; 0.10 = signal fort · IC &lt; 0 = bruit
      </div>
      <div class="concept">
        <strong>ICIR — Régularité du signal</strong><br>
        <code>ICIR = IC_moyen / IC_std</code><br>
        C'est le Sharpe ratio du signal lui-même.<br>
        ICIR &gt; 0.5 = signal exploitable · ICIR &gt; 1.0 = excellent
      </div>
      <div class="concept">
        <strong>Loi fondamentale de Grinold & Kahn</strong><br>
        <code>Sharpe ≈ ICIR × √(N_paris)</code><br>
        Deux leviers pour améliorer la performance :<br>
        1. Améliorer la qualité du signal (IC↑)<br>
        2. Augmenter le nombre de paris indépendants (N↑)
      </div>
    </div>
  </div>

  <!-- Rapport IC -->
  <div class="card">
    <h2>📡 Qualité des signaux — Information Coefficient (IC rolling 63j)</h2>
    <div class="formula">
IC = Spearman(signal[J], return[J+1])    ← mesure de prédictivité
ICIR = IC_mean / IC_std                  ← mesure de régularité (Sharpe du signal)
Poids IC   ∝ IC_mean (les meilleurs signaux pèsent plus)
Poids ICIR ∝ ICIR    (les signaux réguliers pèsent plus)
    </div>
    <table>
      <thead>
        <tr>
          <th>Signal</th>
          <th>IC moyen</th>
          <th>IC std</th>
          <th>ICIR</th>
          <th>IC &gt; 0</th>
          <th>Poids IC</th>
          <th>Poids ICIR</th>
        </tr>
      </thead>
      <tbody>{rows_ic}</tbody>
    </table>
  </div>

  <!-- Comparaison modes -->
  <div class="card">
    <h2>⚖️ Comparaison des modes de pondération — Backtest AAPL</h2>
    <table>
      <thead>
        <tr>
          <th>Mode</th>
          <th>CAGR</th>
          <th>Sharpe</th>
          <th>Calmar</th>
          <th>Max DD</th>
          <th>Win Rate</th>
          <th>Capital final</th>
        </tr>
      </thead>
      <tbody>
        {rows_modes}
        {rows_bah}
      </tbody>
    </table>
    <p style="margin-top:14px;font-size:.88em;color:#666">
      💡 <strong>Note :</strong> Les modes 'ic' et 'icir' utilisent les rendements futurs
      pour calibrer les poids — c'est une mesure de qualité a posteriori,
      pas du look-ahead bias dans les positions (le shift(1) reste en place).
    </p>
  </div>

  <!-- Facteurs -->
  <div class="card">
    <h2>🏗️ Les 3 facteurs implémentés</h2>
    <div class="grid">
      <div class="concept">
        <strong>📈 Momentum (Jegadeesh &amp; Titman, 1993)</strong><br>
        Rendement sur 12 mois excluant le dernier mois (skip=21j).<br>
        L'exclusion du dernier mois évite l'effet de retournement à court terme.<br>
        <code>signal = close.pct_change(231).shift(21)</code>
      </div>
      <div class="concept">
        <strong>🛡️ Low-Volatility (Black, 1972)</strong><br>
        Anomalie paradoxale : les actifs peu risqués surperforment à long terme.<br>
        Cause : contraintes institutionnelles + biais comportementaux.<br>
        <code>signal = -(vol_réalisée_63j × √252)</code>
      </div>
      <div class="concept">
        <strong>⭐ Qualité (proxy Sharpe rolling)</strong><br>
        Proxy disponible sans données fondamentales : Sharpe ratio rolling.<br>
        Corrélé avec ROE, marge brute et endettement mais imparfait.<br>
        <code>signal = mean(R) / std(R) × √252   [sur 252j]</code>
      </div>
      <div class="concept">
        <strong>🔬 Alpha (CAPM simplifié)</strong><br>
        <code>R_strat = α + β × R_mkt + ε</code><br>
        α &gt; 0 et p &lt; 0.05 → valeur ajoutée réelle<br>
        α ≈ 0 → du beta-déguisé (un ETF ferait pareil)
      </div>
    </div>
  </div>

  <!-- Prochaine étape -->
  <div class="card">
    <h2>🚀 Suite du projet — Séance 9</h2>
    <div class="concept">
      <strong>Positions short &amp; stratégies Long/Short</strong><br>
      Jusqu'ici le backtester est <em>long only</em> : position ∈ [0, 1].<br>
      La séance 9 étend à <em>long/short</em> : position ∈ [-1, +1].<br><br>
      Concepts prévus : dollar-neutral, beta-neutral, pair trading,
      portefeuille quantile (long top 20% / short bottom 20%).
    </div>
  </div>

</div>
<footer>
  Backtester algorithmique — Séance 8/N · Généré automatiquement
</footer>
</body>
</html>"""
    return html


# Génération du rapport
try:
    html_content = generate_html_report(results_by_mode, report, bah_metrics)
    report_path  = "reports/session8_report.html"

    import os
    os.makedirs("reports", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n✓ Rapport HTML généré : {report_path}")
    print(f"  → Ouvre dans ton navigateur : file://$(pwd)/{report_path}")
except Exception as e:
    print(f"✗ Erreur génération HTML : {e}")


# ═══════════════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("RÉSUMÉ SÉANCE 8")
print("═"*65)
print("""
Modules créés :
  ✓ backtester/factors.py
      momentum_factor()      — rendement 12-1 mois, z-score cross-sectionnel
      volatility_factor()    — low-vol anomaly, signal négatif
      quality_factor()       — Sharpe rolling comme proxy qualité
      compute_rolling_ic()   — corrélation de Spearman rolling
      compute_icir()         — IC_mean / IC_std (Sharpe du signal)
      fama_french_alpha()    — régression CAPM pour isoler l'alpha

  ✓ backtester/combination.py
      combine_signals()          — 4 modes : equal, fixed, ic, icir
      rolling_ic_weights()       — poids IC dynamiques par fenêtre glissante
      rolling_icir_weights()     — poids ICIR dynamiques
      signal_quality_report()    — tableau IC/ICIR/poids par signal
      compare_weighting_modes()  — comparaison backtest des 3 modes

  ✓ tests/test_factors.py      — 18 tests unitaires
  ✓ tests/test_combination.py  — 16 tests unitaires
  ✓ reports/session8_report.html

Commit suggéré :
  git add backtester/factors.py backtester/combination.py
  git add tests/test_factors.py tests/test_combination.py
  git add reports/session8_report.html
  git commit -m "feat: multi-signal combination and factor analysis"
""")

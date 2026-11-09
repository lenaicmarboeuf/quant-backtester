"""
scripts/run_wfo.py — Walk-Forward Optimisation configurable en CLI

Usage :
    python scripts/run_wfo.py --ticker AAPL --start 2018-01-01 --end 2024-01-01
    python scripts/run_wfo.py --ticker MSFT --is-window 504 --oos-window 126 --anchored
    python scripts/run_wfo.py --ticker SPY --metric "CAGR (%)" --fast 10 20 30 --slow 40 60 80

Options complètes :
    --ticker      : symbole Yahoo Finance (défaut : AAPL)
    --start       : date de début (défaut : 2018-01-01)
    --end         : date de fin   (défaut : 2024-01-01)
    --is-window   : fenêtre IS en jours (défaut : 252)
    --oos-window  : fenêtre OOS en jours (défaut : 63)
    --anchored    : IS croissant au lieu de glissant
    --metric      : métrique d'optimisation IS (défaut : Sharpe)
    --fast        : valeurs à tester pour fast SMA (défaut : 10 20 30)
    --slow        : valeurs à tester pour slow SMA (défaut : 40 60 80 100)
    --capital     : capital initial (défaut : 10000)
    --output      : chemin du rapport HTML (défaut : reports/wfo_report.html)
    --no-cpcv     : désactive la CPCV (plus rapide)
"""

import argparse
import json
import sys
from pathlib import Path

# Permet d'importer backtester depuis la racine du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtester.data import download_data, compute_returns
from backtester.signals import sma_crossover
from backtester.validation import walk_forward, purged_cv, robustness_surface
from typing import Callable, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Parsing des arguments CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Walk-Forward Optimisation pour le backtester.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ticker",     default="AAPL",       help="Symbole Yahoo Finance")
    parser.add_argument("--start",      default="2018-01-01", help="Date de début YYYY-MM-DD")
    parser.add_argument("--end",        default="2024-01-01", help="Date de fin YYYY-MM-DD")
    parser.add_argument("--is-window",  type=int, default=252, help="Fenêtre IS (jours)")
    parser.add_argument("--oos-window", type=int, default=63,  help="Fenêtre OOS (jours)")
    parser.add_argument("--anchored",   action="store_true",   help="IS croissant (ancré)")
    parser.add_argument("--metric",     default="Sharpe",      help="Métrique d'optimisation IS")
    parser.add_argument("--fast",       nargs="+", type=int, default=[10, 20, 30],
                        help="Valeurs fast SMA à tester")
    parser.add_argument("--slow",       nargs="+", type=int, default=[40, 60, 80, 100],
                        help="Valeurs slow SMA à tester")
    parser.add_argument("--capital",    type=float, default=10_000.0, help="Capital initial")
    parser.add_argument("--output",     default="reports/wfo_report.html", help="Fichier de sortie")
    parser.add_argument("--no-cpcv",    action="store_true", help="Désactiver la CPCV")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Génération du rapport HTML
# ─────────────────────────────────────────────────────────────────────────────

def _series_to_json(series: pd.Series) -> str:
    """Convertit une Series en JSON pour Chart.js."""
    clean = series.dropna()
    labels = [str(d.date()) for d in clean.index]
    values = [round(float(v), 2) for v in clean.values]
    return json.dumps({"labels": labels, "values": values})


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Convertit un DataFrame en table HTML stylée."""
    display = df.head(max_rows).copy()
    # Arrondir les numériques
    for col in display.select_dtypes(include=[float]).columns:
        display[col] = display[col].round(3)

    rows = ""
    for _, row in display.iterrows():
        cells = "".join(f"<td>{v}</td>" for v in row)
        rows += f"<tr>{cells}</tr>"

    headers = "".join(f"<th>{c}</th>" for c in display.columns)
    return f"""
    <table>
      <thead><tr>{headers}</tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """


def generate_html_report(
    ticker: str,
    args,
    wfo_result: dict,
    cpcv_result: Optional[dict],
    output_path: str,
) -> None:
    """Génère le rapport HTML complet."""

    summary      = wfo_result.get("summary", {})
    oos_results  = wfo_result.get("oos_results", pd.DataFrame())
    best_params  = wfo_result.get("best_params",  pd.DataFrame())
    equity_oos   = wfo_result.get("equity_oos",   pd.Series())

    equity_json  = _series_to_json(equity_oos)
    table_oos    = _df_to_html_table(oos_results)
    table_params = _df_to_html_table(best_params)

    # Bloc CPCV
    cpcv_html = ""
    if cpcv_result and "fold_metrics" in cpcv_result:
        mean_m   = cpcv_result.get("mean_metrics", {})
        def_sr   = cpcv_result.get("deflated_sharpe", "N/A")
        fold_tbl = _df_to_html_table(cpcv_result["fold_metrics"])
        cpcv_html = f"""
        <section>
          <h2>📊 Combinatorial Purged Cross-Validation (CPCV)</h2>
          <div class="kpi-grid">
            <div class="kpi">
              <span class="kpi-label">Sharpe moyen (folds)</span>
              <span class="kpi-value">{mean_m.get('Sharpe', 'N/A')}</span>
            </div>
            <div class="kpi">
              <span class="kpi-label">Deflated Sharpe</span>
              <span class="kpi-value">{def_sr}</span>
            </div>
            <div class="kpi">
              <span class="kpi-label">CAGR moyen (%)</span>
              <span class="kpi-value">{mean_m.get('CAGR (%)', 'N/A')}</span>
            </div>
            <div class="kpi">
              <span class="kpi-label">Max DD moyen (%)</span>
              <span class="kpi-value">{mean_m.get('Max Drawdown (%)', 'N/A')}</span>
            </div>
          </div>
          <h3>Métriques par fold</h3>
          {fold_tbl}
          <p class="note">
            Le <strong>Deflated Sharpe</strong> corrige le Sharpe pour le nombre d'essais
            (n_splits folds). Un DSR &gt; 0 signifie que la performance n'est
            probablement pas due au hasard.
          </p>
        </section>
        """

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>WFO Report — {ticker}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0f1117; color: #e2e8f0;
      padding: 2rem; max-width: 1200px; margin: 0 auto;
    }}
    h1 {{ font-size: 1.8rem; color: #7c3aed; margin-bottom: 0.25rem; }}
    h2 {{ font-size: 1.2rem; color: #a78bfa; margin: 2rem 0 1rem; }}
    h3 {{ font-size: 1rem; color: #c4b5fd; margin: 1.5rem 0 0.5rem; }}
    .subtitle {{ color: #64748b; font-size: 0.9rem; margin-bottom: 2rem; }}

    .kpi-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 1rem; margin: 1rem 0;
    }}
    .kpi {{
      background: #1e1b4b; border: 1px solid #312e81;
      border-radius: 8px; padding: 1rem; text-align: center;
    }}
    .kpi-label {{ display: block; font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.5rem; }}
    .kpi-value {{ display: block; font-size: 1.5rem; font-weight: 700; color: #a78bfa; }}

    section {{
      background: #161b2e; border: 1px solid #1e2a40;
      border-radius: 12px; padding: 1.5rem; margin: 1.5rem 0;
    }}
    canvas {{ max-height: 300px; margin: 1rem 0; }}

    table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; margin: 1rem 0; }}
    th {{ background: #1e1b4b; color: #a78bfa; padding: 0.5rem; text-align: left; }}
    td {{ padding: 0.4rem 0.5rem; border-bottom: 1px solid #1e2a40; }}
    tr:hover td {{ background: #1a2035; }}

    .config-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 0.5rem; font-size: 0.85rem;
    }}
    .config-item {{ background: #1e2a40; border-radius: 6px; padding: 0.5rem 0.75rem; }}
    .config-item span {{ color: #64748b; }}

    .note {{ font-size: 0.8rem; color: #64748b; margin-top: 1rem;
             border-left: 3px solid #312e81; padding-left: 0.75rem; }}
    .badge-good {{ color: #34d399; }} .badge-warn {{ color: #fbbf24; }}
    .badge-bad  {{ color: #f87171; }}
  </style>
</head>
<body>

<h1>📈 Walk-Forward Report — {ticker}</h1>
<p class="subtitle">Généré le {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} |
  {args.start} → {args.end} | Signal: SMA Crossover</p>

<!-- Configuration -->
<section>
  <h2>⚙️ Configuration</h2>
  <div class="config-grid">
    <div class="config-item"><span>Ticker</span> &nbsp; {ticker}</div>
    <div class="config-item"><span>IS window</span> &nbsp; {args.is_window} jours</div>
    <div class="config-item"><span>OOS window</span> &nbsp; {args.oos_window} jours</div>
    <div class="config-item"><span>Mode IS</span> &nbsp; {"Ancré" if args.anchored else "Glissant"}</div>
    <div class="config-item"><span>Métrique opt.</span> &nbsp; {args.metric}</div>
    <div class="config-item"><span>Fast SMA</span> &nbsp; {args.fast}</div>
    <div class="config-item"><span>Slow SMA</span> &nbsp; {args.slow}</div>
    <div class="config-item"><span>Capital</span> &nbsp; {args.capital:,.0f} €</div>
  </div>
</section>

<!-- Résumé OOS -->
<section>
  <h2>🎯 Performances Out-of-Sample (OOS)</h2>
  <div class="kpi-grid">
    <div class="kpi">
      <span class="kpi-label">CAGR OOS</span>
      <span class="kpi-value">{summary.get('CAGR (%)', 'N/A')} %</span>
    </div>
    <div class="kpi">
      <span class="kpi-label">Sharpe OOS</span>
      <span class="kpi-value">{summary.get('Sharpe', 'N/A')}</span>
    </div>
    <div class="kpi">
      <span class="kpi-label">Max Drawdown</span>
      <span class="kpi-value">{summary.get('Max Drawdown (%)', 'N/A')} %</span>
    </div>
    <div class="kpi">
      <span class="kpi-label">Capital final</span>
      <span class="kpi-value">{summary.get('Capital final (€)', 'N/A')} €</span>
    </div>
    <div class="kpi">
      <span class="kpi-label">Win Rate</span>
      <span class="kpi-value">{summary.get('Win Rate (%)', 'N/A')} %</span>
    </div>
    <div class="kpi">
      <span class="kpi-label">Profit Factor</span>
      <span class="kpi-value">{summary.get('Profit Factor', 'N/A')}</span>
    </div>
    <div class="kpi">
      <span class="kpi-label">VaR 95%</span>
      <span class="kpi-value">{summary.get('VaR 95% (%)', 'N/A')} %</span>
    </div>
    <div class="kpi">
      <span class="kpi-label">Fenêtres OOS</span>
      <span class="kpi-value">{summary.get('Nb fenêtres OOS', 'N/A')}</span>
    </div>
  </div>

  <!-- Equity OOS -->
  <h3>Courbe d'equity OOS</h3>
  <canvas id="equityChart"></canvas>

  <h3>Métriques par fenêtre OOS</h3>
  {table_oos}
</section>

<!-- Meilleurs paramètres -->
<section>
  <h2>🔧 Meilleurs paramètres IS par fenêtre</h2>
  {table_params}
  <p class="note">
    Si les paramètres changent beaucoup d'une fenêtre à l'autre, la stratégie
    est instable (elle "chasse" le passé). Des paramètres stables = signal robuste.
  </p>
</section>

{cpcv_html}

<!-- Interprétation -->
<section>
  <h2>💡 Comment interpréter ce rapport</h2>
  <p style="font-size:0.9rem; color:#94a3b8; line-height:1.7;">
    <strong>CAGR OOS &gt; 0%</strong> → la stratégie crée de la valeur hors-échantillon.<br>
    <strong>Sharpe OOS &gt; 0.5</strong> → acceptable | <strong>&gt; 1.0</strong> → bon | <strong>&gt; 2.0</strong> → excellent.<br>
    <strong>IS/OOS ratio</strong> → si le Sharpe IS est 3× supérieur au Sharpe OOS,
    il y a du sur-apprentissage. Viser un ratio &lt; 2.<br>
    <strong>Deflated Sharpe &gt; 0</strong> → la performance n'est probablement pas due au hasard
    (correction pour le nombre d'essais testés).
  </p>
</section>

<script>
const data = {equity_json};
new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{
    labels: data.labels,
    datasets: [{{
      label: 'Equity OOS (€)',
      data: data.values,
      borderColor: '#7c3aed',
      backgroundColor: 'rgba(124,58,237,0.08)',
      borderWidth: 2,
      pointRadius: 0,
      fill: true,
      tension: 0.2,
    }}]
  }},
  options: {{
    responsive: true,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{ legend: {{ display: false }},
               tooltip: {{ callbacks: {{
                 label: ctx => '€' + ctx.parsed.y.toFixed(0)
               }} }} }},
    scales: {{
      x: {{ ticks: {{ maxTicksLimit: 12, color: '#64748b' }},
            grid: {{ color: '#1e2a40' }} }},
      y: {{ ticks: {{ color: '#64748b',
                     callback: v => '€' + v.toLocaleString() }},
            grid: {{ color: '#1e2a40' }} }},
    }}
  }}
}});
</script>

</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Rapport généré : {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Walk-Forward Optimisation — {args.ticker}")
    print(f"  {args.start} → {args.end}")
    print(f"  IS: {args.is_window}j | OOS: {args.oos_window}j | {'Ancré' if args.anchored else 'Glissant'}")
    print(f"  Paramètres testés: fast={args.fast}, slow={args.slow}")
    print(f"{'='*60}\n")

    # ── Données ──────────────────────────────────────────────────────────────
    print("📥 Téléchargement des données...")
    df = download_data(args.ticker, args.start, args.end)
    df = compute_returns(df)
    print(f"   {len(df)} jours de données chargées.\n")

    # ── WFO ──────────────────────────────────────────────────────────────────
    print("🔄 Walk-Forward Optimisation en cours...")
    param_grid = {"fast": args.fast, "slow": args.slow}

    wfo_result = walk_forward(
        df            = df,
        signal_func   = sma_crossover,
        param_grid    = param_grid,
        is_window     = args.is_window,
        oos_window    = args.oos_window,
        anchored      = args.anchored,
        metric        = args.metric,
        initial_capital = args.capital,
    )

    if "error" in wfo_result:
        print(f"❌ Erreur WFO : {wfo_result['error']}")
        sys.exit(1)

    summary = wfo_result["summary"]
    print("✅ WFO terminé.")
    print(f"\n{'─'*40}")
    print("  Résumé OOS :")
    for k, v in summary.items():
        print(f"    {k:<25} {v}")
    print(f"{'─'*40}\n")

    # ── CPCV ─────────────────────────────────────────────────────────────────
    cpcv_result = None
    if not args.no_cpcv:
        print("🔄 Combinatorial Purged Cross-Validation...")
        # On prend les meilleurs paramètres de la dernière fenêtre WFO comme référence
        best_row   = wfo_result["best_params"].iloc[-1]
        import ast
        best_params = ast.literal_eval(best_row["best_params"])

        cpcv_result = purged_cv(
            df          = df,
            signal_func = sma_crossover,
            params      = best_params,
            n_splits    = 5,
            purge_gap   = 5,
            embargo_gap = 5,
            initial_capital = args.capital,
        )

        if "error" not in cpcv_result:
            print(f"✅ CPCV terminé. Deflated Sharpe = {cpcv_result.get('deflated_sharpe', 'N/A')}\n")
        else:
            print(f"⚠️  CPCV : {cpcv_result['error']}\n")

    # ── Rapport HTML ──────────────────────────────────────────────────────────
    print("📄 Génération du rapport HTML...")
    generate_html_report(
        ticker      = args.ticker,
        args        = args,
        wfo_result  = wfo_result,
        cpcv_result = cpcv_result,
        output_path = args.output,
    )


if __name__ == "__main__":
    # Import ici pour éviter une erreur de typage en haut du fichier
    from typing import Optional
    main()

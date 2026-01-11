"""
HTML report generator for backtest results.

Uses Jinja2 templates and Plotly for interactive charts.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Template

from src.backtest.engine import BacktestResult
from src.backtest.metrics import compute_monthly_returns
from src.reporting.visualizations import (
    create_drawdown_chart,
    create_equity_curve,
    create_ewm_vol_chart,
    create_monthly_returns_heatmap,
    create_pnl_decomposition_chart,
    create_position_count_chart,
    create_rolling_sharpe,
    create_sharpe_heatmap,
)

# HTML template
REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #4a90d9;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 40px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #4a90d9;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4a90d9;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .footer {
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <p>Generated: {{ generated_at }}</p>

        <h2>Performance Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value {{ sharpe_class }}">{{ sharpe }}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {{ return_class }}">{{ ann_return }}</div>
                <div class="metric-label">Annual Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ ann_vol }}</div>
                <div class="metric-label">Annual Volatility</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">{{ max_drawdown }}</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ calmar }}</div>
                <div class="metric-label">Calmar Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ win_rate }}</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ turnover }}</div>
                <div class="metric-label">Avg Turnover</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ avg_positions }}</div>
                <div class="metric-label">Avg Positions</div>
            </div>
        </div>

        <h2>Equity Curve</h2>
        <div class="chart-container">
            <div id="equity-chart"></div>
        </div>

        <h2>Drawdown</h2>
        <div class="chart-container">
            <div id="drawdown-chart"></div>
        </div>

        <h2>Parameter Optimization</h2>
        <div class="chart-container">
            <div id="heatmap-chart"></div>
        </div>

        <h2>Rolling Sharpe Ratio (60-day)</h2>
        <div class="chart-container">
            <div id="rolling-sharpe-chart"></div>
        </div>

        <h2>PnL Decomposition</h2>
        <div class="chart-container">
            <div id="pnl-chart"></div>
        </div>

        <h2>Position Count</h2>
        <div class="chart-container">
            <div id="position-chart"></div>
        </div>

        <h2>Monthly Returns</h2>
        <div class="chart-container">
            <div id="monthly-chart"></div>
        </div>

        <h2>Top Parameter Sets</h2>
        {{ summary_table }}

        <div class="footer">
            <p>Hyperliquid Double-Sort Strategy Backtest</p>
            <p>Target Volatility: {{ target_vol }} | Max Leverage: {{ max_leverage }}</p>
        </div>
    </div>

    <script>
        {{ equity_chart_js }}
        {{ drawdown_chart_js }}
        {{ heatmap_chart_js }}
        {{ rolling_sharpe_chart_js }}
        {{ pnl_chart_js }}
        {{ position_chart_js }}
        {{ monthly_chart_js }}
    </script>
</body>
</html>
"""


def generate_chart_js(fig, div_id: str) -> str:
    """Convert Plotly figure to JavaScript for embedding."""
    fig_json = fig.to_json()
    return f"Plotly.newPlot('{div_id}', {fig_json}.data, {fig_json}.layout);"


def generate_report(
    result: BacktestResult,
    summary_df: pd.DataFrame,
    heatmap_df: pd.DataFrame,
    ewm_vols: pd.DataFrame | None = None,
    output_path: str | Path = "reports/backtest_report.html",
    target_vol: float = 0.35,
    max_leverage: float = 3.0,
) -> None:
    """
    Generate HTML report for backtest results.

    Args:
        result: BacktestResult for best parameters
        summary_df: Summary DataFrame from all parameter combinations
        heatmap_df: Sharpe heatmap DataFrame
        ewm_vols: Optional EWM volatility DataFrame
        output_path: Path for output HTML file
        target_vol: Target volatility used
        max_leverage: Max leverage used
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create charts
    equity_fig = create_equity_curve(result.returns)
    drawdown_fig = create_drawdown_chart(result.returns)
    heatmap_fig = create_sharpe_heatmap(heatmap_df)
    rolling_sharpe_fig = create_rolling_sharpe(result.returns)
    pnl_fig = create_pnl_decomposition_chart(result.funding_pnl, result.price_pnl)
    position_fig = create_position_count_chart(result.weights)
    monthly_fig = create_monthly_returns_heatmap(result.returns)

    # Format metrics
    metrics = result.metrics

    # Create summary table HTML
    summary_html = summary_df.head(10).to_html(
        classes="",
        index=False,
        float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x,
    )

    # Render template
    template = Template(REPORT_TEMPLATE)
    html = template.render(
        title=f"Backtest Report: {result.params}",
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        sharpe=f"{metrics.sharpe:.3f}",
        sharpe_class="positive" if metrics.sharpe > 0 else "negative",
        ann_return=f"{metrics.ann_return:.2%}",
        return_class="positive" if metrics.ann_return > 0 else "negative",
        ann_vol=f"{metrics.ann_vol:.2%}",
        max_drawdown=f"{metrics.max_drawdown:.2%}",
        calmar=f"{metrics.calmar:.3f}" if metrics.calmar else "N/A",
        win_rate=f"{metrics.win_rate:.2%}",
        turnover=f"{metrics.turnover:.4f}",
        avg_positions=f"{metrics.avg_positions:.1f}",
        target_vol=f"{target_vol:.0%}",
        max_leverage=f"{max_leverage:.1f}x",
        summary_table=summary_html,
        equity_chart_js=generate_chart_js(equity_fig, "equity-chart"),
        drawdown_chart_js=generate_chart_js(drawdown_fig, "drawdown-chart"),
        heatmap_chart_js=generate_chart_js(heatmap_fig, "heatmap-chart"),
        rolling_sharpe_chart_js=generate_chart_js(rolling_sharpe_fig, "rolling-sharpe-chart"),
        pnl_chart_js=generate_chart_js(pnl_fig, "pnl-chart"),
        position_chart_js=generate_chart_js(position_fig, "position-chart"),
        monthly_chart_js=generate_chart_js(monthly_fig, "monthly-chart"),
    )

    # Write file
    with open(output_path, "w") as f:
        f.write(html)


def generate_comparison_report(
    results: dict[str, BacktestResult],
    output_path: str | Path = "reports/comparison_report.html",
) -> None:
    """
    Generate comparison report for multiple parameter sets.

    Args:
        results: Dict mapping parameter name -> BacktestResult
        output_path: Path for output HTML file
    """
    # Create comparison DataFrame
    data = []
    for name, result in results.items():
        data.append({
            "params": name,
            "sharpe": result.metrics.sharpe,
            "ann_return": result.metrics.ann_return,
            "ann_vol": result.metrics.ann_vol,
            "max_drawdown": result.metrics.max_drawdown,
            "calmar": result.metrics.calmar,
        })

    df = pd.DataFrame(data).sort_values("sharpe", ascending=False)

    # Simple HTML output
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Parameter Comparison</title>
        <style>
            body {{ font-family: sans-serif; padding: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4a90d9; color: white; }}
        </style>
    </head>
    <body>
        <h1>Parameter Comparison</h1>
        {df.to_html(index=False, float_format=lambda x: f"{x:.4f}")}
    </body>
    </html>
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

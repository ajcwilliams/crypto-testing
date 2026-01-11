#!/usr/bin/env python3
"""
Generate HTML report from backtest results.

Usage:
    python -m scripts.generate_report [--results-dir reports/backtest_results]
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.engine import create_sharpe_heatmap
from src.reporting.html_report import generate_comparison_report, generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML report from backtest results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="reports/backtest_results",
        help="Directory with backtest results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML file path (default: results-dir/report.html)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = args.output or results_dir / "report.html"

    # Check for required files
    results_path = results_dir / "results.pkl"
    summary_path = results_dir / "summary.csv"
    heatmap_path = results_dir / "sharpe_heatmap.csv"
    metadata_path = results_dir / "metadata.json"

    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        logger.error("Run 'python -m scripts.run_backtest' first")
        sys.exit(1)

    # Load results
    logger.info("Loading backtest results...")

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    summary_df = pd.read_csv(summary_path)
    heatmap_df = pd.read_csv(heatmap_path, index_col=0)

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Get best result
    best_name = metadata["best_params"]
    best_result = results[best_name]

    logger.info(f"Best parameters: {best_name}")
    logger.info(f"Generating report to {output_path}...")

    # Generate main report
    generate_report(
        result=best_result,
        summary_df=summary_df,
        heatmap_df=heatmap_df,
        output_path=output_path,
        target_vol=metadata["config"]["target_vol"],
        max_leverage=metadata["config"]["max_leverage"],
    )

    # Generate comparison report
    comparison_path = results_dir / "comparison.html"
    generate_comparison_report(results, output_path=comparison_path)

    logger.info(f"Main report: {output_path}")
    logger.info(f"Comparison report: {comparison_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()

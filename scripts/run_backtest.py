#!/usr/bin/env python3
"""
Run backtest for the double-sort systematic strategy.

Usage:
    python -m scripts.run_backtest [--quick] [--output-dir reports/backtest_results]
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

from src.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    create_results_summary,
    create_sharpe_heatmap,
)
from src.backtest.parameter_grid import get_default_grid, get_quick_grid
from src.data.data_processor import DataProcessor
from src.portfolio.risk_model import (
    analyze_regime_performance,
    compute_ewm_vol_cross_section,
    compute_forward_realized_vol,
    compute_regime_indicator,
    evaluate_ewm_prediction,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run backtest for double-sort strategy"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick grid (4x4) instead of full grid (8x8)",
    )
    parser.add_argument(
        "--target-vol",
        type=float,
        default=0.35,
        help="Target volatility (default: 0.35)",
    )
    parser.add_argument(
        "--max-leverage",
        type=float,
        default=3.0,
        help="Maximum leverage (default: 3.0)",
    )
    parser.add_argument(
        "--no-costs",
        action="store_true",
        help="Exclude transaction costs",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory with processed data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/backtest_results",
        help="Directory for output files",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    processor = DataProcessor(processed_dir=args.data_dir)
    data = processor.load_processed()

    prices = data["prices"]
    returns = data["returns"]
    funding = data["funding"]
    volumes = data["volume"]

    logger.info(f"Data shape: {prices.shape}")
    logger.info(f"Date range: {prices.index.min()} to {prices.index.max()}")
    logger.info(f"Coins: {list(prices.columns)}")

    # Configure backtest
    config = BacktestConfig(
        target_vol=args.target_vol,
        max_leverage=args.max_leverage,
        include_costs=not args.no_costs,
    )

    # Select grid
    grid = get_quick_grid() if args.quick else get_default_grid()
    logger.info(f"Running {len(grid)} parameter combinations...")

    # Run backtest
    engine = BacktestEngine(
        prices=prices,
        returns=returns,
        funding=funding,
        volumes=volumes,
        config=config,
    )

    results = engine.run_grid(grid)
    logger.info(f"Completed {len(results)} backtests")

    # Create summary
    summary = create_results_summary(results)
    logger.info("\nTop 10 parameter sets by Sharpe:")
    print(summary.head(10).to_string())

    # Create heatmap
    heatmap = create_sharpe_heatmap(results)
    logger.info("\nSharpe Heatmap (Funding x Momentum lookback):")
    print(heatmap.to_string())

    # EWM Risk Analysis
    logger.info("\nRunning EWM risk analysis...")
    best_params = summary.iloc[0]["name"]
    best_result = results[best_params]

    # Compute EWM vols for best strategy
    port_returns = best_result.returns
    ewm_vols = pd.DataFrame()
    for hl in [2, 4, 8, 16, 32, 64]:
        periods = hl * 3
        ewm_vols[f"ewm_{hl}d"] = port_returns.ewm(halflife=periods).std() * (365 * 3) ** 0.5

    # Forward realized vol
    forward_vol = compute_forward_realized_vol(port_returns, forward_periods=60)

    # Evaluate prediction
    ewm_eval = evaluate_ewm_prediction(ewm_vols, forward_vol)
    logger.info("\nEWM Prediction Evaluation:")
    print(ewm_eval.to_string())

    # Regime analysis
    regime = compute_regime_indicator(ewm_vols)
    regime_perf = analyze_regime_performance(port_returns, regime, n_regimes=3)
    logger.info("\nPerformance by Risk Regime:")
    print(regime_perf.to_string())

    # Save results
    logger.info(f"\nSaving results to {output_dir}...")

    # Save summary
    summary.to_csv(output_dir / "summary.csv", index=False)

    # Save heatmap
    heatmap.to_csv(output_dir / "sharpe_heatmap.csv")

    # Save EWM analysis
    ewm_eval.to_csv(output_dir / "ewm_evaluation.csv", index=False)
    regime_perf.to_csv(output_dir / "regime_performance.csv", index=False)

    # Save best result returns
    best_result.returns.to_csv(output_dir / "best_returns.csv")
    best_result.weights.to_csv(output_dir / "best_weights.csv")

    # Save full results (pickle for later analysis)
    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    # Save metadata
    metadata = {
        "config": {
            "target_vol": config.target_vol,
            "max_leverage": config.max_leverage,
            "include_costs": config.include_costs,
        },
        "grid_size": len(grid),
        "best_params": best_params,
        "best_sharpe": float(summary.iloc[0]["sharpe"]),
        "data_shape": list(prices.shape),
        "date_range": [str(prices.index.min()), str(prices.index.max())],
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Done!")
    logger.info(f"\nBest parameters: {best_params}")
    logger.info(f"Best Sharpe: {summary.iloc[0]['sharpe']:.3f}")
    logger.info(f"Annual Return: {summary.iloc[0]['ann_return']:.2%}")
    logger.info(f"Max Drawdown: {summary.iloc[0]['max_drawdown']:.2%}")


if __name__ == "__main__":
    main()

"""
Backtest engine for systematic strategy.

Runs backtests with:
- 8-hourly rebalancing
- Funding payment accounting
- Transaction costs
- Volatility targeting
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.backtest.metrics import BacktestMetrics, compute_all_metrics
from src.backtest.parameter_grid import ParameterGrid, ParameterSet, get_default_grid
from src.portfolio.construction import construct_weights, filter_universe
from src.portfolio.vol_targeting import apply_vol_targeting
from src.signals.signal_combiner import compute_double_sort_signals

logger = logging.getLogger(__name__)

# Transaction costs
TAKER_FEE = 0.00035  # 0.035%
SLIPPAGE = 0.0002  # 0.02%
TOTAL_COST_PER_SIDE = TAKER_FEE + SLIPPAGE  # ~0.055% per side


@dataclass
class BacktestResult:
    """Container for backtest results."""

    params: ParameterSet
    metrics: BacktestMetrics
    returns: pd.Series
    weights: pd.DataFrame
    funding_pnl: pd.Series
    price_pnl: pd.Series
    scale_factors: pd.Series


@dataclass
class BacktestConfig:
    """Configuration for backtest."""

    target_vol: float = 0.35
    vol_lookback_periods: int = 60  # ~20 days
    max_leverage: float = 3.0
    include_costs: bool = True
    universe_percentile: float = 0.25
    min_history_periods: int = 60


class BacktestEngine:
    """Engine for running systematic backtests."""

    def __init__(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        funding: pd.DataFrame,
        volumes: pd.DataFrame,
        config: BacktestConfig | None = None,
    ):
        """
        Initialize backtest engine.

        Args:
            prices: DataFrame of prices (periods x coins)
            returns: DataFrame of returns (periods x coins)
            funding: DataFrame of funding rates (periods x coins)
            volumes: DataFrame of volumes (periods x coins)
            config: BacktestConfig object
        """
        self.prices = prices
        self.returns = returns
        self.funding = funding
        self.volumes = volumes
        self.config = config or BacktestConfig()

        # Pre-compute universe filter
        self.universe = filter_universe(
            volumes,
            percentile=self.config.universe_percentile,
            min_history=self.config.min_history_periods,
        )

    def run_single(self, params: ParameterSet) -> BacktestResult:
        """
        Run backtest for a single parameter set.

        Args:
            params: ParameterSet with lookback periods

        Returns:
            BacktestResult object
        """
        # Compute signals and positions
        funding_q, momentum_q, positions = compute_double_sort_signals(
            prices=self.prices,
            returns=self.returns,
            funding=self.funding,
            funding_lookback=params.funding_lookback_periods,
            momentum_lookback=params.momentum_lookback_periods,
            risk_adjusted=True,
        )

        # Construct weights
        raw_weights = construct_weights(
            positions=positions,
            universe=self.universe,
            equal_weight=True,
        )

        # Apply volatility targeting
        scaled_weights, scale_factors = apply_vol_targeting(
            weights=raw_weights,
            returns=self.returns,
            target_vol=self.config.target_vol,
            lookback_periods=self.config.vol_lookback_periods,
            max_leverage=self.config.max_leverage,
        )

        # Compute returns
        portfolio_returns, funding_pnl, price_pnl = self._compute_returns(
            scaled_weights
        )

        # Apply transaction costs
        if self.config.include_costs:
            costs = self._compute_transaction_costs(scaled_weights)
            portfolio_returns = portfolio_returns - costs

        # Compute metrics
        metrics = compute_all_metrics(portfolio_returns, scaled_weights)

        return BacktestResult(
            params=params,
            metrics=metrics,
            returns=portfolio_returns,
            weights=scaled_weights,
            funding_pnl=funding_pnl,
            price_pnl=price_pnl,
            scale_factors=scale_factors,
        )

    def run_grid(
        self,
        grid: ParameterGrid | None = None,
        parallel: bool = False,
    ) -> dict[str, BacktestResult]:
        """
        Run backtest for all parameter combinations.

        Args:
            grid: ParameterGrid to test (default: full 8x8 grid)
            parallel: If True, run in parallel (not implemented)

        Returns:
            Dict mapping parameter name -> BacktestResult
        """
        if grid is None:
            grid = get_default_grid()

        results = {}

        for params in tqdm(grid, desc="Running backtests", total=len(grid)):
            try:
                result = self.run_single(params)
                results[str(params)] = result
            except Exception as e:
                logger.warning(f"Failed for {params}: {e}")
                continue

        return results

    def _compute_returns(
        self,
        weights: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute portfolio returns including funding.

        Args:
            weights: DataFrame of weights (use lagged for proper timing)

        Returns:
            Tuple of (total_returns, funding_pnl, price_pnl)
        """
        # Use previous period weights for current period returns
        lagged_weights = weights.shift(1)

        # Price returns
        price_pnl = (lagged_weights * self.returns).sum(axis=1)

        # Funding PnL: long pays funding, short receives funding
        # funding > 0: longs pay shorts
        # weight > 0: long position -> pays funding
        # weight < 0: short position -> receives funding
        funding_pnl = -(lagged_weights * self.funding).sum(axis=1)

        # Total returns
        total_returns = price_pnl + funding_pnl

        return total_returns, funding_pnl, price_pnl

    def _compute_transaction_costs(self, weights: pd.DataFrame) -> pd.Series:
        """
        Compute transaction costs from weight changes.

        Args:
            weights: DataFrame of weights

        Returns:
            Series of transaction costs per period
        """
        weight_changes = weights.diff().abs()
        turnover = weight_changes.sum(axis=1)

        # Cost = turnover * cost per side (half of round-trip)
        costs = turnover * TOTAL_COST_PER_SIDE

        return costs


def create_results_summary(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """
    Create summary DataFrame from backtest results.

    Args:
        results: Dict of BacktestResult objects

    Returns:
        DataFrame with metrics for each parameter set
    """
    summary = []

    for name, result in results.items():
        row = {
            "name": name,
            "funding_lookback_days": result.params.funding_lookback_days,
            "momentum_lookback_days": result.params.momentum_lookback_days,
            "sharpe": result.metrics.sharpe,
            "ann_return": result.metrics.ann_return,
            "ann_vol": result.metrics.ann_vol,
            "max_drawdown": result.metrics.max_drawdown,
            "calmar": result.metrics.calmar,
            "win_rate": result.metrics.win_rate,
            "turnover": result.metrics.turnover,
            "avg_positions": result.metrics.avg_positions,
            "total_return": result.metrics.total_return,
        }
        summary.append(row)

    df = pd.DataFrame(summary)
    df = df.sort_values("sharpe", ascending=False)

    return df


def create_sharpe_heatmap(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """
    Create Sharpe ratio heatmap (funding x momentum).

    Args:
        results: Dict of BacktestResult objects

    Returns:
        DataFrame with funding rows and momentum columns
    """
    data = []
    for result in results.values():
        data.append({
            "funding": result.params.funding_lookback_days,
            "momentum": result.params.momentum_lookback_days,
            "sharpe": result.metrics.sharpe,
        })

    df = pd.DataFrame(data)
    heatmap = df.pivot(index="funding", columns="momentum", values="sharpe")
    heatmap = heatmap.sort_index(ascending=True)

    return heatmap


def run_backtest(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    funding: pd.DataFrame,
    volumes: pd.DataFrame,
    grid: ParameterGrid | None = None,
    config: BacktestConfig | None = None,
) -> tuple[dict[str, BacktestResult], pd.DataFrame]:
    """
    Convenience function to run full backtest grid.

    Args:
        prices: DataFrame of prices
        returns: DataFrame of returns
        funding: DataFrame of funding rates
        volumes: DataFrame of volumes
        grid: ParameterGrid to test
        config: BacktestConfig object

    Returns:
        Tuple of (results dict, summary DataFrame)
    """
    engine = BacktestEngine(
        prices=prices,
        returns=returns,
        funding=funding,
        volumes=volumes,
        config=config,
    )

    results = engine.run_grid(grid)
    summary = create_results_summary(results)

    return results, summary

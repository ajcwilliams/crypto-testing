"""
Performance metrics for backtest evaluation.

Computes standard portfolio performance statistics:
- Sharpe ratio
- Max drawdown
- Calmar ratio
- Win rate
- Turnover
- PnL decomposition
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# 8-hour periods per year
PERIODS_PER_YEAR = 365 * 3


@dataclass
class BacktestMetrics:
    """Container for backtest performance metrics."""

    # Returns
    total_return: float
    ann_return: float
    ann_vol: float

    # Risk-adjusted
    sharpe: float
    sortino: float
    calmar: float

    # Drawdown
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int  # periods

    # Win/Loss
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float

    # Trading
    turnover: float  # Average per period
    avg_positions: float

    # Time
    start_date: str
    end_date: str
    n_periods: int


def compute_returns_metrics(returns: pd.Series) -> dict:
    """
    Compute return-based metrics.

    Args:
        returns: Series of period returns

    Returns:
        Dict of metrics
    """
    total_return = (1 + returns).prod() - 1
    ann_return = returns.mean() * PERIODS_PER_YEAR
    ann_vol = returns.std() * np.sqrt(PERIODS_PER_YEAR)

    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Sortino (downside deviation)
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(PERIODS_PER_YEAR) if len(downside) > 0 else np.nan
    sortino = ann_return / downside_vol if downside_vol and downside_vol > 0 else np.nan

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
    }


def compute_drawdown_metrics(returns: pd.Series) -> dict:
    """
    Compute drawdown metrics.

    Args:
        returns: Series of period returns

    Returns:
        Dict of drawdown metrics
    """
    # Cumulative returns
    cum_returns = (1 + returns).cumprod()

    # Running max
    running_max = cum_returns.expanding().max()

    # Drawdown
    drawdown = (cum_returns - running_max) / running_max

    max_drawdown = drawdown.min()
    avg_drawdown = drawdown.mean()

    # Max drawdown duration
    is_in_drawdown = drawdown < 0
    drawdown_groups = (~is_in_drawdown).cumsum()
    drawdown_durations = is_in_drawdown.groupby(drawdown_groups).sum()
    max_duration = drawdown_durations.max() if len(drawdown_durations) > 0 else 0

    # Calmar ratio
    calmar = returns.mean() * PERIODS_PER_YEAR / abs(max_drawdown) if max_drawdown < 0 else np.nan

    return {
        "max_drawdown": max_drawdown,
        "avg_drawdown": avg_drawdown,
        "max_drawdown_duration": int(max_duration),
        "calmar": calmar,
        "drawdown_series": drawdown,
    }


def compute_win_loss_metrics(returns: pd.Series) -> dict:
    """
    Compute win/loss statistics.

    Args:
        returns: Series of period returns

    Returns:
        Dict of win/loss metrics
    """
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0

    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0

    # Profit factor
    total_gains = wins.sum() if len(wins) > 0 else 0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = total_gains / total_losses if total_losses > 0 else np.inf

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def compute_trading_metrics(weights: pd.DataFrame) -> dict:
    """
    Compute trading/turnover metrics.

    Args:
        weights: DataFrame of portfolio weights

    Returns:
        Dict of trading metrics
    """
    # Turnover: sum of absolute weight changes
    weight_changes = weights.diff().abs()
    turnover_per_period = weight_changes.sum(axis=1)
    avg_turnover = turnover_per_period.mean()

    # Average number of positions
    n_positions = (weights != 0).sum(axis=1)
    avg_positions = n_positions.mean()

    return {
        "turnover": avg_turnover,
        "avg_positions": avg_positions,
        "turnover_series": turnover_per_period,
    }


def compute_all_metrics(
    returns: pd.Series,
    weights: pd.DataFrame,
) -> BacktestMetrics:
    """
    Compute all backtest metrics.

    Args:
        returns: Series of portfolio returns
        weights: DataFrame of portfolio weights

    Returns:
        BacktestMetrics object
    """
    returns_m = compute_returns_metrics(returns)
    dd_m = compute_drawdown_metrics(returns)
    wl_m = compute_win_loss_metrics(returns)
    trade_m = compute_trading_metrics(weights)

    return BacktestMetrics(
        total_return=returns_m["total_return"],
        ann_return=returns_m["ann_return"],
        ann_vol=returns_m["ann_vol"],
        sharpe=returns_m["sharpe"],
        sortino=returns_m["sortino"],
        calmar=dd_m["calmar"],
        max_drawdown=dd_m["max_drawdown"],
        avg_drawdown=dd_m["avg_drawdown"],
        max_drawdown_duration=dd_m["max_drawdown_duration"],
        win_rate=wl_m["win_rate"],
        profit_factor=wl_m["profit_factor"],
        avg_win=wl_m["avg_win"],
        avg_loss=wl_m["avg_loss"],
        turnover=trade_m["turnover"],
        avg_positions=trade_m["avg_positions"],
        start_date=str(returns.index.min()),
        end_date=str(returns.index.max()),
        n_periods=len(returns),
    )


def compute_rolling_sharpe(
    returns: pd.Series,
    window_periods: int = 180,  # ~60 days
) -> pd.Series:
    """
    Compute rolling Sharpe ratio.

    Args:
        returns: Series of returns
        window_periods: Rolling window size

    Returns:
        Series of rolling Sharpe ratios
    """
    rolling_mean = returns.rolling(window_periods).mean()
    rolling_std = returns.rolling(window_periods).std()

    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(PERIODS_PER_YEAR)

    return rolling_sharpe


def compute_monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """
    Compute monthly returns table.

    Args:
        returns: Series of returns (DatetimeIndex)

    Returns:
        DataFrame with year rows and month columns
    """
    # Resample to monthly
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    # Create pivot table
    monthly_df = monthly.to_frame("return")
    monthly_df["year"] = monthly_df.index.year
    monthly_df["month"] = monthly_df.index.month

    pivot = monthly_df.pivot(index="year", columns="month", values="return")
    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ][: len(pivot.columns)]

    # Add yearly total
    yearly = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    pivot["Year"] = yearly.values[: len(pivot)]

    return pivot


def compute_pnl_decomposition(
    returns: pd.Series,
    funding_pnl: pd.Series,
) -> dict:
    """
    Decompose PnL into carry (funding) and price components.

    Args:
        returns: Series of total returns
        funding_pnl: Series of funding PnL

    Returns:
        Dict with decomposition
    """
    price_pnl = returns - funding_pnl

    return {
        "total_pnl": returns.sum(),
        "funding_pnl": funding_pnl.sum(),
        "price_pnl": price_pnl.sum(),
        "funding_pct": funding_pnl.sum() / returns.sum() if returns.sum() != 0 else 0,
        "price_pct": price_pnl.sum() / returns.sum() if returns.sum() != 0 else 0,
    }

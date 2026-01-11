"""
Signal combiner for double-sort strategy.

Combines funding (carry) and momentum signals using 5x5 quintile sort.
Positions are assigned based on corner cells:
- LONG: High momentum (Q5) + Low funding (Q1) - strong trend, cheap carry
- SHORT: Low momentum (Q1) + High funding (Q5) - weak trend, expensive carry
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from src.signals.funding_rate import (
    assign_funding_quintile,
    compute_funding_signal,
)
from src.signals.momentum import (
    assign_momentum_quintile,
    compute_momentum_signal,
)


class Position(Enum):
    """Position direction."""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


@dataclass
class DoubleSortConfig:
    """Configuration for double-sort strategy."""
    funding_lookback: int  # 8h periods
    momentum_lookback: int  # 8h periods
    long_funding_quintile: int = 5  # Low funding = Q5 (high rank with ascending=True)
    long_momentum_quintile: int = 5  # High momentum = Q5
    short_funding_quintile: int = 1  # High funding = Q1
    short_momentum_quintile: int = 1  # Low momentum = Q1


def compute_double_sort_positions(
    funding_quintiles: pd.DataFrame,
    momentum_quintiles: pd.DataFrame,
    config: DoubleSortConfig,
) -> pd.DataFrame:
    """
    Assign positions based on double-sort quintiles.

    Args:
        funding_quintiles: DataFrame of funding quintiles (1-5)
        momentum_quintiles: DataFrame of momentum quintiles (1-5)
        config: DoubleSortConfig with quintile thresholds

    Returns:
        DataFrame of positions (-1, 0, 1)
    """
    positions = pd.DataFrame(
        0,
        index=funding_quintiles.index,
        columns=funding_quintiles.columns,
    )

    # LONG: High momentum AND low funding (good carry)
    long_mask = (
        (momentum_quintiles == config.long_momentum_quintile) &
        (funding_quintiles == config.long_funding_quintile)
    )
    positions[long_mask] = Position.LONG.value

    # SHORT: Low momentum AND high funding (bad carry)
    short_mask = (
        (momentum_quintiles == config.short_momentum_quintile) &
        (funding_quintiles == config.short_funding_quintile)
    )
    positions[short_mask] = Position.SHORT.value

    return positions


def compute_double_sort_signals(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    funding: pd.DataFrame,
    funding_lookback: int,
    momentum_lookback: int,
    risk_adjusted: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute double-sort signals and positions.

    Args:
        prices: DataFrame of prices
        returns: DataFrame of returns
        funding: DataFrame of funding rates
        funding_lookback: Lookback for funding signal (8h periods)
        momentum_lookback: Lookback for momentum signal (8h periods)
        risk_adjusted: Use risk-adjusted momentum

    Returns:
        Tuple of (funding_quintiles, momentum_quintiles, positions)
    """
    # Compute signals
    funding_signal = compute_funding_signal(funding, funding_lookback)
    momentum_signal = compute_momentum_signal(
        prices, returns, momentum_lookback, risk_adjusted=risk_adjusted
    )

    # Assign quintiles
    funding_quintiles = assign_funding_quintile(funding_signal, ascending=True)
    momentum_quintiles = assign_momentum_quintile(momentum_signal, ascending=False)

    # Compute positions
    config = DoubleSortConfig(
        funding_lookback=funding_lookback,
        momentum_lookback=momentum_lookback,
    )
    positions = compute_double_sort_positions(
        funding_quintiles, momentum_quintiles, config
    )

    return funding_quintiles, momentum_quintiles, positions


def get_long_short_coins(
    positions: pd.DataFrame,
    timestamp: pd.Timestamp,
) -> tuple[list[str], list[str]]:
    """
    Get lists of long and short coins at a specific timestamp.

    Args:
        positions: DataFrame of positions
        timestamp: Timestamp to check

    Returns:
        Tuple of (long_coins, short_coins)
    """
    row = positions.loc[timestamp]

    long_coins = row[row == Position.LONG.value].index.tolist()
    short_coins = row[row == Position.SHORT.value].index.tolist()

    return long_coins, short_coins


def count_positions(positions: pd.DataFrame) -> pd.DataFrame:
    """
    Count number of long, short, and total positions at each timestamp.

    Args:
        positions: DataFrame of positions

    Returns:
        DataFrame with columns: n_long, n_short, n_total
    """
    n_long = (positions == Position.LONG.value).sum(axis=1)
    n_short = (positions == Position.SHORT.value).sum(axis=1)
    n_total = n_long + n_short

    return pd.DataFrame({
        "n_long": n_long,
        "n_short": n_short,
        "n_total": n_total,
    })


def compute_position_grid(
    funding_quintiles: pd.DataFrame,
    momentum_quintiles: pd.DataFrame,
) -> dict[tuple[int, int], pd.DataFrame]:
    """
    Create position masks for each cell in the 5x5 grid.

    Args:
        funding_quintiles: DataFrame of funding quintiles
        momentum_quintiles: DataFrame of momentum quintiles

    Returns:
        Dict mapping (funding_q, momentum_q) -> mask DataFrame
    """
    grid = {}

    for fq in range(1, 6):
        for mq in range(1, 6):
            mask = (funding_quintiles == fq) & (momentum_quintiles == mq)
            grid[(fq, mq)] = mask

    return grid


def analyze_grid_returns(
    grid: dict[tuple[int, int], pd.DataFrame],
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute average returns for each cell in the 5x5 grid.

    Args:
        grid: Dict from compute_position_grid
        returns: DataFrame of returns

    Returns:
        DataFrame with funding quintile rows and momentum quintile columns
    """
    results = np.zeros((5, 5))

    for fq in range(1, 6):
        for mq in range(1, 6):
            mask = grid[(fq, mq)]
            cell_returns = returns.where(mask)
            results[fq - 1, mq - 1] = cell_returns.mean().mean()

    return pd.DataFrame(
        results,
        index=[f"F{q}" for q in range(1, 6)],
        columns=[f"M{q}" for q in range(1, 6)],
    )

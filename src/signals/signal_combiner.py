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


def compute_adaptive_positions(
    funding_signal: pd.DataFrame,
    momentum_signal: pd.DataFrame,
    long_pct: float = 0.2,
    short_pct: float = 0.2,
) -> pd.DataFrame:
    """
    Adaptive position assignment using percentile ranks.
    Works better for small universes.

    Long: Top long_pct of momentum AND bottom long_pct of funding
    Short: Bottom short_pct of momentum AND top short_pct of funding

    Args:
        funding_signal: DataFrame of funding signals (raw, not quintiled)
        momentum_signal: DataFrame of momentum signals (raw, not quintiled)
        long_pct: Top/bottom percentile for long positions
        short_pct: Top/bottom percentile for short positions

    Returns:
        DataFrame of positions (-1, 0, 1)
    """
    # Rank signals cross-sectionally (0 to 1)
    funding_rank = funding_signal.rank(axis=1, pct=True)  # Lower = better carry
    momentum_rank = momentum_signal.rank(axis=1, pct=True)  # Higher = better momentum

    positions = pd.DataFrame(
        0,
        index=funding_signal.index,
        columns=funding_signal.columns,
    )

    # LONG: High momentum (top pct) AND low funding (bottom pct)
    long_mask = (momentum_rank >= (1 - long_pct)) & (funding_rank <= long_pct)
    positions[long_mask] = Position.LONG.value

    # SHORT: Low momentum (bottom pct) AND high funding (top pct)
    short_mask = (momentum_rank <= short_pct) & (funding_rank >= (1 - short_pct))
    positions[short_mask] = Position.SHORT.value

    return positions


def compute_double_sort_signals(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    funding: pd.DataFrame,
    funding_lookback: int,
    momentum_lookback: int,
    risk_adjusted: bool = True,
    use_composite: bool = True,
    long_pct: float = 0.2,
    short_pct: float = 0.2,
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
        use_composite: If True, use composite ranking instead of quintile corners
        long_pct: Top percentile for long positions (default 20%)
        short_pct: Bottom percentile for short positions (default 20%)

    Returns:
        Tuple of (funding_signal, momentum_signal, positions)
    """
    # Compute signals
    funding_signal = compute_funding_signal(funding, funding_lookback)
    momentum_signal = compute_momentum_signal(
        prices, returns, momentum_lookback, risk_adjusted=risk_adjusted
    )

    if use_composite:
        # Use composite ranking approach
        positions = compute_composite_positions(
            funding_signal, momentum_signal,
            long_pct=long_pct, short_pct=short_pct
        )
    else:
        # Legacy quintile corners approach
        funding_quintiles = assign_funding_quintile(funding_signal, ascending=True)
        momentum_quintiles = assign_momentum_quintile(momentum_signal, ascending=False)
        config = DoubleSortConfig(
            funding_lookback=funding_lookback,
            momentum_lookback=momentum_lookback,
        )
        positions = compute_double_sort_positions(
            funding_quintiles, momentum_quintiles, config
        )

    return funding_signal, momentum_signal, positions


def compute_composite_positions(
    funding_signal: pd.DataFrame,
    momentum_signal: pd.DataFrame,
    long_pct: float = 0.2,
    short_pct: float = 0.2,
) -> pd.DataFrame:
    """
    Compute positions using composite ranking across both signals.

    Composite score = average of:
    - Funding rank (lower funding = higher rank, better for longs)
    - Momentum rank (higher momentum = higher rank)

    Long: Top long_pct by composite score
    Short: Bottom short_pct by composite score

    Args:
        funding_signal: DataFrame of funding signals
        momentum_signal: DataFrame of momentum signals
        long_pct: Top percentile for long positions
        short_pct: Bottom percentile for short positions

    Returns:
        DataFrame of positions (-1, 0, 1)
    """
    # Rank signals cross-sectionally (0 to 1)
    # For funding: lower is better, so we invert (1 - rank)
    funding_rank = 1 - funding_signal.rank(axis=1, pct=True)
    # For momentum: higher is better
    momentum_rank = momentum_signal.rank(axis=1, pct=True)

    # Composite score = average of ranks
    composite = (funding_rank + momentum_rank) / 2

    # Rank the composite score
    composite_rank = composite.rank(axis=1, pct=True)

    positions = pd.DataFrame(
        0,
        index=funding_signal.index,
        columns=funding_signal.columns,
    )

    # LONG: Top composite scores (high momentum + low funding)
    long_mask = composite_rank >= (1 - long_pct)
    positions[long_mask] = Position.LONG.value

    # SHORT: Bottom composite scores (low momentum + high funding)
    short_mask = composite_rank <= short_pct
    positions[short_mask] = Position.SHORT.value

    return positions


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

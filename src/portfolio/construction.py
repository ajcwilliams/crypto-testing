"""
Portfolio construction module.

Handles:
- Universe filtering (top 25% by market cap, exclude stablecoins)
- Position sizing (equal weight within long/short legs)
- Dollar-neutral portfolio construction
"""

import numpy as np
import pandas as pd

from src.signals.signal_combiner import Position


def filter_universe(
    volumes: pd.DataFrame,
    percentile: float = 0.25,
    min_history: int = 60,  # Minimum 60 periods (~20 days) of history
) -> pd.DataFrame:
    """
    Filter universe to top coins by volume.

    Args:
        volumes: DataFrame of 24h volumes (periods x coins)
        percentile: Top percentile to include (0.25 = top 25%)
        min_history: Minimum periods of history required

    Returns:
        Boolean DataFrame indicating which coins are in universe
    """

    def filter_at_time(row: pd.Series, historical_count: pd.Series) -> pd.Series:
        """Filter coins for a single timestamp."""
        # Exclude coins with insufficient history
        valid = row.notna() & (historical_count >= min_history)
        valid_volumes = row[valid]

        if len(valid_volumes) < 5:
            return pd.Series(False, index=row.index)

        # Get threshold for top percentile
        threshold = valid_volumes.quantile(1 - percentile)

        # Create mask
        in_universe = (row >= threshold) & valid
        return in_universe.reindex(row.index, fill_value=False)

    # Count historical observations for each coin
    historical_count = volumes.notna().cumsum()

    # Apply filter at each timestamp
    universe = volumes.apply(
        lambda row: filter_at_time(row, historical_count.loc[row.name]),
        axis=1,
    )

    return universe


def construct_weights(
    positions: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    equal_weight: bool = True,
) -> pd.DataFrame:
    """
    Construct portfolio weights from positions.

    Args:
        positions: DataFrame of positions (-1, 0, 1)
        universe: Optional boolean DataFrame of universe membership
        equal_weight: If True, equal weight within each leg

    Returns:
        DataFrame of weights (sum of abs weights = 1 at each timestamp)
    """
    # Apply universe filter if provided
    if universe is not None:
        positions = positions.where(universe, 0)

    # Count longs and shorts
    n_long = (positions == Position.LONG.value).sum(axis=1)
    n_short = (positions == Position.SHORT.value).sum(axis=1)

    if equal_weight:
        # Equal weight within each leg
        # Total exposure: 0.5 long, 0.5 short (dollar neutral)
        long_weight = 0.5 / n_long.replace(0, np.nan)
        short_weight = -0.5 / n_short.replace(0, np.nan)

        weights = positions.copy().astype(float)
        weights[positions == Position.LONG.value] = weights[
            positions == Position.LONG.value
        ].apply(lambda x: long_weight, axis=1)
        weights[positions == Position.SHORT.value] = weights[
            positions == Position.SHORT.value
        ].apply(lambda x: short_weight, axis=1)
        weights[positions == Position.NEUTRAL.value] = 0

        # Vectorized approach
        weights = pd.DataFrame(0.0, index=positions.index, columns=positions.columns)

        for col in positions.columns:
            is_long = positions[col] == Position.LONG.value
            is_short = positions[col] == Position.SHORT.value

            weights.loc[is_long, col] = 0.5 / n_long.loc[is_long]
            weights.loc[is_short, col] = -0.5 / n_short.loc[is_short]

    else:
        # Just use positions as weights (will be normalized by vol targeting)
        weights = positions.astype(float)

    # Fill NaN with 0
    weights = weights.fillna(0)

    return weights


def compute_gross_exposure(weights: pd.DataFrame) -> pd.Series:
    """Compute gross exposure (sum of absolute weights)."""
    return weights.abs().sum(axis=1)


def compute_net_exposure(weights: pd.DataFrame) -> pd.Series:
    """Compute net exposure (sum of weights)."""
    return weights.sum(axis=1)


def normalize_weights(
    weights: pd.DataFrame,
    target_gross_exposure: float = 1.0,
) -> pd.DataFrame:
    """
    Normalize weights to target gross exposure.

    Args:
        weights: DataFrame of weights
        target_gross_exposure: Target sum of absolute weights

    Returns:
        Normalized weights DataFrame
    """
    gross = compute_gross_exposure(weights)
    scale = target_gross_exposure / gross.replace(0, np.nan)

    return weights.multiply(scale, axis=0).fillna(0)


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """
    Compute turnover (sum of absolute weight changes).

    Args:
        weights: DataFrame of weights

    Returns:
        Series of turnover at each rebalance
    """
    weight_changes = weights.diff().abs()
    return weight_changes.sum(axis=1)


def compute_position_count(weights: pd.DataFrame) -> pd.DataFrame:
    """
    Count active positions.

    Args:
        weights: DataFrame of weights

    Returns:
        DataFrame with n_long, n_short, n_total columns
    """
    n_long = (weights > 0).sum(axis=1)
    n_short = (weights < 0).sum(axis=1)

    return pd.DataFrame({
        "n_long": n_long,
        "n_short": n_short,
        "n_total": n_long + n_short,
    })

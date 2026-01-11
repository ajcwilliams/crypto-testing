"""
Funding rate signal for carry strategy.

Lower funding = cheaper to hold long = positive carry signal
Higher funding = expensive to hold long = negative carry signal (better to short)
"""

import numpy as np
import pandas as pd


def compute_funding_signal(
    funding: pd.DataFrame,
    lookback_periods: int,
) -> pd.DataFrame:
    """
    Compute funding rate signal (carry).

    Args:
        funding: DataFrame of funding rates (periods x coins)
        lookback_periods: Number of 8h periods for rolling average

    Returns:
        DataFrame of funding signals (periods x coins)
        Lower values = better carry for longs
    """
    # Forward-fill NaN funding rates
    funding_filled = funding.ffill()

    # Rolling mean of funding rates
    signal = funding_filled.rolling(lookback_periods, min_periods=1).mean()

    return signal


def rank_funding_signal(
    signal: pd.DataFrame,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Rank funding signal cross-sectionally.

    Args:
        signal: DataFrame of funding signals (periods x coins)
        ascending: If True, lower funding gets higher rank (better for longs)

    Returns:
        DataFrame of ranks (0 to 1, with 1 being best)
    """
    # Rank cross-sectionally at each timestamp
    # For funding: lower is better for longs, so ascending=True
    ranks = signal.rank(axis=1, ascending=ascending, pct=True)

    return ranks


def assign_funding_quintile(
    signal: pd.DataFrame,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Assign funding signal to quintiles cross-sectionally.

    Args:
        signal: DataFrame of funding signals (periods x coins)
        ascending: If True, lower funding gets quintile 5 (best for longs)

    Returns:
        DataFrame of quintiles (1-5, with 5 being best for longs)
    """

    def quintile_at_time(row: pd.Series) -> pd.Series:
        """Assign quintile for a single timestamp."""
        valid = row.dropna()
        if len(valid) < 5:
            # Not enough data for quintiles
            return pd.Series(np.nan, index=row.index)

        # pd.qcut with duplicates='drop' can cause issues, use rank-based approach
        ranks = valid.rank(ascending=ascending, pct=True)
        quintiles = pd.cut(
            ranks,
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=[1, 2, 3, 4, 5],
            include_lowest=True,
        ).astype(float)

        return quintiles.reindex(row.index)

    quintiles = signal.apply(quintile_at_time, axis=1)

    return quintiles


def compute_all_funding_signals(
    funding: pd.DataFrame,
    lookbacks: list[int],
) -> dict[int, pd.DataFrame]:
    """
    Compute funding signals for multiple lookback periods.

    Args:
        funding: DataFrame of funding rates
        lookbacks: List of lookback periods (in 8h periods)

    Returns:
        Dict mapping lookback -> signal DataFrame
    """
    signals = {}
    for lb in lookbacks:
        signals[lb] = compute_funding_signal(funding, lb)
    return signals


def compute_all_funding_quintiles(
    funding: pd.DataFrame,
    lookbacks: list[int],
) -> dict[int, pd.DataFrame]:
    """
    Compute funding quintiles for multiple lookback periods.

    Args:
        funding: DataFrame of funding rates
        lookbacks: List of lookback periods (in 8h periods)

    Returns:
        Dict mapping lookback -> quintile DataFrame
    """
    quintiles = {}
    for lb in lookbacks:
        signal = compute_funding_signal(funding, lb)
        quintiles[lb] = assign_funding_quintile(signal, ascending=True)
    return quintiles

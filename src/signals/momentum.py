"""
Momentum signal for trend-following strategy.

Uses risk-adjusted momentum (return / volatility) for better signal quality.
Higher momentum = stronger trend = go long
Lower momentum = weaker trend = go short
"""

import numpy as np
import pandas as pd

# 8-hour periods per year
PERIODS_PER_YEAR = 365 * 3


def compute_returns(
    prices: pd.DataFrame,
    lookback_periods: int,
) -> pd.DataFrame:
    """
    Compute log returns over lookback period.

    Args:
        prices: DataFrame of prices (periods x coins)
        lookback_periods: Number of 8h periods

    Returns:
        DataFrame of log returns
    """
    return np.log(prices / prices.shift(lookback_periods))


def compute_volatility(
    returns: pd.DataFrame,
    lookback_periods: int,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling volatility.

    Args:
        returns: DataFrame of period returns (periods x coins)
        lookback_periods: Number of 8h periods for rolling window
        annualize: If True, annualize the volatility

    Returns:
        DataFrame of volatilities
    """
    # Use period returns (not cumulative) for vol calculation
    vol = returns.rolling(lookback_periods, min_periods=max(1, lookback_periods // 2)).std()

    if annualize:
        vol = vol * np.sqrt(PERIODS_PER_YEAR)

    return vol


def compute_momentum_signal(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    lookback_periods: int,
    risk_adjusted: bool = True,
) -> pd.DataFrame:
    """
    Compute momentum signal.

    Args:
        prices: DataFrame of prices (periods x coins)
        returns: DataFrame of period returns (periods x coins)
        lookback_periods: Number of 8h periods
        risk_adjusted: If True, divide return by volatility (Sharpe-like)

    Returns:
        DataFrame of momentum signals
    """
    # Cumulative return over lookback
    cum_return = compute_returns(prices, lookback_periods)

    if not risk_adjusted:
        return cum_return

    # Volatility over same period
    vol = compute_volatility(returns, lookback_periods, annualize=False)

    # Risk-adjusted momentum (avoid division by zero)
    vol_safe = vol.replace(0, np.nan)
    signal = cum_return / vol_safe

    return signal


def rank_momentum_signal(
    signal: pd.DataFrame,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Rank momentum signal cross-sectionally.

    Args:
        signal: DataFrame of momentum signals (periods x coins)
        ascending: If False, higher momentum gets higher rank

    Returns:
        DataFrame of ranks (0 to 1, with 1 being best)
    """
    ranks = signal.rank(axis=1, ascending=ascending, pct=True)
    return ranks


def assign_momentum_quintile(
    signal: pd.DataFrame,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Assign momentum signal to quintiles cross-sectionally.

    Args:
        signal: DataFrame of momentum signals (periods x coins)
        ascending: If False, higher momentum gets quintile 5 (best)

    Returns:
        DataFrame of quintiles (1-5, with 5 being best momentum)
    """

    def quintile_at_time(row: pd.Series) -> pd.Series:
        """Assign quintile for a single timestamp."""
        valid = row.dropna()
        if len(valid) < 5:
            return pd.Series(np.nan, index=row.index)

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


def compute_all_momentum_signals(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    lookbacks: list[int],
    risk_adjusted: bool = True,
) -> dict[int, pd.DataFrame]:
    """
    Compute momentum signals for multiple lookback periods.

    Args:
        prices: DataFrame of prices
        returns: DataFrame of period returns
        lookbacks: List of lookback periods (in 8h periods)
        risk_adjusted: If True, use risk-adjusted momentum

    Returns:
        Dict mapping lookback -> signal DataFrame
    """
    signals = {}
    for lb in lookbacks:
        signals[lb] = compute_momentum_signal(
            prices, returns, lb, risk_adjusted=risk_adjusted
        )
    return signals


def compute_all_momentum_quintiles(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    lookbacks: list[int],
    risk_adjusted: bool = True,
) -> dict[int, pd.DataFrame]:
    """
    Compute momentum quintiles for multiple lookback periods.

    Args:
        prices: DataFrame of prices
        returns: DataFrame of period returns
        lookbacks: List of lookback periods (in 8h periods)
        risk_adjusted: If True, use risk-adjusted momentum

    Returns:
        Dict mapping lookback -> quintile DataFrame
    """
    quintiles = {}
    for lb in lookbacks:
        signal = compute_momentum_signal(
            prices, returns, lb, risk_adjusted=risk_adjusted
        )
        quintiles[lb] = assign_momentum_quintile(signal, ascending=False)
    return quintiles

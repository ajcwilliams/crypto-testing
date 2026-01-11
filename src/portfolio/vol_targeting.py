"""
Volatility targeting module.

Scales portfolio weights to achieve target volatility (35% annualized).
Uses recent realized volatility to determine scaling factor.
"""

import numpy as np
import pandas as pd

# 8-hour periods per year
PERIODS_PER_YEAR = 365 * 3

# Default target volatility
DEFAULT_TARGET_VOL = 0.35  # 35% annualized


def compute_portfolio_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.Series:
    """
    Compute portfolio returns from weights and asset returns.

    Args:
        weights: DataFrame of weights (lagged by 1 period for proper timing)
        returns: DataFrame of asset returns

    Returns:
        Series of portfolio returns
    """
    # Align weights and returns
    weights_aligned, returns_aligned = weights.align(returns, join="inner")

    # Portfolio return = sum of (weight * return) for each asset
    port_returns = (weights_aligned.shift(1) * returns_aligned).sum(axis=1)

    return port_returns


def compute_realized_vol(
    returns: pd.Series,
    lookback_periods: int = 60,  # ~20 days
    annualize: bool = True,
) -> pd.Series:
    """
    Compute rolling realized volatility.

    Args:
        returns: Series of returns
        lookback_periods: Number of periods for rolling window
        annualize: If True, annualize the volatility

    Returns:
        Series of realized volatility
    """
    vol = returns.rolling(lookback_periods, min_periods=lookback_periods // 2).std()

    if annualize:
        vol = vol * np.sqrt(PERIODS_PER_YEAR)

    return vol


def compute_vol_scale_factor(
    realized_vol: pd.Series,
    target_vol: float = DEFAULT_TARGET_VOL,
    max_leverage: float = 3.0,
    min_leverage: float = 0.1,
) -> pd.Series:
    """
    Compute scaling factor to achieve target volatility.

    Args:
        realized_vol: Series of realized volatility (annualized)
        target_vol: Target annualized volatility
        max_leverage: Maximum allowed leverage (cap)
        min_leverage: Minimum leverage (floor)

    Returns:
        Series of scale factors
    """
    # Scale = target / realized
    scale = target_vol / realized_vol.replace(0, np.nan)

    # Apply caps
    scale = scale.clip(lower=min_leverage, upper=max_leverage)

    # Forward fill NaN (use last valid scale)
    scale = scale.ffill().fillna(1.0)

    return scale


def apply_vol_targeting(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    target_vol: float = DEFAULT_TARGET_VOL,
    lookback_periods: int = 60,
    max_leverage: float = 3.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply volatility targeting to portfolio weights.

    Args:
        weights: DataFrame of raw weights
        returns: DataFrame of asset returns
        target_vol: Target annualized volatility
        lookback_periods: Periods for realized vol calculation
        max_leverage: Maximum allowed leverage

    Returns:
        Tuple of (scaled_weights, scale_factors)
    """
    # Compute portfolio returns with current weights
    port_returns = compute_portfolio_returns(weights, returns)

    # Compute realized volatility
    realized_vol = compute_realized_vol(port_returns, lookback_periods)

    # Compute scale factor
    scale = compute_vol_scale_factor(
        realized_vol,
        target_vol=target_vol,
        max_leverage=max_leverage,
    )

    # Apply scale to weights
    scaled_weights = weights.multiply(scale, axis=0)

    return scaled_weights, scale


def estimate_ex_ante_vol(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    lookback_periods: int = 60,
) -> pd.Series:
    """
    Estimate ex-ante portfolio volatility using covariance.

    Args:
        weights: DataFrame of weights
        returns: DataFrame of asset returns
        lookback_periods: Periods for covariance estimation

    Returns:
        Series of estimated annualized volatility
    """

    def compute_vol_at_time(idx: int) -> float:
        """Compute vol for a single timestamp using historical returns."""
        if idx < lookback_periods:
            return np.nan

        # Get historical returns window
        hist_returns = returns.iloc[idx - lookback_periods : idx]
        w = weights.iloc[idx]

        # Only use coins with valid weights and returns
        valid_coins = w[w != 0].index
        valid_coins = valid_coins.intersection(hist_returns.columns)

        if len(valid_coins) < 2:
            return np.nan

        # Compute covariance matrix
        cov = hist_returns[valid_coins].cov()
        w_valid = w[valid_coins]

        # Portfolio variance = w' * Cov * w
        port_var = w_valid @ cov @ w_valid

        # Annualize
        return np.sqrt(port_var * PERIODS_PER_YEAR)

    # Compute for each timestamp (expensive, use sparingly)
    vols = [compute_vol_at_time(i) for i in range(len(weights))]

    return pd.Series(vols, index=weights.index)


def compute_target_vol_stats(
    scaled_weights: pd.DataFrame,
    returns: pd.DataFrame,
    target_vol: float,
) -> dict:
    """
    Compute statistics about vol targeting performance.

    Args:
        scaled_weights: DataFrame of vol-targeted weights
        returns: DataFrame of asset returns
        target_vol: Target volatility

    Returns:
        Dict with vol targeting statistics
    """
    port_returns = compute_portfolio_returns(scaled_weights, returns)

    # Realized vol at different windows
    vol_20d = compute_realized_vol(port_returns, 60)  # 20 days
    vol_60d = compute_realized_vol(port_returns, 180)  # 60 days

    return {
        "target_vol": target_vol,
        "realized_vol_20d_mean": vol_20d.mean(),
        "realized_vol_20d_std": vol_20d.std(),
        "realized_vol_60d_mean": vol_60d.mean(),
        "realized_vol_60d_std": vol_60d.std(),
        "vol_ratio_20d": vol_20d.mean() / target_vol,
        "vol_ratio_60d": vol_60d.mean() / target_vol,
    }

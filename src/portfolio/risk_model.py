"""
EWM Risk Model for forward volatility prediction.

Uses cross-section of EWM standard deviations with multiple halflifes
to predict forward volatility and identify risk regimes.
"""

import numpy as np
import pandas as pd
from scipy import stats

# 8-hour periods per year
PERIODS_PER_YEAR = 365 * 3

# Default halflifes in days
DEFAULT_HALFLIFES = [2, 4, 8, 16, 32, 64]


def days_to_periods(days: int, periods_per_day: int = 3) -> int:
    """Convert days to 8-hour periods."""
    return days * periods_per_day


def compute_ewm_vol(
    returns: pd.DataFrame | pd.Series,
    halflife_periods: int,
    annualize: bool = True,
) -> pd.DataFrame | pd.Series:
    """
    Compute EWM volatility.

    Args:
        returns: DataFrame or Series of returns
        halflife_periods: Halflife in periods
        annualize: If True, annualize the volatility

    Returns:
        EWM volatility (same shape as input)
    """
    ewm_vol = returns.ewm(halflife=halflife_periods, min_periods=halflife_periods).std()

    if annualize:
        ewm_vol = ewm_vol * np.sqrt(PERIODS_PER_YEAR)

    return ewm_vol


def compute_ewm_vol_cross_section(
    returns: pd.DataFrame,
    halflifes_days: list[int] | None = None,
) -> dict[int, pd.DataFrame]:
    """
    Compute cross-section of EWM volatilities for each asset.

    Args:
        returns: DataFrame of returns (periods x coins)
        halflifes_days: List of halflife values in days

    Returns:
        Dict mapping halflife (days) -> DataFrame of EWM vols
    """
    if halflifes_days is None:
        halflifes_days = DEFAULT_HALFLIFES

    vols = {}
    for hl_days in halflifes_days:
        hl_periods = days_to_periods(hl_days)
        vols[hl_days] = compute_ewm_vol(returns, hl_periods, annualize=True)

    return vols


def compute_portfolio_ewm_vols(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    halflifes_days: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute portfolio-level EWM volatilities.

    Args:
        weights: DataFrame of portfolio weights
        returns: DataFrame of asset returns
        halflifes_days: List of halflife values in days

    Returns:
        DataFrame with columns for each halflife
    """
    if halflifes_days is None:
        halflifes_days = DEFAULT_HALFLIFES

    # Compute portfolio returns
    port_returns = (weights.shift(1) * returns).sum(axis=1)

    # Compute EWM vol for each halflife
    vols = {}
    for hl_days in halflifes_days:
        hl_periods = days_to_periods(hl_days)
        vols[f"ewm_{hl_days}d"] = compute_ewm_vol(port_returns, hl_periods, annualize=True)

    return pd.DataFrame(vols)


def compute_forward_realized_vol(
    returns: pd.Series,
    forward_periods: int = 60,  # ~20 days forward
) -> pd.Series:
    """
    Compute forward realized volatility (for prediction testing).

    Args:
        returns: Series of returns
        forward_periods: Number of periods to look forward

    Returns:
        Series of forward realized vol (shifted back to align with predictions)
    """
    # Rolling vol shifted back by forward_periods
    forward_vol = (
        returns.rolling(forward_periods, min_periods=forward_periods).std()
        * np.sqrt(PERIODS_PER_YEAR)
    )
    return forward_vol.shift(-forward_periods)


def evaluate_ewm_prediction(
    ewm_vols: pd.DataFrame,
    realized_vol: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate predictive power of each EWM volatility measure.

    Args:
        ewm_vols: DataFrame with EWM vol columns
        realized_vol: Series of forward realized volatility

    Returns:
        DataFrame with evaluation metrics for each EWM
    """
    results = []

    for col in ewm_vols.columns:
        # Align and drop NaN
        df = pd.DataFrame({"ewm": ewm_vols[col], "realized": realized_vol}).dropna()

        if len(df) < 30:
            continue

        # Correlation
        corr = df["ewm"].corr(df["realized"])

        # R-squared from linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df["ewm"], df["realized"]
        )

        # Mean absolute error
        mae = (df["ewm"] - df["realized"]).abs().mean()

        # Bias
        bias = (df["ewm"] - df["realized"]).mean()

        results.append({
            "ewm": col,
            "correlation": corr,
            "r_squared": r_value**2,
            "slope": slope,
            "intercept": intercept,
            "p_value": p_value,
            "mae": mae,
            "bias": bias,
        })

    return pd.DataFrame(results)


def compute_ensemble_vol_forecast(
    ewm_vols: pd.DataFrame,
    weights: pd.Series | None = None,
) -> pd.Series:
    """
    Compute ensemble volatility forecast from multiple EWM measures.

    Args:
        ewm_vols: DataFrame with EWM vol columns
        weights: Optional weights for each EWM (default: equal weight)

    Returns:
        Series of ensemble vol forecast
    """
    if weights is None:
        # Equal weight
        weights = pd.Series(1 / len(ewm_vols.columns), index=ewm_vols.columns)

    ensemble = (ewm_vols * weights).sum(axis=1)

    return ensemble


def compute_optimal_ewm_weights(
    ewm_vols: pd.DataFrame,
    realized_vol: pd.Series,
    lookback_periods: int = 180,  # ~60 days
) -> pd.DataFrame:
    """
    Compute optimal weights for EWM ensemble using rolling regression.

    Args:
        ewm_vols: DataFrame with EWM vol columns
        realized_vol: Series of realized volatility
        lookback_periods: Periods for rolling regression

    Returns:
        DataFrame of rolling weights for each EWM
    """
    weights_list = []

    for i in range(lookback_periods, len(ewm_vols)):
        # Get training window
        ewm_window = ewm_vols.iloc[i - lookback_periods : i]
        real_window = realized_vol.iloc[i - lookback_periods : i]

        # Align and drop NaN
        df = pd.concat([ewm_window, real_window.rename("target")], axis=1).dropna()

        if len(df) < 30:
            weights_list.append({col: np.nan for col in ewm_vols.columns})
            continue

        # Simple approach: weight by correlation
        X = df.drop("target", axis=1)
        y = df["target"]

        corrs = X.corrwith(y)
        corrs = corrs.clip(lower=0)  # Only positive correlations
        weights = corrs / corrs.sum() if corrs.sum() > 0 else corrs * 0 + 1 / len(corrs)

        weights_list.append(weights.to_dict())

    # Create DataFrame
    weights_df = pd.DataFrame(weights_list, index=ewm_vols.index[lookback_periods:])

    return weights_df


def compute_regime_indicator(
    ewm_vols: pd.DataFrame,
) -> pd.Series:
    """
    Compute risk regime indicator from EWM vol dispersion.

    High dispersion = transitioning regime (fast/slow EWMs diverge)
    Low dispersion = stable regime

    Args:
        ewm_vols: DataFrame with EWM vol columns

    Returns:
        Series of regime indicator (higher = more uncertainty)
    """
    # Cross-EWM dispersion (std across EWM measures)
    dispersion = ewm_vols.std(axis=1)

    # Normalize by mean level
    mean_vol = ewm_vols.mean(axis=1)
    regime = dispersion / mean_vol.replace(0, np.nan)

    return regime


def analyze_regime_performance(
    returns: pd.Series,
    regime: pd.Series,
    n_regimes: int = 3,
) -> pd.DataFrame:
    """
    Analyze performance in different risk regimes.

    Args:
        returns: Series of portfolio returns
        regime: Series of regime indicator
        n_regimes: Number of regime buckets

    Returns:
        DataFrame with performance by regime
    """
    # Bucket into regimes
    regime_buckets = pd.qcut(regime, n_regimes, labels=range(1, n_regimes + 1))

    results = []
    for r in range(1, n_regimes + 1):
        mask = regime_buckets == r
        regime_returns = returns[mask]

        if len(regime_returns) < 10:
            continue

        ann_return = regime_returns.mean() * PERIODS_PER_YEAR
        ann_vol = regime_returns.std() * np.sqrt(PERIODS_PER_YEAR)
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

        results.append({
            "regime": r,
            "n_periods": len(regime_returns),
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "avg_regime_indicator": regime[mask].mean(),
        })

    return pd.DataFrame(results)

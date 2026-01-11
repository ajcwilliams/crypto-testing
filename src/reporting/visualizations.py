"""
Visualization functions for backtest reports.

Uses Plotly for interactive charts.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_equity_curve(
    returns: pd.Series,
    title: str = "Equity Curve",
) -> go.Figure:
    """
    Create equity curve chart.

    Args:
        returns: Series of portfolio returns
        title: Chart title

    Returns:
        Plotly Figure
    """
    cumulative = (1 + returns).cumprod()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cumulative.index,
            y=cumulative.values,
            mode="lines",
            name="Portfolio",
            line=dict(color="blue"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode="x unified",
    )

    return fig


def create_drawdown_chart(
    returns: pd.Series,
    title: str = "Drawdown",
) -> go.Figure:
    """
    Create drawdown chart.

    Args:
        returns: Series of portfolio returns
        title: Chart title

    Returns:
        Plotly Figure
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="red"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
    )

    return fig


def create_sharpe_heatmap(
    heatmap_df: pd.DataFrame,
    title: str = "Sharpe Ratio by Parameters",
) -> go.Figure:
    """
    Create Sharpe ratio heatmap.

    Args:
        heatmap_df: DataFrame with funding rows and momentum columns
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_df.values,
            x=[str(c) for c in heatmap_df.columns],
            y=[str(i) for i in heatmap_df.index],
            colorscale="RdYlGn",
            text=np.round(heatmap_df.values, 2),
            texttemplate="%{text}",
            hovertemplate="Momentum: %{x}d<br>Funding: %{y}d<br>Sharpe: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Momentum Lookback (days)",
        yaxis_title="Funding Lookback (days)",
    )

    return fig


def create_rolling_sharpe(
    returns: pd.Series,
    window_periods: int = 180,
    title: str = "Rolling Sharpe Ratio",
) -> go.Figure:
    """
    Create rolling Sharpe ratio chart.

    Args:
        returns: Series of portfolio returns
        window_periods: Rolling window size
        title: Chart title

    Returns:
        Plotly Figure
    """
    rolling_mean = returns.rolling(window_periods).mean()
    rolling_std = returns.rolling(window_periods).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(365 * 3)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            mode="lines",
            name="Rolling Sharpe",
            line=dict(color="purple"),
        )
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        hovermode="x unified",
    )

    return fig


def create_monthly_returns_heatmap(
    returns: pd.Series,
    title: str = "Monthly Returns",
) -> go.Figure:
    """
    Create monthly returns heatmap.

    Args:
        returns: Series of portfolio returns
        title: Chart title

    Returns:
        Plotly Figure
    """
    # Resample to monthly
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    # Create DataFrame
    monthly_df = pd.DataFrame({"return": monthly})
    monthly_df["year"] = monthly_df.index.year
    monthly_df["month"] = monthly_df.index.month_name().str[:3]

    # Pivot
    pivot = monthly_df.pivot(index="year", columns="month", values="return")

    # Reorder months
    month_order = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    pivot = pivot[[m for m in month_order if m in pivot.columns]]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values * 100,
            x=pivot.columns.tolist(),
            y=[str(y) for y in pivot.index],
            colorscale="RdYlGn",
            text=np.round(pivot.values * 100, 1),
            texttemplate="%{text:.1f}%",
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
            zmid=0,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Year",
    )

    return fig


def create_position_count_chart(
    weights: pd.DataFrame,
    title: str = "Position Count Over Time",
) -> go.Figure:
    """
    Create chart showing number of positions over time.

    Args:
        weights: DataFrame of portfolio weights
        title: Chart title

    Returns:
        Plotly Figure
    """
    n_long = (weights > 0).sum(axis=1)
    n_short = (weights < 0).sum(axis=1)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=n_long.index,
            y=n_long.values,
            mode="lines",
            name="Long",
            line=dict(color="green"),
            stackgroup="one",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=n_short.index,
            y=n_short.values,
            mode="lines",
            name="Short",
            line=dict(color="red"),
            stackgroup="one",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Number of Positions",
        hovermode="x unified",
    )

    return fig


def create_ewm_vol_chart(
    ewm_vols: pd.DataFrame,
    realized_vol: pd.Series | None = None,
    title: str = "EWM Volatility Estimates",
) -> go.Figure:
    """
    Create chart comparing EWM volatility estimates.

    Args:
        ewm_vols: DataFrame with EWM vol columns
        realized_vol: Optional series of realized vol for comparison
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Add each EWM line
    colors = px.colors.qualitative.Plotly
    for i, col in enumerate(ewm_vols.columns):
        fig.add_trace(
            go.Scatter(
                x=ewm_vols.index,
                y=ewm_vols[col].values * 100,
                mode="lines",
                name=col,
                line=dict(color=colors[i % len(colors)]),
            )
        )

    # Add realized vol if provided
    if realized_vol is not None:
        fig.add_trace(
            go.Scatter(
                x=realized_vol.index,
                y=realized_vol.values * 100,
                mode="lines",
                name="Realized",
                line=dict(color="black", dash="dot"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        hovermode="x unified",
    )

    return fig


def create_regime_chart(
    returns: pd.Series,
    regime: pd.Series,
    title: str = "Returns by Risk Regime",
) -> go.Figure:
    """
    Create chart showing returns colored by regime.

    Args:
        returns: Series of portfolio returns
        regime: Series of regime indicator
        title: Chart title

    Returns:
        Plotly Figure
    """
    cumulative = (1 + returns).cumprod()

    # Bucket regime into levels
    regime_levels = pd.qcut(regime, 3, labels=["Low Risk", "Medium Risk", "High Risk"])

    fig = go.Figure()

    for level in ["Low Risk", "Medium Risk", "High Risk"]:
        mask = regime_levels == level
        fig.add_trace(
            go.Scatter(
                x=cumulative[mask].index,
                y=cumulative[mask].values,
                mode="markers",
                name=level,
                marker=dict(size=3),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode="closest",
    )

    return fig


def create_pnl_decomposition_chart(
    funding_pnl: pd.Series,
    price_pnl: pd.Series,
    title: str = "PnL Decomposition",
) -> go.Figure:
    """
    Create chart showing cumulative PnL decomposition.

    Args:
        funding_pnl: Series of funding PnL
        price_pnl: Series of price PnL
        title: Chart title

    Returns:
        Plotly Figure
    """
    cum_funding = funding_pnl.cumsum()
    cum_price = price_pnl.cumsum()
    cum_total = cum_funding + cum_price

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=cum_total.index,
            y=cum_total.values,
            mode="lines",
            name="Total",
            line=dict(color="blue", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=cum_funding.index,
            y=cum_funding.values,
            mode="lines",
            name="Funding (Carry)",
            line=dict(color="green"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=cum_price.index,
            y=cum_price.values,
            mode="lines",
            name="Price (Momentum)",
            line=dict(color="orange"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative PnL",
        hovermode="x unified",
    )

    return fig

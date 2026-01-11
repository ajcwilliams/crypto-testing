"""
Parameter grid for backtest optimization.

Defines the lookback periods to test for funding and momentum signals.
"""

from dataclasses import dataclass
from itertools import product
from typing import Iterator

# Lookback periods in days (will be converted to 8h periods)
LOOKBACK_DAYS = [2, 4, 8, 16, 32, 64, 128, 256]

# 8h periods per day
PERIODS_PER_DAY = 3


@dataclass(frozen=True)
class ParameterSet:
    """Single parameter combination."""

    funding_lookback_days: int
    momentum_lookback_days: int

    @property
    def funding_lookback_periods(self) -> int:
        """Funding lookback in 8h periods."""
        return self.funding_lookback_days * PERIODS_PER_DAY

    @property
    def momentum_lookback_periods(self) -> int:
        """Momentum lookback in 8h periods."""
        return self.momentum_lookback_days * PERIODS_PER_DAY

    def __str__(self) -> str:
        return f"F{self.funding_lookback_days}d_M{self.momentum_lookback_days}d"


class ParameterGrid:
    """Grid of parameter combinations to test."""

    def __init__(
        self,
        funding_lookbacks: list[int] | None = None,
        momentum_lookbacks: list[int] | None = None,
    ):
        """
        Initialize parameter grid.

        Args:
            funding_lookbacks: List of funding lookback periods in days
            momentum_lookbacks: List of momentum lookback periods in days
        """
        self.funding_lookbacks = funding_lookbacks or LOOKBACK_DAYS
        self.momentum_lookbacks = momentum_lookbacks or LOOKBACK_DAYS

    def __iter__(self) -> Iterator[ParameterSet]:
        """Iterate over all parameter combinations."""
        for f_lb, m_lb in product(self.funding_lookbacks, self.momentum_lookbacks):
            yield ParameterSet(
                funding_lookback_days=f_lb,
                momentum_lookback_days=m_lb,
            )

    def __len__(self) -> int:
        """Total number of parameter combinations."""
        return len(self.funding_lookbacks) * len(self.momentum_lookbacks)

    def to_list(self) -> list[ParameterSet]:
        """Convert to list of parameter sets."""
        return list(self)

    def to_dict_list(self) -> list[dict]:
        """Convert to list of dicts for DataFrame creation."""
        return [
            {
                "funding_lookback_days": ps.funding_lookback_days,
                "momentum_lookback_days": ps.momentum_lookback_days,
                "funding_lookback_periods": ps.funding_lookback_periods,
                "momentum_lookback_periods": ps.momentum_lookback_periods,
                "name": str(ps),
            }
            for ps in self
        ]


def get_default_grid() -> ParameterGrid:
    """Get default parameter grid (8x8 = 64 combinations)."""
    return ParameterGrid()


def get_quick_grid() -> ParameterGrid:
    """Get quick parameter grid for testing (4x4 = 16 combinations)."""
    return ParameterGrid(
        funding_lookbacks=[4, 16, 64, 256],
        momentum_lookbacks=[4, 16, 64, 256],
    )


def get_fine_grid() -> ParameterGrid:
    """Get finer parameter grid (more combinations)."""
    return ParameterGrid(
        funding_lookbacks=[2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
        momentum_lookbacks=[2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
    )

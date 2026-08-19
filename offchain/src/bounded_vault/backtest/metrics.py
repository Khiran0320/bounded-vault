"""Performance metrics computed from a completed backtest run.

Nothing here knows about agents or market data, so metrics can be
recomputed from a saved parquet without rerunning the backtest.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

DAYS_PER_YEAR = 365


def cumulative_return(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("need at least two vault values")
    return values[-1] / values[0] - 1.0


def annualised_return(
    values: Sequence[float], days_per_year: int = DAYS_PER_YEAR
) -> float:
    """Geometric annualisation of the realised path."""
    periods = len(values) - 1
    if periods < 1:
        raise ValueError("need at least two vault values")
    return (values[-1] / values[0]) ** (days_per_year / periods) - 1.0


def annualised_volatility(
    daily_returns: Sequence[float], days_per_year: int = DAYS_PER_YEAR
) -> float:
    """Sample standard deviation of daily returns, scaled by sqrt of time."""
    n = len(daily_returns)
    if n < 2:
        raise ValueError("need at least two daily returns")
    mean = sum(daily_returns) / n
    variance = sum((r - mean) ** 2 for r in daily_returns) / (n - 1)
    return math.sqrt(variance) * math.sqrt(days_per_year)


def downside_deviation(
    daily_returns: Sequence[float],
    target_annual: float = 0.0,
    days_per_year: int = DAYS_PER_YEAR,
) -> float:
    """Annualised deviation of returns falling below a target.

    Squared shortfalls are averaged over every observation rather than
    over the shortfall days alone, following the target semideviation
    convention. Averaging over shortfall days only would report a
    strategy that rarely loses as riskier than one that loses often but
    mildly, which inverts the quantity the measure exists to capture.
    """
    n = len(daily_returns)
    if n < 2:
        raise ValueError("need at least two daily returns")
    target_daily = (1.0 + target_annual) ** (1.0 / days_per_year) - 1.0
    shortfalls = [min(0.0, r - target_daily) for r in daily_returns]
    variance = sum(s * s for s in shortfalls) / n
    return math.sqrt(variance) * math.sqrt(days_per_year)


def sharpe_ratio(
    daily_returns: Sequence[float],
    risk_free_annual: float = 0.0,
    days_per_year: int = DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio.

    Over a short evaluation window this estimate is extremely noisy and
    should be reported with that caveat rather than read as a ranking.
    """
    vol = annualised_volatility(daily_returns, days_per_year)
    if vol == 0.0:
        return 0.0
    mean_daily = sum(daily_returns) / len(daily_returns)
    ann_return = (1.0 + mean_daily) ** days_per_year - 1.0
    return (ann_return - risk_free_annual) / vol


def sortino_ratio(
    daily_returns: Sequence[float],
    target_annual: float = 0.0,
    days_per_year: int = DAYS_PER_YEAR,
) -> float:
    """Excess return per unit of downside deviation.

    Preferred to Sharpe when comparing strategies whose weights are
    externally constrained. A binding concentration cap forces exposure
    to a volatile asset, and Sharpe charges that exposure symmetrically,
    penalising the upside days the constraint also produced. Sortino
    charges only the shortfalls, so it separates the risk a constraint
    imposes from the variance it merely permits.

    Returns zero when there are no shortfalls at all, matching the
    convention in sharpe_ratio for zero volatility. Note that a strategy
    with very few losing days produces a very large ratio, which
    reflects the sparsity of the denominator rather than skill, so the
    figure should be read alongside the raw return.
    """
    deviation = downside_deviation(daily_returns, target_annual, days_per_year)
    if deviation == 0.0:
        return 0.0
    mean_daily = sum(daily_returns) / len(daily_returns)
    ann_return = (1.0 + mean_daily) ** days_per_year - 1.0
    return (ann_return - target_annual) / deviation


def max_drawdown(values: Sequence[float]) -> float:
    """Largest peak to trough decline, as a positive fraction."""
    if not values:
        raise ValueError("need at least one vault value")
    peak = values[0]
    worst = 0.0
    for v in values:
        peak = max(peak, v)
        worst = max(worst, 1.0 - v / peak)
    return worst


def summarise(
    values: Sequence[float],
    daily_returns: Sequence[float],
    turnovers: Sequence[float],
    violations: int,
    days_per_year: int = DAYS_PER_YEAR,
) -> dict:
    """Single row summary for one agent over one backtest run."""
    n = len(daily_returns)
    return {
        "days": n,
        "cumulative_return": cumulative_return(values),
        "annualised_return": annualised_return(values, days_per_year),
        "annualised_volatility": annualised_volatility(daily_returns, days_per_year),
        "downside_deviation": downside_deviation(daily_returns, 0.0, days_per_year),
        "sharpe_ratio": sharpe_ratio(daily_returns, 0.0, days_per_year),
        "sortino_ratio": sortino_ratio(daily_returns, 0.0, days_per_year),
        "max_drawdown": max_drawdown(values),
        "mean_daily_turnover": sum(turnovers) / len(turnovers) if turnovers else 0.0,
        "violation_count": int(violations),
        "violation_rate": violations / n if n else 0.0,
    }
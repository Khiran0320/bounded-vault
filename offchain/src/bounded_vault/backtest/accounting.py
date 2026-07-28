"""Return accounting primitives for the backtest harness.

Every function here is pure and deterministic. No network, no file I/O,
no mutable state. This keeps the engine thin and makes each piece of the
return maths independently unit testable.
"""

from __future__ import annotations

from bounded_vault.schema import AdapterId

BPS_DENOMINATOR = 10_000
DAYS_PER_YEAR = 365


def apy_to_daily(apy: float, days_per_year: int = DAYS_PER_YEAR) -> float:
    """Convert an annual yield to its compounded daily equivalent.

    apy is a fraction, not a percent: 0.052 means 5.2 percent per year.
    Compounding the result over a full year reproduces the stated apy.
    """
    if apy <= -1.0:
        raise ValueError(f"apy must be greater than -1.0, got {apy}")
    return (1.0 + apy) ** (1.0 / days_per_year) - 1.0


def adapter_daily_return(price_return: float, apy: float) -> float:
    """Total one day return for a single adapter position.

    Combines the price move of the underlying asset with the yield
    accrued over the same day, multiplicatively so the two compound.
    """
    return (1.0 + price_return) * (1.0 + apy_to_daily(apy)) - 1.0


def portfolio_daily_return(
    weights_bps: dict[AdapterId, int],
    adapter_returns: dict[AdapterId, float],
) -> float:
    """Weighted one day return for the whole vault.

    Basis points not allocated to an adapter are treated as idle cash
    earning nothing. Weights are validated rather than normalised, so a
    bad proposal surfaces as an error instead of silent rescaling.
    """
    if any(bps < 0 for bps in weights_bps.values()):
        raise ValueError(f"negative weight in {weights_bps}")

    total_bps = sum(weights_bps.values())
    if total_bps > BPS_DENOMINATOR:
        raise ValueError(
            f"weights sum to {total_bps} bps, exceeds {BPS_DENOMINATOR}"
        )

    total = 0.0
    for adapter, bps in weights_bps.items():
        if bps == 0:
            continue
        if adapter not in adapter_returns:
            raise KeyError(f"no return supplied for allocated adapter {adapter}")
        total += (bps / BPS_DENOMINATOR) * adapter_returns[adapter]
    return total


def turnover(
    prev_bps: dict[AdapterId, int],
    new_bps: dict[AdapterId, int],
) -> float:
    """One way turnover between two weight vectors, as a fraction of value.

    Absolute weight changes summed then halved, so moving the entire
    vault from one adapter to another reports 1.0 rather than 2.0.
    """
    adapters = set(prev_bps) | set(new_bps)
    gross = sum(abs(new_bps.get(a, 0) - prev_bps.get(a, 0)) for a in adapters)
    return gross / 2.0 / BPS_DENOMINATOR


def rebalance_cost(
    prev_bps: dict[AdapterId, int],
    new_bps: dict[AdapterId, int],
    cost_bps: int = 0,
) -> float:
    """Cost of moving between weight vectors, expressed as a return drag.

    Zero by default: main results are frictionless. A non zero cost_bps
    gives the sensitivity check without touching the engine.
    """
    if cost_bps == 0:
        return 0.0
    return turnover(prev_bps, new_bps) * (cost_bps / BPS_DENOMINATOR)
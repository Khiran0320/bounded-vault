"""Agent 2: mean-variance portfolio optimiser.

Solves a convex quadratic program over the adapter set, then converts the
continuous solution to integer basis points. Where Agent 1 allocates
proportionally to yield and ignores risk entirely, this agent trades
expected return against covariance, so it will hold a stable adapter even
when a volatile one advertises more yield.

Expected return per adapter is annualised price drift plus current APY.
Covariance is estimated from price returns alone, since daily yield accrual
carries no meaningful variance over a one day horizon.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from bounded_vault.agents.base import Agent
from bounded_vault.market.view import MarketView
from bounded_vault.schema import AdapterId, Proposal, StrategyAllocation
from bounded_vault.weights import to_basis_points

BPS_DENOMINATOR = 10_000
DAYS_PER_YEAR = 365


def _column_map(returns: pd.DataFrame) -> dict[AdapterId, object]:
    """Map each adapter to the column label pandas actually stored.

    pandas stores IntEnum keys as plain integers, so column labels come back
    as ints rather than enum members while MarketView.yields keeps the enum.
    This recovers the correspondence so the two can be aligned.
    """
    mapping: dict[AdapterId, object] = {}
    for column in returns.columns:
        adapter = column if isinstance(column, AdapterId) else AdapterId(int(column))
        mapping[adapter] = column
    return mapping


def _enforce_cap(bps: dict[AdapterId, int], cap: int) -> dict[AdapterId, int]:
    """Push any post-rounding overshoot onto adapters with headroom.

    Largest remainder can lift a weight one basis point above the cap when
    the continuous solution sat exactly on the boundary, which mean-variance
    corner solutions do routinely. Redistributing preserves the exact sum.
    """
    adjusted = dict(bps)
    excess = sum(max(0, w - cap) for w in adjusted.values())
    if excess == 0:
        return adjusted

    for adapter, weight in adjusted.items():
        if weight > cap:
            adjusted[adapter] = cap

    for adapter in sorted(adjusted, key=lambda a: adjusted[a]):
        if excess == 0:
            break
        headroom = cap - adjusted[adapter]
        moved = min(headroom, excess)
        adjusted[adapter] += moved
        excess -= moved

    if excess != 0:
        raise ValueError(
            f"cannot satisfy cap of {cap} bps across {len(adjusted)} adapters"
        )
    return adjusted


class MeanVarianceAgent(Agent):
    """Maximises annualised mean-variance utility over the adapter simplex.

    Objective is mu' w minus half of risk_aversion times w' Sigma w, subject
    to non-negative weights summing to one, plus an optional per-adapter
    ceiling.

    max_strategy_bps is deliberately optional. Left as None, the agent
    optimises without knowledge of the vault's per-strategy limit and will
    propose allocations the constraint layer refuses. Set to the vault's
    actual cap, the agent self-constrains. Running both configurations
    isolates the effect of duplicating a safety constraint inside an agent.
    """

    def __init__(
        self,
        risk_aversion: float = 3.0,
        lookback_days: int = 60,
        max_strategy_bps: int | None = None,
        ridge: float = 1e-8,
        name: str = "mean_variance",
    ) -> None:
        if risk_aversion <= 0.0:
            raise ValueError(f"risk_aversion must be positive, got {risk_aversion}")
        if lookback_days < 2:
            raise ValueError(f"lookback_days must be at least 2, got {lookback_days}")
        if max_strategy_bps is not None and not 0 < max_strategy_bps <= BPS_DENOMINATOR:
            raise ValueError(f"max_strategy_bps out of range: {max_strategy_bps}")

        self.name = name
        self.risk_aversion = risk_aversion
        self.lookback_days = lookback_days
        self.max_strategy_bps = max_strategy_bps
        self.ridge = ridge

    def _estimate(
        self, market: MarketView
    ) -> tuple[list[AdapterId], np.ndarray, np.ndarray]:
        """Build the annualised expected return vector and covariance matrix."""
        columns = _column_map(market.returns)
        adapters = list(columns)
        labels = [columns[adapter] for adapter in adapters]

        window = market.returns[labels].tail(self.lookback_days)
        if len(window) < 2:
            raise ValueError(
                f"need at least 2 observations to estimate covariance, "
                f"got {len(window)}"
            )

        missing = [a for a in adapters if a not in market.yields]
        if missing:
            raise KeyError(f"no yield supplied for {missing}")

        drift = window.mean().to_numpy(dtype=float) * DAYS_PER_YEAR
        apy = np.array([float(market.yields[a]) for a in adapters])
        mu = drift + apy

        sigma = window.cov().to_numpy(dtype=float) * DAYS_PER_YEAR
        sigma = sigma + np.eye(len(adapters)) * self.ridge
        return adapters, mu, sigma

    def _solve(self, mu: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, str]:
        """Solve the quadratic program, falling back to equal weight on failure.

        A solver failure is recorded rather than raised. Killing a 137 day
        backtest because one day's covariance was awkward would lose more
        information than the fallback costs, and the rationale field on the
        Proposal preserves which days fell back.
        """
        n = len(mu)
        weights = cp.Variable(n, nonneg=True)
        constraints = [cp.sum(weights) == 1.0]

        if self.max_strategy_bps is not None:
            ceiling = self.max_strategy_bps / BPS_DENOMINATOR
            if n * ceiling < 1.0:
                raise ValueError(
                    f"cap of {self.max_strategy_bps} bps across {n} adapters "
                    "cannot reach a full allocation"
                )
            constraints.append(weights <= ceiling)

        utility = mu @ weights - 0.5 * self.risk_aversion * cp.quad_form(
            weights, cp.psd_wrap(sigma)
        )
        problem = cp.Problem(cp.Maximize(utility), constraints)

        try:
            problem.solve()
        except cp.error.SolverError:
            return np.full(n, 1.0 / n), "solver_error"

        if weights.value is None:
            return np.full(n, 1.0 / n), f"no_solution_{problem.status}"

        solution = np.clip(np.asarray(weights.value, dtype=float).ravel(), 0.0, None)
        total = solution.sum()
        if total <= 0.0:
            return np.full(n, 1.0 / n), "degenerate_solution"
        return solution / total, str(problem.status)

    def propose(self, market: MarketView) -> Proposal:
        adapters, mu, sigma = self._estimate(market)
        solution, status = self._solve(mu, sigma)

       # to_basis_points is positional, so the adapter ordering from
        # _estimate must be preserved when zipping the result back.
        bps = dict(
            zip(adapters, to_basis_points([float(w) for w in solution]))
        )
        if self.max_strategy_bps is not None:
            bps = _enforce_cap(bps, self.max_strategy_bps)

        expected = ", ".join(
            f"{a.name}={m:.4f}" for a, m in zip(adapters, mu)
        )
        rationale = (
            f"mean-variance, risk_aversion={self.risk_aversion}, "
            f"lookback={self.lookback_days}d, status={status}, "
            f"annualised expected return: {expected}"
        )

        return Proposal(
            agent_name=self.name,
            as_of=market.as_of,
            allocations=[
                StrategyAllocation(adapter=adapter, weight_bps=int(weight))
                for adapter, weight in bps.items()
            ],
            rationale=rationale,
        )
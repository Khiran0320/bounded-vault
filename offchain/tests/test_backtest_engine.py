"""Tests for the backtest engine.

Uses synthetic market data with known analytic answers rather than the
frozen snapshot, so each assertion pins an exact expected value instead of
merely checking that the run completed.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from bounded_vault.agents.base import Agent
from bounded_vault.backtest.engine import (
    BacktestConfig,
    run_backtest,
    summarise_run,
)
from bounded_vault.constraints import ConstraintConfig, ViolationReason
from bounded_vault.schema import AdapterId, Proposal, StrategyAllocation

LENDING = AdapterId.LENDING
LST = AdapterId.LIQUID_STAKING


def make_frames(days, price_returns, apys):
    """Build aligned returns and yields frames holding constant values."""
    index = pd.date_range("2026-01-01", periods=days, freq="D")
    returns = pd.DataFrame(
        {adapter: [r] * days for adapter, r in price_returns.items()}, index=index
    )
    yields = pd.DataFrame(
        {adapter: [a] * days for adapter, a in apys.items()}, index=index
    )
    return returns, yields


def build_proposal(name, as_of, weights_bps):
    return Proposal(
        agent_name=name,
        as_of=as_of,
        allocations=[
            StrategyAllocation(adapter=adapter, weight_bps=bps)
            for adapter, bps in weights_bps.items()
        ],
        rationale=None,
    )


class FixedAgent(Agent):
    """Proposes the same weight vector every day."""

    def __init__(self, weights_bps, name="fixed"):
        self.name = name
        self._weights = weights_bps

    def propose(self, market):
        return build_proposal(self.name, market.as_of, self._weights)


class SwitchingAgent(Agent):
    """Proposes one vector on the first day and another on every day after."""

    def __init__(self, first, later, name="switching"):
        self.name = name
        self._first = first
        self._later = later
        self._calls = 0

    def propose(self, market):
        weights = self._first if self._calls == 0 else self._later
        self._calls += 1
        return build_proposal(self.name, market.as_of, weights)


class SpyAgent(Agent):
    """Records the latest date visible in each MarketView it receives."""

    def __init__(self, weights_bps, name="spy"):
        self.name = name
        self._weights = weights_bps
        self.observed = []

    def propose(self, market):
        self.observed.append((market.as_of, market.returns.index.max()))
        return build_proposal(self.name, market.as_of, self._weights)


class WallClockAgent(Agent):
    """Stamps proposals with a fixed real world date instead of view time."""

    def __init__(self, weights_bps, name="wallclock"):
        self.name = name
        self._weights = weights_bps

    def propose(self, market):
        return build_proposal(self.name, datetime(2000, 1, 1), self._weights)


def test_constant_yield_compounds_to_stated_apy():
    """A 5 percent APY held for 365 days with no price movement gives 1.05.

    This exercises the full accounting chain: apy_to_daily, the
    multiplicative combination with price return, and the daily compounding
    of vault value. Any error in the conversion shows up here.
    """
    returns, yields = make_frames(366, {LENDING: 0.0}, {LENDING: 0.05})
    agent = FixedAgent({LENDING: 10_000})
    constraints = ConstraintConfig(
        allowed_adapters=frozenset({LENDING}),
        max_strategy_bps=10_000,
    )

    results = run_backtest(
        agent, returns, yields, constraints, BacktestConfig(warmup_days=0)
    )

    assert len(results) == 365
    assert results["accepted"].all()
    assert results["vault_value_close"].iloc[-1] == pytest.approx(1.05, rel=1e-9)


def test_rejected_proposal_holds_previous_weights():
    """The reference monitor property, as an executable claim.

    An agent that proposes an allocation above the per strategy cap is
    refused, the vault keeps the allocation it already had, and the refusal
    is counted rather than raised.
    """
    returns, yields = make_frames(
        12, {LENDING: 0.0, LST: 0.0}, {LENDING: 0.05, LST: 0.07}
    )
    agent = SwitchingAgent(
        first={LENDING: 6_000, LST: 4_000},
        later={LENDING: 9_000, LST: 1_000},
    )
    constraints = ConstraintConfig(
        allowed_adapters=frozenset({LENDING, LST}),
        max_strategy_bps=6_000,
    )

    results = run_backtest(
        agent, returns, yields, constraints, BacktestConfig(warmup_days=0)
    )

    assert results["accepted"].iloc[0]
    assert not results["accepted"].iloc[1:].any()
    assert (
        results["rejection_reason"].iloc[1:]
        == ViolationReason.STRATEGY_CAP_EXCEEDED.value
    ).all()

    assert (results["weight_lending_bps"] == 6_000).all()
    assert (results["weight_liquid_staking_bps"] == 4_000).all()
    assert (results["turnover"].iloc[1:] == 0.0).all()

    assert summarise_run(results)["violation_count"] == len(results) - 1


def test_view_never_contains_future_data():
    """No-lookahead is structural, not a promise the agent has to keep.

    The most an agent can possibly see is the frame it was handed. This
    asserts that frame always ends on the decision date, so tomorrow's
    return is absent rather than merely off limits.
    """
    returns, yields = make_frames(20, {LENDING: 0.01}, {LENDING: 0.05})
    agent = SpyAgent({LENDING: 10_000})
    constraints = ConstraintConfig(
        allowed_adapters=frozenset({LENDING}),
        max_strategy_bps=10_000,
    )

    run_backtest(agent, returns, yields, constraints, BacktestConfig(warmup_days=0))

    assert agent.observed
    for as_of, last_visible in agent.observed:
        assert last_visible == as_of

    # The final date is never a decision date, since it has no following day.
    assert agent.observed[-1][0] == returns.index[-2]


def test_agent_using_wall_clock_time_is_rejected():
    """An agent that reads the real clock is not reproducible, so it fails."""
    returns, yields = make_frames(10, {LENDING: 0.0}, {LENDING: 0.05})
    agent = WallClockAgent({LENDING: 10_000})
    constraints = ConstraintConfig(
        allowed_adapters=frozenset({LENDING}),
        max_strategy_bps=10_000,
    )

    with pytest.raises(ValueError, match="wall clock"):
        run_backtest(
            agent, returns, yields, constraints, BacktestConfig(warmup_days=0)
        )


def test_exact_sum_and_sub_total_cap_are_unreachable():
    """The known constraint composition conflict, stated as a test.

    No weight vector can sum to exactly 10,000 bps while staying under a
    total cap below 10,000. Relaxing the exact sum rule makes the same
    configuration satisfiable, which isolates the cause to that rule.
    """
    conflicting = ConstraintConfig(
        allowed_adapters=frozenset({LENDING, LST}),
        max_strategy_bps=6_000,
        total_cap_bps=9_000,
    )
    assert not conflicting.is_reachable()

    relaxed = ConstraintConfig(
        allowed_adapters=frozenset({LENDING, LST}),
        max_strategy_bps=6_000,
        total_cap_bps=9_000,
        require_exact_sum=False,
    )
    assert relaxed.is_reachable()
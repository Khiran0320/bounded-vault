"""Tests for the mean-variance agent.

Constructs market views where the optimal answer is known by inspection,
so each assertion pins behaviour rather than merely checking the solver
returned something.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bounded_vault.agents.mean_variance import MeanVarianceAgent
from bounded_vault.backtest.engine import BacktestConfig, run_backtest
from bounded_vault.constraints import ConstraintConfig
from bounded_vault.market.view import MarketView
from bounded_vault.schema import AdapterId

LENDING = AdapterId.LENDING
LST = AdapterId.LIQUID_STAKING


def make_view(days=90, lending_vol=0.0001, lst_vol=0.03, lst_drift=0.0, yields=None):
    """Build a view with a near-riskless lending adapter and a volatile LST."""
    rng = np.random.default_rng(42)
    index = pd.date_range("2026-01-01", periods=days, freq="D")
    frame = pd.DataFrame(
        {
            LENDING: rng.normal(0.0, lending_vol, days),
            LST: rng.normal(lst_drift, lst_vol, days),
        },
        index=index,
    )
    return MarketView(
        as_of=index[-1],
        returns=frame,
        yields=yields or {LENDING: 0.05, LST: 0.08},
    )


def make_driftless_view(days=80, lending_vol=0.002, lst_vol=0.02, yields=None):
    """Build a view whose sample moments are exact rather than estimated.

    Returns alternate with period 2 and period 4, so over any whole number
    of four day cycles both means are exactly zero and the cross products
    cancel to give exactly zero covariance. Expected return is then purely
    the APY, which is what lets a test assert on yield sensitivity without
    estimation noise drowning the signal.

    With the default volatilities and a risk aversion of 3, the optimum sits
    strictly inside the simplex, so allocation can move in either direction.
    """
    if days % 4:
        raise ValueError("days must be a multiple of 4 to cancel exactly")

    index = pd.date_range("2026-01-01", periods=days, freq="D")
    frame = pd.DataFrame(
        {
            LENDING: [lending_vol if i % 2 == 0 else -lending_vol for i in range(days)],
            LST: [lst_vol if i % 4 < 2 else -lst_vol for i in range(days)],
        },
        index=index,
    )
    return MarketView(
        as_of=index[-1],
        returns=frame,
        yields=yields or {LENDING: 0.01, LST: 0.08},
    )


def weights_of(proposal):
    return {a.adapter: a.weight_bps for a in proposal.allocations}


def test_weights_sum_to_exactly_ten_thousand_bps():
    """The on-chain exact sum rule leaves no room for rounding slack."""
    proposal = MeanVarianceAgent().propose(make_view())
    assert sum(weights_of(proposal).values()) == 10_000


def test_high_risk_aversion_prefers_the_stable_adapter():
    """With risk dominating, the near-riskless adapter takes most of the book.

    The LST offers more yield but carries roughly 300 times the volatility,
    so a sufficiently risk-averse optimiser should refuse to chase it.
    """
    proposal = MeanVarianceAgent(risk_aversion=500.0).propose(make_view())
    weights = weights_of(proposal)
    assert weights[LENDING] > weights[LST]


def test_low_risk_aversion_chases_the_higher_expected_return():
    """With risk nearly ignored, the higher expected return adapter wins."""
    view = make_view(lst_drift=0.002)
    proposal = MeanVarianceAgent(risk_aversion=0.01).propose(view)
    weights = weights_of(proposal)
    assert weights[LST] > weights[LENDING]


def test_solution_is_interior_on_driftless_data():
    """Both adapters hold a real position, so allocation can move either way.

    Guards the other driftless tests: an assertion about weight direction is
    vacuous if the optimum is pinned to a corner.
    """
    weights = weights_of(MeanVarianceAgent().propose(make_driftless_view()))
    assert all(0 < w < 10_000 for w in weights.values())


def test_yield_is_included_in_expected_return():
    """Raising an adapter's APY must shift allocation toward it.

    This is the check that would fail if the agent optimised on price
    returns alone, which would make the lending adapter look return-free.
    Uses driftless data so the only thing changing between the two runs is
    the yield, with expected return equal to the APY exactly.
    """
    agent = MeanVarianceAgent(risk_aversion=3.0)
    low = weights_of(
        agent.propose(make_driftless_view(yields={LENDING: 0.01, LST: 0.08}))
    )
    high = weights_of(
        agent.propose(make_driftless_view(yields={LENDING: 0.06, LST: 0.08}))
    )
    assert high[LENDING] > low[LENDING]
    assert high[LST] < low[LST]


def test_higher_risk_aversion_shifts_toward_the_stable_adapter():
    """Raising risk aversion moves weight to the lower variance adapter.

    The counterpart to the yield test: holding yields fixed and varying only
    the risk preference must move allocation in the opposite direction.
    """
    view_args = {"yields": {LENDING: 0.01, LST: 0.08}}
    timid = weights_of(
        MeanVarianceAgent(risk_aversion=6.0).propose(make_driftless_view(**view_args))
    )
    bold = weights_of(
        MeanVarianceAgent(risk_aversion=1.5).propose(make_driftless_view(**view_args))
    )
    assert timid[LENDING] > bold[LENDING]


def test_cap_is_respected_after_rounding():
    """Corner solutions sit exactly on the bound, where rounding can overshoot."""
    agent = MeanVarianceAgent(risk_aversion=500.0, max_strategy_bps=6_000)
    weights = weights_of(agent.propose(make_view()))
    assert all(w <= 6_000 for w in weights.values())
    assert sum(weights.values()) == 10_000


def test_as_of_is_taken_from_the_view():
    """Required by the engine's wall clock guard."""
    view = make_view()
    assert pd.Timestamp(MeanVarianceAgent().propose(view).as_of) == pd.Timestamp(
        view.as_of
    )


def test_uncapped_agent_violates_a_vault_cap_the_chain_enforces():
    """The dissertation result, as an executable claim.

    An agent optimising without knowledge of the vault's per-strategy limit
    proposes allocations the constraint layer refuses. The same agent given
    the cap does not. Nothing else differs between the two runs.
    """
    rng = np.random.default_rng(7)
    days = 120
    index = pd.date_range("2026-01-01", periods=days, freq="D")
    returns = pd.DataFrame(
        {
            LENDING: rng.normal(0.0, 0.0001, days),
            LST: rng.normal(0.0, 0.03, days),
        },
        index=index,
    )
    yields = pd.DataFrame(
        {LENDING: [0.05] * days, LST: [0.08] * days}, index=index
    )
    constraints = ConstraintConfig(
        allowed_adapters=frozenset({LENDING, LST}),
        max_strategy_bps=6_000,
    )
    config = BacktestConfig(warmup_days=60)

    unaware = run_backtest(
        MeanVarianceAgent(risk_aversion=500.0),
        returns,
        yields,
        constraints,
        config,
    )
    aware = run_backtest(
        MeanVarianceAgent(risk_aversion=500.0, max_strategy_bps=6_000),
        returns,
        yields,
        constraints,
        config,
    )

    assert (~unaware["accepted"]).sum() > 0
    assert aware["accepted"].all()
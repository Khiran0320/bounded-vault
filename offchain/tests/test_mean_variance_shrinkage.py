"""Tests for the Bayesian shrinkage estimator in Agent 2.

Verifies that tau behaves as a prior standard deviation should: at zero it
discards the sample entirely, unset it discards the prior entirely, and in
between it retains drift in proportion to how precisely it was measured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bounded_vault.agents.mean_variance import MeanVarianceAgent
from bounded_vault.market.view import MarketView
from bounded_vault.schema import AdapterId

DAYS_PER_YEAR = 365


def _market(n_days: int = 60) -> MarketView:
    """Deterministic two adapter view: one near-riskless, one volatile.

    Both carry a positive mean return so the shrinkage factor has something
    to act on. Alternating signs on the volatile series keep its mean small
    relative to its dispersion, which is the realistic case.
    """
    index = pd.date_range("2026-03-01", periods=n_days, freq="D")

    stable = np.full(n_days, 0.00002)
    volatile = np.where(np.arange(n_days) % 2 == 0, 0.031, -0.029)

    returns = pd.DataFrame(
        {AdapterId.LENDING: stable, AdapterId.LIQUID_STAKING: volatile},
        index=index,
    )
    yields = {AdapterId.LENDING: 0.048, AdapterId.LIQUID_STAKING: 0.072}
    return MarketView(as_of=index[-1], returns=returns, yields=yields)


def test_tau_zero_gives_apy_only():
    """A zero width prior admits no sample information, so mu is the yield."""
    agent = MeanVarianceAgent(tau=0.0)
    adapters, mu, _, diagnostics = agent._estimate(_market())

    np.testing.assert_allclose(diagnostics["keep"], 0.0)
    np.testing.assert_allclose(mu, diagnostics["apy"])


def test_tau_none_recovers_raw_drift():
    """Disabling shrinkage reproduces the pre-shrinkage estimator exactly."""
    agent = MeanVarianceAgent(tau=None)
    adapters, mu, _, diagnostics = agent._estimate(_market())

    np.testing.assert_allclose(diagnostics["keep"], 1.0)
    np.testing.assert_allclose(mu, diagnostics["drift"] + diagnostics["apy"])


def test_shrinkage_is_precision_weighted():
    """Volatile drift is discarded, stable drift survives, at the default tau."""
    agent = MeanVarianceAgent(tau=0.15)
    adapters, mu, _, diagnostics = agent._estimate(_market())

    keep = dict(zip(adapters, diagnostics["keep"]))
    assert keep[AdapterId.LIQUID_STAKING] < 0.05
    assert keep[AdapterId.LENDING] > 0.90


def test_shrinkage_is_monotone_in_tau():
    """A wider prior always retains at least as much of the observation."""
    market = _market()
    previous = None
    for tau in (0.05, 0.10, 0.15, 0.25, 0.50):
        _, _, _, diagnostics = MeanVarianceAgent(tau=tau)._estimate(market)
        current = diagnostics["keep"]
        if previous is not None:
            assert np.all(current >= previous - 1e-12)
        previous = current


def test_volatile_mu_is_dominated_by_yield():
    """At the default tau the volatile adapter's mu is essentially its APY."""
    agent = MeanVarianceAgent(tau=0.15)
    adapters, mu, _, diagnostics = agent._estimate(_market())

    i = adapters.index(AdapterId.LIQUID_STAKING)
    contribution = abs(mu[i] - diagnostics["apy"][i])
    assert contribution < 0.1 * diagnostics["apy"][i]


def test_negative_tau_rejected():
    with pytest.raises(ValueError):
        MeanVarianceAgent(tau=-0.1)
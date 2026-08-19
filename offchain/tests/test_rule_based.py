from datetime import datetime

import pandas as pd

from bounded_vault.agents.rule_based import RuleBasedAgent
from bounded_vault.market import MarketView
from bounded_vault.schema import AdapterId


def test_weights_proportional_to_yield(two_adapter_market):
    proposal = RuleBasedAgent().propose(two_adapter_market)
    weights = {a.adapter: a.weight_bps for a in proposal.allocations}
    assert weights[AdapterId.LENDING] == 2500
    assert weights[AdapterId.LIQUID_STAKING] == 7500
    assert proposal.total_bps == 10_000
    assert proposal.agent_name == "rule_based"


def test_equal_weight_when_no_yield_signal():
    idx = pd.to_datetime(["2025-01-01"])
    returns = pd.DataFrame(
        {AdapterId.LENDING: [0.0], AdapterId.LIQUID_STAKING: [0.0]}, index=idx
    )
    market = MarketView(
        as_of=datetime(2025, 1, 1),
        returns=returns,
        yields={AdapterId.LENDING: 0.0, AdapterId.LIQUID_STAKING: 0.0},
    )
    proposal = RuleBasedAgent().propose(market)
    weights = {a.adapter: a.weight_bps for a in proposal.allocations}
    assert weights[AdapterId.LENDING] == 5000
    assert weights[AdapterId.LIQUID_STAKING] == 5000
    assert proposal.total_bps == 10_000
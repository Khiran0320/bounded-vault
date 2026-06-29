"""Agent 1: rule-based. The deterministic baseline."""

from __future__ import annotations

from bounded_vault.agents.base import Agent
from bounded_vault.market import MarketView
from bounded_vault.schema import Proposal, StrategyAllocation
from bounded_vault.weights import to_basis_points


class RuleBasedAgent(Agent):
    """Allocates to each adapter in proportion to its current yield.

    The thesis is deliberately naive: put more capital where the yield is
    higher. No optimisation, no training, no parameters. This is the baseline
    the smarter agents are measured against.
    """

    name = "rule-based"

    def propose(self, market: MarketView) -> Proposal:
        adapters = market.adapters
        if not adapters:
            return Proposal(agent_name=self.name, allocations=[])

        ylds = [max(market.yields.get(a, 0.0), 0.0) for a in adapters]
        total = sum(ylds)

        if total <= 0.0:
            fractions = [1.0 / len(adapters)] * len(adapters)
            rationale = "no yield signal; equal-weight fallback"
        else:
            fractions = [y / total for y in ylds]
            rationale = "weights proportional to current yield"

        bps = to_basis_points(fractions)
        allocations = [
            StrategyAllocation(adapter=a, weight_bps=b)
            for a, b in zip(adapters, bps)
        ]
        return Proposal(agent_name=self.name, allocations=allocations, rationale=rationale)
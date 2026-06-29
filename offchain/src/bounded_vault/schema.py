"""
Shared proposal schema for the off-chain agent layer.

A Proposal is an OPINION: a set of target weights an agent would like the
vault to adopt. It carries no authority to move funds. The on-chain program
(and the Python constraint mirror used in backtests) is the sole place where
safety constraints are enforced. This file validates STRUCTURE only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import IntEnum

from pydantic import BaseModel, Field


class AdapterId(IntEnum):
    """Mirror of the on-chain AdapterId enum. Order MUST match the Rust enum."""
    LENDING = 0
    LIQUID_STAKING = 1


class StrategyAllocation(BaseModel):
    """One line of a proposal: how much to put into one adapter."""
    adapter: AdapterId
    weight_bps: int = Field(ge=0)


class Proposal(BaseModel):
    """The full opinion produced by one agent for one rebalance decision."""
    agent_name: str
    allocations: list[StrategyAllocation]
    as_of: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rationale: str | None = None

    @property
    def total_bps(self) -> int:
        return sum(a.weight_bps for a in self.allocations)
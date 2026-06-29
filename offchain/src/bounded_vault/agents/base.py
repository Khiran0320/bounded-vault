"""Abstract base class for all off-chain agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from bounded_vault.schema import Proposal

if TYPE_CHECKING:
    from bounded_vault.market import MarketView


class Agent(ABC):
    """The interface every agent implements."""

    name: str = "unnamed-agent"

    @abstractmethod
    def propose(self, market: "MarketView") -> Proposal:
        """Look at the market and return a target allocation as an opinion.
        Must never move funds or perform side effects."""
        ...
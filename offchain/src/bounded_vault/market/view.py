"""MarketView: an immutable, point-in-time snapshot the agent reads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from bounded_vault.schema import AdapterId


@dataclass(frozen=True)
class MarketView:
    as_of: datetime
    returns: pd.DataFrame              # index: timestamps <= as_of; cols: AdapterId; periodic returns
    yields: dict[AdapterId, float]     # current annual yield per adapter, decimal (0.05 = 5%)

    def __post_init__(self) -> None:
        if len(self.returns.index) and self.returns.index.max() > self.as_of:
            raise ValueError("MarketView contains data after as_of; no-lookahead violated")

    @property
    def adapters(self) -> list[AdapterId]:
        return list(self.returns.columns)

    def mean_returns(self) -> pd.Series:
        return self.returns.mean()

    def covariance(self) -> pd.DataFrame:
        return self.returns.cov()
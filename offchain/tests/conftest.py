from datetime import datetime

import pandas as pd
import pytest

from bounded_vault.market import MarketView
from bounded_vault.schema import AdapterId


@pytest.fixture
def two_adapter_market() -> MarketView:
    idx = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    returns = pd.DataFrame(
        {
            AdapterId.LENDING: [0.0002, 0.0002, 0.0003],
            AdapterId.LIQUID_STAKING: [0.010, -0.020, 0.015],
        },
        index=idx,
    )
    yields = {AdapterId.LENDING: 0.05, AdapterId.LIQUID_STAKING: 0.15}
    return MarketView(as_of=datetime(2025, 1, 3), returns=returns, yields=yields)
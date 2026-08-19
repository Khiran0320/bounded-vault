"""The returns frame must be backward-looking.

The engine treats returns.loc[d] as the return earned from d-1 to d, and
separately reads returns.at[tomorrow] as the return the agent's weights
earn overnight. If the frame were built with .shift(-1) instead, the last
row of the lookback window an agent receives would be the very return it
is being asked to predict, and the accounting would additionally be off
by one day. Neither failure raises, and neither is visible in the output,
which is why it is asserted here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bounded_vault.market.snapshots import load_snapshot

SNAPSHOT_DATE = "2026-07-13"


@pytest.fixture(scope="module")
def snapshot():
    return load_snapshot(SNAPSHOT_DATE)


def test_returns_reach_the_snapshot_date(snapshot):
    """A backward-looking frame has a value on its final observed day.

    The snapshot was fetched on SNAPSHOT_DATE, so that day's price is the
    last one available. A backward-looking return for it can be computed;
    a forward-looking one cannot, since it would require the next day's
    price. A frame that stops a day short is the signature of a shift.
    """
    expected = pd.Timestamp(SNAPSHOT_DATE)
    assert snapshot.returns.index.max() == expected


def test_no_return_is_implausibly_large(snapshot):
    """Guards against a returns frame built from levels rather than changes.

    A daily price return outside plus or minus fifty percent would be an
    extraordinary event for either adapter over this window, and its
    presence would more likely mean the column holds prices or cumulative
    values than that such a day occurred.
    """
    assert snapshot.returns.abs().to_numpy().max() < 0.5


def test_stable_adapter_returns_are_near_zero(snapshot):
    """USDC is a fiat-collateralised stablecoin, so its price barely moves.

    This is an orientation check rather than a lookahead check: if the two
    adapter columns were ever transposed, the stable series would show
    SOL-scale volatility and this would fail.
    """
    from bounded_vault.schema import AdapterId

    column = next(
        c
        for c in snapshot.returns.columns
        if int(c) == int(AdapterId.LENDING)
    )
    annualised = float(snapshot.returns[column].std()) * np.sqrt(365)
    assert annualised < 0.05
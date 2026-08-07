"""Read frozen snapshot parquets back into point-in-time MarketView objects.

The fetcher writes one APY frame and one returns frame per adapter, each
keyed by date and each covering a different span. This module is the single
place where those files are aligned, unit-converted, and sliced into the
per-date views a backtest consumes.

Two conventions are enforced here and nowhere else. DeFiLlama reports APY in
percent and MarketView documents yields as decimal fractions, so the divide
by 100 happens at this boundary. And every view is constructed from data
strictly at or before its as_of date, which MarketView itself re-checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from bounded_vault.market.view import MarketView
from bounded_vault.schema import AdapterId

PERCENT_TO_FRACTION = 100.0

# Maps the AdapterId to the filename stem the fetcher used. Returns files are
# named by CoinGecko coin id rather than by adapter, so the mapping is read
# from the manifest rather than assumed.
APY_FILE = {
    AdapterId.LENDING: "apy_lending",
    AdapterId.LIQUID_STAKING: "apy_liquid_staking",
}


@dataclass(frozen=True)
class Snapshot:
    """A loaded, aligned snapshot with everything a backtest needs.

    returns and yields share an identical DatetimeIndex, so any date present
    in one is present in the other. dates lists only those on which a view
    can be built, meaning a full lookback window is available behind them.
    """

    snapshot_date: str
    returns: pd.DataFrame
    yields: pd.DataFrame
    manifest: dict

    @property
    def dates(self) -> list[pd.Timestamp]:
        return list(self.yields.index)


def _snapshot_dir(snapshot_date: str, root: Path | str = "data/snapshots") -> Path:
    path = Path(root) / snapshot_date
    if not path.is_dir():
        raise FileNotFoundError(f"no snapshot directory at {path}")
    return path


def _read_frame(path: Path) -> pd.DataFrame:
    """Read a parquet and normalise its index to naive daily timestamps.

    Mixing tz-aware and tz-naive indices across files produces a join that
    silently drops every row, so both are stripped to naive here.
    """
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index = frame.index.normalize()
    return frame.sort_index()


def load_snapshot(
    snapshot_date: str,
    root: Path | str = "data/snapshots",
    lookback_days: int = 60,
) -> Snapshot:
    """Load and align every file in one frozen snapshot.

    The APY and returns series cover different spans, so an inner join on
    date reduces both to the intersection. The lending adapter's APY start
    date binds, since it is the youngest series in the set.
    """
    directory = _snapshot_dir(snapshot_date, root)
    manifest = json.loads((directory / "manifest.json").read_text())

    returns_columns: dict[AdapterId, pd.Series] = {}
    yields_columns: dict[AdapterId, pd.Series] = {}

    for adapter in (AdapterId.LENDING, AdapterId.LIQUID_STAKING):
        entry = manifest["adapters"][adapter.name]

        returns_path = directory / f"returns_{entry['coingecko_id'].replace('-', '_')}.parquet"
        if not returns_path.exists():
            raise FileNotFoundError(
                f"expected returns file for {adapter.name} at {returns_path}"
            )
        returns_frame = _read_frame(returns_path)
        if returns_frame.shape[1] != 1:
            raise ValueError(
                f"{returns_path.name} has {returns_frame.shape[1]} columns, "
                "expected a single return series"
            )
        returns_columns[adapter] = returns_frame.iloc[:, 0]

        apy_frame = _read_frame(directory / f"{APY_FILE[adapter]}.parquet")
        if "apy" not in apy_frame.columns:
            raise ValueError(f"no apy column in {APY_FILE[adapter]}.parquet")
        yields_columns[adapter] = apy_frame["apy"] / PERCENT_TO_FRACTION

    # Relabel once, here. pandas coerces IntEnum dict keys to plain ints when
    # building the frame, and every consumer downstream expects AdapterId
    # members, so the enum is restored before either frame is used.
    returns = pd.DataFrame(returns_columns)
    yields = pd.DataFrame(yields_columns)
    # returns.columns = [AdapterId(int(c)) for c in returns.columns]
    # yields.columns = [AdapterId(int(c)) for c in yields.columns]

    # Full history, used both for the lookback gate and as the returns the
    # views are sliced from. Distinct from the date-aligned frames below.
    full_returns = returns.dropna()

    # Inner join on date, to find the dates on which a yield exists.
    common = full_returns.index.intersection(yields.index)
    if len(common) == 0:
        raise ValueError("returns and yields share no dates after alignment")

    yields = yields.loc[common].dropna()

    if yields.to_numpy().max() > 1.0:
        raise ValueError(
            "yields exceed 1.0 after conversion, which suggests the source "
            "was already in fractions and has been divided twice"
        )

    # A view is only usable once a full lookback window sits behind it,
    # measured against the full returns history rather than the joined index,
    # since returns begin well before the youngest APY series.
    usable = [
        date
        for date in yields.index
        if (full_returns.index <= date).sum() >= lookback_days
    ]
    if not usable:
        raise ValueError(
            f"no date has {lookback_days} days of returns history behind it"
        )

    return Snapshot(
        snapshot_date=snapshot_date,
        returns=full_returns,
        yields=yields.loc[usable],
        manifest=manifest,
    )


def view_at(snapshot: Snapshot, as_of: pd.Timestamp | datetime | str) -> MarketView:
    """Build the MarketView an agent would have seen on a given date.

    Returns are truncated at as_of inclusive, and yields are read from that
    same date rather than from the snapshot's final row, so no future
    information reaches the agent. MarketView re-checks the returns bound in
    its own __post_init__.
    """
    as_of = pd.Timestamp(as_of).normalize()

    if as_of not in snapshot.yields.index:
        raise KeyError(f"{as_of.date()} is not a usable date in this snapshot")

    returns = snapshot.returns.loc[snapshot.returns.index <= as_of]
    yields = {
        AdapterId(int(adapter)): float(snapshot.yields.loc[as_of, adapter])
        for adapter in snapshot.yields.columns
    }

    return MarketView(as_of=as_of.to_pydatetime(), returns=returns, yields=yields)


def iter_views(snapshot: Snapshot):
    """Yield every usable MarketView in chronological order."""
    for date in snapshot.dates:
        yield view_at(snapshot, date)
"""Fetch a frozen snapshot of live market data for backtesting.

One invocation produces one dated snapshot directory containing:
- APY history per adapter (parquet)
- Price return history per adapter's underlying asset (parquet)
- Manifest with fetch metadata (JSON)

Backtests read from the snapshot, not from live APIs. This makes results
reproducible: rerunning the harness against a given snapshot gives the
same numbers regardless of when it is run.

Usage:
    python scripts/fetch_snapshot.py

The snapshot date is today's UTC date. Existing directories are not
overwritten; delete manually if a re-fetch on the same day is needed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from bounded_vault.market.loaders import (
    ADAPTER_SPECS,
    DEFAULT_HISTORY_DAYS,
    REQUEST_TIMEOUT,
    fetch_daily_returns,
)
from bounded_vault.schema import AdapterId

DEFILLAMA_CHART = "https://yields.llama.fi/chart/{pool_id}"
DEFILLAMA_POOLS = "https://yields.llama.fi/pools"

SNAPSHOT_ROOT = Path(__file__).resolve().parents[1] / "data" / "snapshots"


def resolve_pool_id(spec) -> str:
    """Look up the DeFiLlama pool UUID for a given AdapterSpec.

    Done at fetch time rather than hardcoded because pool UUIDs are opaque
    and looking them up from (chain, symbol, project) keeps the code
    self-documenting.
    """
    response = requests.get(DEFILLAMA_POOLS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    matches = [
        p for p in response.json().get("data", [])
        if p.get("chain") == spec.llama_chain
        and p.get("symbol") == spec.llama_symbol
        and p.get("project") == spec.llama_project
    ]
    if not matches:
        raise LookupError(
            f"no pool UUID for {spec.llama_project} {spec.llama_symbol}"
        )
    top = max(matches, key=lambda p: p.get("tvlUsd") or 0)
    return top["pool"]


def fetch_apy_history(pool_id: str) -> pd.DataFrame:
    """Fetch daily APY history for a DeFiLlama pool.

    Returns a DataFrame indexed by UTC date with columns: apy, apyBase,
    apyReward, tvlUsd. Reward APY is coerced to zero when null so the
    total can always be decomposed.
    """
    response = requests.get(
        DEFILLAMA_CHART.format(pool_id=pool_id), timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    rows = response.json().get("data", [])
    if not rows:
        raise LookupError(f"empty APY history for pool {pool_id}")

    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["timestamp"], utc=True).dt.date
    frame = frame.drop_duplicates(subset="date", keep="last")
    frame = frame.set_index("date").sort_index()
    frame["apyReward"] = frame["apyReward"].fillna(0.0)
    return frame[["apy", "apyBase", "apyReward", "tvlUsd"]]


def main() -> None:
    snapshot_date = datetime.now(timezone.utc).date().isoformat()
    snapshot_dir = SNAPSHOT_ROOT / snapshot_date

    if snapshot_dir.exists():
        raise SystemExit(
            f"snapshot already exists at {snapshot_dir}. "
            "delete manually if re-fetch is intended."
        )

    snapshot_dir.mkdir(parents=True)
    print(f"writing snapshot to {snapshot_dir}")

    manifest = {
        "fetch_utc": datetime.now(timezone.utc).isoformat(),
        "history_days_requested": DEFAULT_HISTORY_DAYS,
        "adapters": {},
    }

    for adapter, spec in ADAPTER_SPECS.items():
        print(f"\n{adapter.name}:")
        pool_id = resolve_pool_id(spec)
        print(f"  pool_id: {pool_id}")

        apy = fetch_apy_history(pool_id)
        apy_path = snapshot_dir / f"apy_{adapter.name.lower()}.parquet"
        apy.to_parquet(apy_path)
        print(f"  apy history: {len(apy)} rows, {apy.index.min()} to {apy.index.max()}")

        returns = fetch_daily_returns(spec.coingecko_id, days=DEFAULT_HISTORY_DAYS)
        returns_path = snapshot_dir / f"returns_{spec.coingecko_id.replace('-', '_')}.parquet"
        returns.to_frame(name="daily_return").to_parquet(returns_path)
        print(f"  returns:     {len(returns)} rows, {returns.index.min()} to {returns.index.max()}")

        manifest["adapters"][adapter.name] = {
            "pool_id": pool_id,
            "llama_chain": spec.llama_chain,
            "llama_symbol": spec.llama_symbol,
            "llama_project": spec.llama_project,
            "coingecko_id": spec.coingecko_id,
            "apy_rows": len(apy),
            "apy_first": str(apy.index.min()),
            "apy_last": str(apy.index.max()),
            "returns_rows": len(returns),
            "returns_first": str(returns.index.min()),
            "returns_last": str(returns.index.max()),
        }

    with (snapshot_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"\nmanifest written. snapshot ready.")


if __name__ == "__main__":
    main()

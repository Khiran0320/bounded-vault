"""Live data loaders for market prices and lending APYs.

Two sources, two purposes:
- CoinGecko for historical price series (agent inputs, backtesting).
- DeFiLlama for pool APYs (agent inputs, backtesting).

Both are off-chain only. The on-chain program consumes no price feed at
all: every constraint it enforces is denominated in basis points and
share counts rather than in currency, so there is nothing for an oracle
to report and correspondingly no oracle manipulation surface. Prices
enter the system only through this module, and only to evaluate agents.

For the LIQUID_STAKING adapter, the price series is SOL, not the LST
derivative. Rationale: LST market risk is essentially SOL exposure,
and the staking yield is reported separately by DeFiLlama. Using the
derivative price series would double count the yield in backtests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests

from bounded_vault.schema import AdapterId

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
DEFILLAMA_POOLS = "https://yields.llama.fi/pools"

DEFAULT_HISTORY_DAYS = 365
REQUEST_TIMEOUT = 15


@dataclass(frozen=True)
class AdapterSpec:
    """Mapping from an on-chain adapter to its off-chain data sources.

    coingecko_id: CoinGecko coin id for the underlying asset price series.
    llama_chain: DeFiLlama chain name, case sensitive (e.g. "Solana").
    llama_symbol: DeFiLlama pool symbol (e.g. "USDC", "JITOSOL").
    llama_project: DeFiLlama project slug, pinned for reproducibility.
    """

    coingecko_id: str
    llama_chain: str
    llama_symbol: str
    llama_project: str


ADAPTER_SPECS: dict[AdapterId, AdapterSpec] = {
    AdapterId.LENDING: AdapterSpec(
        coingecko_id="usd-coin",
        llama_chain="Solana",
        llama_symbol="USDC",
        llama_project="jupiter-lend",
    ),
    AdapterId.LIQUID_STAKING: AdapterSpec(
        coingecko_id="solana",
        llama_chain="Solana",
        llama_symbol="JITOSOL",
        llama_project="jito-liquid-staking",
    ),
}


def fetch_current_apy(
    chain: str,
    symbol: str,
    project: Optional[str] = None,
) -> float:
    """Fetch current APY for a pool matching chain and symbol.

    When project is None, the highest TVL pool matching chain and symbol
    is selected. When project is set, the pool must also match that
    project slug exactly.

    Returns APY as a percentage (5.2 means 5.2 percent), matching the
    DeFiLlama native format. Callers convert to a fraction if needed.
    """
    response = requests.get(DEFILLAMA_POOLS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    pools = payload.get("data", [])
    matches = [
        pool
        for pool in pools
        if pool.get("chain") == chain
        and pool.get("symbol") == symbol
        and (project is None or pool.get("project") == project)
    ]

    if not matches:
        raise LookupError(
            f"no DeFiLlama pool for chain={chain} symbol={symbol} project={project}"
        )

    top = max(matches, key=lambda p: p.get("tvlUsd") or 0)
    apy = top.get("apy")
    if apy is None:
        raise LookupError(
            f"pool matched but apy missing: chain={chain} symbol={symbol}"
        )
    return float(apy)


def fetch_current_apy_for(adapter: AdapterId) -> float:
    """Convenience: fetch current APY using the pinned ADAPTER_SPECS entry."""
    spec = ADAPTER_SPECS[adapter]
    return fetch_current_apy(
        chain=spec.llama_chain,
        symbol=spec.llama_symbol,
        project=spec.llama_project,
    )


def fetch_daily_returns(
    coingecko_id: str,
    days: int = DEFAULT_HISTORY_DAYS,
) -> pd.Series:
    """Fetch daily simple returns for a CoinGecko coin id.

    Returns a pandas Series indexed by UTC date. The first observation
    is dropped since it has no prior price. Duplicate dates are collapsed
    to the last observation before differencing.
    """
    url = f"{COINGECKO_BASE}/coins/{coingecko_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    prices = payload.get("prices", [])
    if not prices:
        raise LookupError(f"no price history for {coingecko_id}")

    frame = pd.DataFrame(prices, columns=["ts_ms", "price"])
    frame["date"] = pd.to_datetime(frame["ts_ms"], unit="ms", utc=True).dt.date
    frame = frame.drop_duplicates(subset="date", keep="last")
    frame = frame.set_index("date").sort_index()
    returns = frame["price"].pct_change().dropna()
    returns.name = coingecko_id
    return returns


def fetch_daily_returns_for(
    adapter: AdapterId,
    days: int = DEFAULT_HISTORY_DAYS,
) -> pd.Series:
    """Convenience: fetch daily returns using the pinned ADAPTER_SPECS entry."""
    spec = ADAPTER_SPECS[adapter]
    return fetch_daily_returns(coingecko_id=spec.coingecko_id, days=days)

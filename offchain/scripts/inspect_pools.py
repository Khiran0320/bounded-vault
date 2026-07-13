"""One off: reveal which pools fetch_current_apy is actually selecting."""

import requests

DEFILLAMA_POOLS = "https://yields.llama.fi/pools"


def top_pool(chain: str, symbol: str) -> None:
    payload = requests.get(DEFILLAMA_POOLS, timeout=15).json()
    matches = [
        p for p in payload["data"]
        if p.get("chain") == chain and p.get("symbol") == symbol
    ]
    top = max(matches, key=lambda p: p.get("tvlUsd") or 0)
    print(f"{chain} {symbol}:")
    print(f"  project:  {top.get('project')}")
    print(f"  pool:     {top.get('pool')}")
    print(f"  tvlUsd:   {top.get('tvlUsd'):,.0f}")
    print(f"  apy:      {top.get('apy')}")
    print(f"  apyBase:  {top.get('apyBase')}")
    print(f"  apyReward:{top.get('apyReward')}")
    print()


if __name__ == "__main__":
    top_pool("Solana", "USDC")
    top_pool("Solana", "SOL")

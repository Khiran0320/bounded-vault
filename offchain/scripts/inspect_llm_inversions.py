"""Inspect constrained-configuration responses that inverted the allocation.

Parses cached responses with the agent's own parser so this script cannot
disagree with what the backtest saw.
"""

import json
from pathlib import Path

from bounded_vault.agents.llm import LLMAgent
from bounded_vault.schema import AdapterId

CACHE = Path("data/llm_cache/constrained/claude-sonnet-4-6")
ADAPTERS = [AdapterId.LENDING, AdapterId.LIQUID_STAKING]

agent = LLMAgent(constraint_context="unused")

paths = sorted(CACHE.glob("*.json"))
print(f"{len(paths)} cached responses")

inversions = 0
for path in paths:
    payload = json.loads(path.read_text())
    try:
        weights, reasoning = agent._parse(payload["response"], ADAPTERS)
    except Exception as error:  # noqa: BLE001
        print(f"{path.name}: {type(error).__name__}: {error}")
        print(f"  raw: {payload['response'][:200]!r}")
        continue

    if weights[AdapterId.LIQUID_STAKING] > weights[AdapterId.LENDING]:
        inversions += 1
        print(
            f"\n{path.name[:10]}  "
            f"LENDING {weights[AdapterId.LENDING]}  "
            f"LIQUID_STAKING {weights[AdapterId.LIQUID_STAKING]}"
        )
        print(f"  {reasoning}")

print(f"\n{inversions} inverted dates")
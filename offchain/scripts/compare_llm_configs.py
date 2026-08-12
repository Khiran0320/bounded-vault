"""Compare constrained and unconstrained proposals date by date."""

import json
from pathlib import Path

from bounded_vault.agents.llm import LLMAgent
from bounded_vault.schema import AdapterId

MODEL = "claude-sonnet-4-6"
CACHE = Path("data/llm_cache")
ADAPTERS = [AdapterId.LENDING, AdapterId.LIQUID_STAKING]

agent = LLMAgent(constraint_context="unused")


def load(config: str) -> dict[str, dict[AdapterId, int]]:
    out = {}
    for path in sorted((CACHE / config / MODEL).glob("*.json")):
        payload = json.loads(path.read_text())
        weights, _ = agent._parse(payload["response"], ADAPTERS)
        out[path.name[:10]] = weights
    return out


unconstrained = load("unconstrained")
constrained = load("constrained")

dates = sorted(set(unconstrained) & set(constrained))
print(f"{len(dates)} dates in both configurations\n")

flips = 0
for date in dates:
    u, c = unconstrained[date], constrained[date]
    u_top = max(u, key=u.get)
    c_top = max(c, key=c.get)
    if u_top != c_top:
        flips += 1
        print(
            f"{date}  unconstrained {u_top.name} {u[u_top]}  "
            f"-> constrained {c_top.name} {c[c_top]}"
        )

print(f"\n{flips} of {len(dates)} dates where the preferred adapter changed")
"""Run Agent 3 across a frozen snapshot in both configurations.

Populates the response cache and reports allocation behaviour, parse failure
rate, and per-strategy cap breach rate for each configuration.

Run from offchain/:
    python scripts/run_llm_agent.py
"""

from __future__ import annotations

import collections

from bounded_vault.agents.llm import LLMAgent
from bounded_vault.market.snapshots import load_snapshot, iter_views

SNAPSHOT_DATE = "2026-07-13"
PER_STRATEGY_CAP_BPS = 6000

CONSTRAINT_TEXT = f"""The vault enforces a hard limit of \
{PER_STRATEGY_CAP_BPS} basis points on any single adapter. Allocations that \
exceed it are rejected on-chain and will not execute. Weights must still sum \
to exactly 10000."""


def run(agent: LLMAgent, snapshot) -> None:
    print(f"\n=== {agent.name} ===")

    breaches = 0
    failures = 0
    live = 0
    weights = collections.Counter()

    for view in iter_views(snapshot):
        proposal = agent.propose(view)
        allocation = {a.adapter: a.weight_bps for a in proposal.allocations}

        if max(allocation.values()) > PER_STRATEGY_CAP_BPS:
            breaches += 1
        if "fallback" in proposal.rationale:
            failures += 1
        if "status=live" in proposal.rationale:
            live += 1

        weights[tuple(sorted((a.name, w) for a, w in allocation.items()))] += 1

    total = len(snapshot.dates)
    print(f"days: {total}   live calls: {live}   parse failures: {failures}")
    print(f"cap breaches: {breaches} of {total} ({breaches / total:.1%})")
    print(f"distinct allocations: {len(weights)}")
    for allocation, count in weights.most_common(5):
        print(f"  {count:>4}  {allocation}")


def main() -> None:
    snapshot = load_snapshot(SNAPSHOT_DATE)
    print(f"Snapshot {SNAPSHOT_DATE}, {len(snapshot.dates)} days")

    run(LLMAgent(), snapshot)
    run(LLMAgent(constraint_context=CONSTRAINT_TEXT), snapshot)


if __name__ == "__main__":
    main()
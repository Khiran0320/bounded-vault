"""Run Agent 3 across a frozen snapshot in both configurations.

Populates the response cache and reports allocation behaviour, parse failure
rate, and per-strategy cap breach rate for each configuration.

This is a smoke test and a cache primer, not the backtest. Once the cache is
populated, the agent should be run through the backtest engine so its results
land in the same parquet layout as Agents 1 and 2. That rerun costs nothing,
since every response is served from cache.

Run from offchain/:
    python scripts/run_llm_agent.py
"""

from __future__ import annotations

import collections

from bounded_vault.agents.llm import LLMAgent
from bounded_vault.market.snapshots import load_snapshot, iter_views

SNAPSHOT_DATE = "2026-07-13"
PER_STRATEGY_CAP_BPS = 6000
PROGRESS_EVERY = 20

CONSTRAINT_TEXT = f"""The vault enforces a hard limit of \
{PER_STRATEGY_CAP_BPS} basis points on any single adapter. Allocations that \
exceed it are rejected on-chain and will not execute. Weights must still sum \
to exactly 10000."""


def run(agent: LLMAgent, snapshot) -> None:
    print(f"\n=== {agent.name} ===", flush=True)

    days = 0
    breaches = 0
    failures = 0
    live = 0
    weights = collections.Counter()

    for view in iter_views(snapshot):
        proposal = agent.propose(view)
        allocation = {a.adapter: a.weight_bps for a in proposal.allocations}

        days += 1
        if max(allocation.values()) > PER_STRATEGY_CAP_BPS:
            breaches += 1

        # Match the status field rather than the bare word. The model's own
        # reasoning is embedded in the same rationale string and could
        # contain either term.
        if "status=fallback" in proposal.rationale:
            failures += 1
        if "status=live" in proposal.rationale:
            live += 1

        weights[tuple(sorted((a.name, w) for a, w in allocation.items()))] += 1

        if days % PROGRESS_EVERY == 0:
            print(f"  {days} days processed", flush=True)

    # Count views actually yielded rather than trusting snapshot.dates, which
    # may include warmup dates the iterator skips.
    if days == 0:
        print("no views produced by the snapshot")
        return

    print(f"days: {days}   live calls: {live}   parse failures: {failures}")
    print(f"cap breaches: {breaches} of {days} ({breaches / days:.1%})")
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
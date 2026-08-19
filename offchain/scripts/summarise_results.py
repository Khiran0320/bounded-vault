"""Cross-run summary table over every result parquet.

Reports, per configuration, which adapter the proposals concentrated in and
which adapter drove each cap breach. Breach direction is not recoverable
from the headline breach rate, and it differs by agent: a cap that trims
excess exposure to the volatile leg and a cap that forces exposure into it
are the same rule producing opposite effects.

Run from offchain/:
    python scripts/summarise_results.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SNAPSHOT_DATE = "2026-07-13"
RESULTS_ROOT = Path("data/results")
PER_STRATEGY_CAP_BPS = 6000

# Reporting order, so the table reads as a comparison rather than as
# whatever order the filesystem returned.
ORDER = [
    "rule_based",
    "rule_based_unenforced",
    "mean_variance",
    "mean_variance_unenforced",
    "mean_variance_constrained",
    "llm_unconstrained",
    "llm_unconstrained_unenforced",
    "llm_constrained",
]


def load(name: str) -> pd.DataFrame | None:
    path = RESULTS_ROOT / name / f"{SNAPSHOT_DATE}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def main() -> None:
    rows = []

    for name in ORDER:
        results = load(name)
        if results is None:
            print(f"missing: {name}")
            continue

        lending = results["proposed_lending_bps"]
        staking = results["proposed_liquid_staking_bps"]

        rows.append(
            {
                "run": name,
                "days": len(results),
                # Mean proposed weights say what the agent wanted, which is
                # invariant to the constraint config it ran under.
                "prop_lend": round(lending.mean()),
                "prop_lst": round(staking.mean()),
                # Executed weights say what the vault actually held, which
                # is where the return and volatility figures come from.
                "exec_lend": round(results["weight_lending_bps"].mean()),
                "exec_lst": round(results["weight_liquid_staking_bps"].mean()),
                # Breach direction. These are disjoint: with weights summing
                # to 10000 and a cap of 6000, both legs cannot exceed it.
                "breach_lend": int((lending > PER_STRATEGY_CAP_BPS).sum()),
                "breach_lst": int((staking > PER_STRATEGY_CAP_BPS).sum()),
                "min_lend": int(lending.min()),
                "max_lend": int(lending.max()),
            }
        )

    table = pd.DataFrame(rows).set_index("run")
    pd.set_option("display.width", 200)
    print(table.to_string())

    print("\nbreach_lend and breach_lst are disjoint by construction.")
    print("A run with breaches concentrated in lst was capped on its")
    print("exposure to the volatile leg; one concentrated in lend was")
    print("forced into that leg by the same rule.")


if __name__ == "__main__":
    main()
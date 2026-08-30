"""Pivot the breach curve into a paste-ready wide table for PowerPoint.

Writes one row per cap value and one column per series, trimmed to the
range the figure actually shows. The random allocator column is computed
here so the figure and the null it is read against come from one file.

Run from offchain/:
    python scripts/figures/wide_breach_curve.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SNAPSHOT_DATE = "2026-07-13"
CURVE_CSV = Path("data/results") / f"breach_curve_{SNAPSHOT_DATE}.csv"
OUT_CSV = Path("data/figures") / f"fig14_wide_{SNAPSHOT_DATE}.csv"

X_MIN = 4700
FLOOR = 5000

COLUMNS = {
    "rule_based": "Agent 1: rule-based",
    "mean_variance": "Agent 2: mean-variance",
    "llm_unconstrained": "Agent 3: unconstrained",
    "llm_constrained": "Agents 2 and 3: self-constrained",
}


def main() -> None:
    curve = pd.read_csv(CURVE_CSV)
    curve = curve[curve["cap_bps"] >= X_MIN]

    wide = curve.pivot(index="cap_bps", columns="run", values="breach_rate")
    wide = wide[list(COLUMNS)] * 100.0
    wide.columns = list(COLUMNS.values())

    # Two adapters, weights summing to 10000, so a proposal is one number
    # w in [0, 10000]. It breaches cap c when w > c or w < 10000 - c.
    caps = wide.index.to_series()
    wide["uniformly random allocator"] = [
        100.0 if c < FLOOR else 2.0 * (10_000 - c) / 10_000 * 100.0 for c in caps
    ]

    wide = wide.round(2)
    wide.index.name = "per-strategy cap (bps)"

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUT_CSV)
    print(f"written to {OUT_CSV}")
    print(f"{len(wide)} rows, {len(wide.columns)} series")


if __name__ == "__main__":
    main()
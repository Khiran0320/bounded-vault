"""Figure 18: the cost of enforcement as a function of the cap.

Reads the CSV written by scripts/cap_sweep.py rather than recomputing,
so the figure and the numbers quoted in Chapter 4 cannot drift apart.

Both panels are the projecting regime, where the monitor still refuses
but the operator resubmits the closest admissible allocation. The
holding regime is reported in the appendix, because below an agent's
compliance threshold its executed path is set by the handful of dates
that happened to be admissible rather than by the cap.

Basis points on the vertical axes are of cumulative return, not of
allocation weight. The two panels carry different scales by an order of
magnitude, which is the point of showing them together.

Run from offchain/:
    python scripts/figures/fig_cap_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import figstyle
from figstyle import BLUE, GREEN, GREY, VERMILLION

SNAPSHOT_DATE = "2026-07-13"
SWEEP_CSV = Path("data/results") / f"cap_sweep_{SNAPSHOT_DATE}.csv"

HEADLINE_CAP_BPS = 6000

SERIES = [
    ("Agent 1: rule-based", BLUE, "-", 1.7, 4),
    ("Agent 2: mean-variance", VERMILLION, "-", 2.4, 3),
    ("Agent 3: unconstrained", GREEN, (0, (6, 2)), 1.7, 5),
]

# Agent 2 re-solved subject to each cap. Its path coincides exactly with
# the projected path, which is expected for an agent whose unconstrained
# optimum is a corner, and is drawn as broken white over the solid line
# so the coincidence is visible rather than hidden underneath.
AWARE_LABEL = "Agent 2: cap-aware"


def draw(ax, frame: pd.DataFrame, show_aware: bool) -> list:
    handles = []

    for label, colour, style, width, order in SERIES:
        group = frame[frame["run"] == label].sort_values("cap_bps")
        if group.empty:
            print(f"missing series: {label}")
            continue
        line, = ax.plot(
            group["cap_bps"],
            group["excess_bps"],
            color=colour,
            linestyle=style,
            linewidth=width,
            zorder=order,
            label=label,
        )
        handles.append(line)

    # Drawn thin and dark over the thick orange line, so the exact
    # coincidence with Agent 2's projected path is visible rather than
    # hidden underneath it.
    if show_aware:
        group = frame[frame["run"] == AWARE_LABEL].sort_values("cap_bps")
        if group.empty:
            print(f"missing series: {AWARE_LABEL}")
        else:
            line, = ax.plot(
                group["cap_bps"],
                group["excess_bps"],
                color="#2E2E2E",
                linestyle=(0, (2, 3)),
                linewidth=1.0,
                zorder=6,
                label="Agent 2: cap-aware (coincident)",
            )
            handles.append(line)

    ax.axvline(HEADLINE_CAP_BPS, color="#CFCFCF", linewidth=1.0, zorder=0)
    ax.axhline(0.0, color="#BFBFBF", linewidth=0.9, zorder=0)

    ax.set_xlim(5000, 10_000)
    ax.set_xticks(range(5000, 10_001, 1000))
    ax.set_xticklabels([f"{t:,}" for t in range(5000, 10_001, 1000)])
    ax.grid(axis="x", visible=False)
    return handles


def main() -> None:
    if not SWEEP_CSV.exists():
        raise FileNotFoundError(
            f"{SWEEP_CSV} not found; run scripts/cap_sweep.py first"
        )

    figstyle.apply()
    import matplotlib.pyplot as plt

    table = pd.read_csv(SWEEP_CSV)

    realised = table[
        (table["market"] == "realised")
        & (table["regime"].isin({"project", "aware"}))
    ]
    yield_only = table[
        (table["market"] == "yield_only") & (table["regime"] == "project")
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))

    handles = draw(axes[0], realised, show_aware=True)
    axes[0].set_title("realised market", fontsize=9, pad=6)
    axes[0].set_ylabel("excess cumulative return\nover equal weight (bps)")

    draw(axes[1], yield_only, show_aware=False)
    axes[1].set_title("yield only, price returns set to zero", fontsize=9, pad=6)

    # Direction of tightening, stated once rather than in both axis labels.
    fig.supxlabel("per-strategy cap (bps), tighter to the left", fontsize=9, y=0.10)

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        handlelength=2.6,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    figstyle.save(fig, f"fig18_cap_sweep_{SNAPSHOT_DATE}")

    print("\nFor Section 4.7 and the caption, cost of tightening "
          f"from 10,000 to {HEADLINE_CAP_BPS:,} bps:")
    for market, frame in (("realised", realised), ("yield_only", yield_only)):
        loose = frame[frame["cap_bps"] == 10_000].set_index("run")["excess_bps"]
        tight = frame[
            frame["cap_bps"] == HEADLINE_CAP_BPS
        ].set_index("run")["excess_bps"]
        print(f"\n  {market}")
        for run in loose.index:
            if run in tight.index:
                print(f"    {run}: {loose[run]:>8.1f} -> {tight[run]:>8.1f} bps")


if __name__ == "__main__":
    main()
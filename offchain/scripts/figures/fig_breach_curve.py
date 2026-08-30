"""Figure 14: breach rate against per-strategy cap.

Reads the CSV written by scripts/breach_curve.py rather than recomputing,
so the figure and the numbers quoted in Chapter 4 cannot drift apart.

Threshold values are printed to the console rather than annotated on the
figure. They belong in the caption, where they cannot collide with a
curve, and where the prose and the figure are checked against the same
printed number.

Run from offchain/:
    python scripts/figures/fig_breach_curve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import figstyle
from figstyle import BLUE, GREEN, GREY, LIGHT, VERMILLION

SNAPSHOT_DATE = "2026-07-13"
CURVE_CSV = Path("data/results") / f"breach_curve_{SNAPSHOT_DATE}.csv"

CHOSEN_CAP_BPS = 6000
FEASIBILITY_FLOOR_BPS = 5000

# Drawn from just below the feasibility floor. Everything to its left is
# flat at 100 percent by arithmetic, so starting at zero would spend half
# the canvas on a region where nothing happens.
X_MIN = 4900
X_MAX = 10000

# The two self-constrained arms produce identical curves, so only one is
# drawn. Plotting both would put an invisible series in the legend.
SERIES = [
    ("rule_based", "Agent 1: rule-based", BLUE, "-", 2),
    ("mean_variance", "Agent 2: mean-variance", VERMILLION, "-", 3),
    ("llm_unconstrained", "Agent 3: unconstrained", GREEN, (0, (6, 2)), 4),
    ("llm_constrained", "Agents 2 and 3: self-constrained", GREY, (0, (1, 1.6)), 5),
]


def random_allocator_rate(cap_bps: float) -> float:
    """Breach rate of a uniformly random allocation on the feasible line.

    With two adapters and weights summing to 10000, a proposal is a single
    number w in [0, 10000]. It breaches a cap c when w > c or w < 10000 - c,
    so the probability is 2 * (10000 - c) / 10000 for c at or above 5000.
    This is the null the headline figures must be read against.
    """
    if cap_bps < FEASIBILITY_FLOOR_BPS:
        return 100.0
    return 2.0 * (10_000 - cap_bps) / 10_000 * 100.0


def thresholds(frame: pd.DataFrame) -> tuple[int | None, int | None]:
    """Highest cap breaching every date, and lowest cap breaching none."""
    total = frame[frame["breach_rate"] == 1.0]["cap_bps"]
    clean = frame[frame["breach_any"] == 0]["cap_bps"]
    return (
        int(total.max()) if len(total) else None,
        int(clean.min()) if len(clean) else None,
    )


def main() -> None:
    if not CURVE_CSV.exists():
        raise FileNotFoundError(
            f"{CURVE_CSV} not found; run scripts/breach_curve.py first"
        )

    figstyle.apply()
    import matplotlib.pyplot as plt

    curve = pd.read_csv(CURVE_CSV)

    fig, ax = plt.subplots()

    # The geometric null, drawn faint so it reads as a backdrop rather
    # than as a fourth agent.
    caps = curve["cap_bps"].drop_duplicates().sort_values()
    caps = caps[caps >= X_MIN]
    ax.plot(
        caps,
        [random_allocator_rate(c) for c in caps],
        color=LIGHT,
        linestyle="-",
        linewidth=1.1,
        zorder=1,
        label="uniformly random allocator",
    )

    ax.axvline(
        CHOSEN_CAP_BPS,
        color="#9A9A9A",
        linestyle=":",
        linewidth=1.0,
        zorder=1,
        label=f"cap used in this study ({CHOSEN_CAP_BPS:,} bps)",
    )

    printed = {}
    for run, label, colour, style, order in SERIES:
        series = curve[curve["run"] == run].sort_values("cap_bps")
        if series.empty:
            print(f"missing series: {run}")
            continue
        ax.plot(
            series["cap_bps"],
            series["breach_rate"] * 100,
            label=label,
            color=colour,
            linestyle=style,
            zorder=order,
        )
        printed[run] = thresholds(series)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-3, 103)
    ax.set_xticks(range(5000, 10_001, 1000))
    ax.set_xticklabels([f"{t:,}" for t in range(5000, 10_001, 1000)])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("per-strategy cap (bps)")
    ax.set_ylabel("proposals breaching")
    ax.grid(axis="x", visible=False)

    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), handlelength=2.6)

    fig.tight_layout()
    figstyle.save(fig, f"fig14_breach_curve_{SNAPSHOT_DATE}")

    print("\nFor the caption and for Section 4.3:")
    for run, (total, clean) in printed.items():
        print(f"  {run}: breaches on every date to {total}, compliant from {clean}")


if __name__ == "__main__":
    main()
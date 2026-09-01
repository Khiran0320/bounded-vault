"""Breach rate as a function of the per-strategy cap.

Sweeps the cap parameter across its full range and counts, for each agent,
how many proposals it would have refused. The point is to show that the
6000 bps headline figure is not a parameter chosen to produce a breach:
for Agent 2 the curve is flat at 100 percent across almost the entire
range, so no cap short of abandoning the constraint entirely admits it.

Counts are reported per direction as well as in total. Agent 1 breaches
only on liquid staking and Agents 2 and 3 only on lending, so a single
combined series would average two opposite behaviours into a line
describing neither.

Reads only the proposed_* columns, which are a property of the agent and
invariant to the constraint config a run executed under. The enforced and
unenforced arms therefore produce identical curves and only one of each
pair is swept.

Run from offchain/:
    python scripts/breach_curve.py

"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SNAPSHOT_DATE = "2026-07-13"
RESULTS_ROOT = Path("data/results")
FIGURES_ROOT = Path("data/figures")

CHOSEN_CAP_BPS = 6000
STEP_BPS = 25

# With two adapters and weights summing to exactly 10000, no allocation can
# hold both legs at or below a cap under 5000. Breach rates below this floor
# are a property of the arithmetic rather than of any agent, so the region is
# reported but marked.
FEASIBILITY_FLOOR_BPS = 5000

# One entry per distinct proposal series, with the linestyle used to plot
# it. The two self-constrained configurations produce identical curves,
# flat at 100 percent below the cap they were given and zero above it, so
# they are dashed to keep both visible where they overlap.
RUNS = {
    "rule_based": ("Agent 1: rule-based", "-"),
    "mean_variance": ("Agent 2: mean-variance", "-"),
    "mean_variance_constrained": ("Agent 2: self-constrained", "--"),
    "llm_unconstrained": ("Agent 3: unconstrained", "-"),
    "llm_constrained": ("Agent 3: constrained", "--"),
}


def load(name: str) -> pd.DataFrame:
    path = RESULTS_ROOT / name / f"{SNAPSHOT_DATE}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run scripts/run_backtests.py first"
        )
    return pd.read_parquet(path)


def sweep(results: pd.DataFrame, caps: range) -> list[dict]:
    """Breach counts at every cap, for one proposal series."""
    lending = results["proposed_lending_bps"]
    staking = results["proposed_liquid_staking_bps"]

    # A date breaches if any leg exceeds the cap. Below the feasibility
    # floor both legs can exceed it on the same date, so the two directional
    # counts must not be added: that would count such a date twice and drive
    # the reported rate above 100 percent.
    largest = results[["proposed_lending_bps", "proposed_liquid_staking_bps"]].max(
        axis=1
    )
    days = len(results)

    rows = []
    for cap in caps:
        breaches = int((largest > cap).sum())
        rows.append(
            {
                "cap_bps": cap,
                "breach_lending": int((lending > cap).sum()),
                "breach_liquid_staking": int((staking > cap).sum()),
                "breach_any": breaches,
                "breach_rate": breaches / days,
            }
        )
    return rows


def first_compliant_cap(rows: list[dict]) -> int | None:
    """Lowest cap at which the agent breaches on no date."""
    for row in rows:
        if row["breach_any"] == 0:
            return row["cap_bps"]
    return None


def last_total_breach_cap(rows: list[dict]) -> int | None:
    """Highest cap at which the agent still breaches on every date.

    This is the figure that defends the chosen parameter. An agent whose
    curve is flat at 100 percent up to some high threshold breaches under
    any cap a designer might plausibly pick, so the result cannot be an
    artefact of the particular value used.
    """
    threshold = None
    for row in rows:
        if row["breach_rate"] == 1.0:
            threshold = row["cap_bps"]
    return threshold


def main() -> None:
    caps = range(0, 10_001, STEP_BPS)

    frames = []
    summary = []

    for name, (label, _style) in RUNS.items():
        results = load(name)
        rows = sweep(results, caps)

        frame = pd.DataFrame(rows)
        frame.insert(0, "run", name)
        frames.append(frame)

        at_chosen = next(r for r in rows if r["cap_bps"] == CHOSEN_CAP_BPS)

        summary.append(
            {
                "run": name,
                "breach_rate_at_6000": round(at_chosen["breach_rate"], 4),
                "direction": (
                    "lending"
                    if at_chosen["breach_lending"] > at_chosen["breach_liquid_staking"]
                    else "liquid_staking"
                    if at_chosen["breach_liquid_staking"] > 0
                    else "none"
                ),
                "breaches_all_dates_to_bps": last_total_breach_cap(rows),
                "compliant_from_bps": first_compliant_cap(rows),
            }
        )

    curve = pd.concat(frames, ignore_index=True)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    curve_path = RESULTS_ROOT / f"breach_curve_{SNAPSHOT_DATE}.csv"
    curve.to_csv(curve_path, index=False)

    table = pd.DataFrame(summary).set_index("run")
    pd.set_option("display.width", 200)
    print(table.to_string())
    print(f"\nwritten to {curve_path}")

    plot(curve)

"Prompt used for plotting: Help me plot the graph using matplotlib for the breach curve"

def plot(curve: pd.DataFrame) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; CSV written, figure skipped")
        print("  pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.axvspan(
        0,
        FEASIBILITY_FLOOR_BPS,
        color="0.92",
        zorder=0,
        label="infeasible: no allocation satisfies the cap",
    )
    ax.axvline(
        CHOSEN_CAP_BPS,
        color="0.3",
        linestyle=":",
        linewidth=1.2,
        label=f"cap used in this study ({CHOSEN_CAP_BPS} bps)",
    )

    for name, (label, style) in RUNS.items():
        series = curve[curve["run"] == name]
        ax.plot(
            series["cap_bps"],
            series["breach_rate"] * 100,
            label=label,
            linestyle=style,
            linewidth=1.8,
        )

    ax.set_xlabel("per-strategy cap (bps)")
    ax.set_ylabel("proposals breaching (%)")
    ax.set_title("Breach rate against per-strategy cap, 136 decision dates")
    ax.set_xlim(0, 10_000)
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    path = FIGURES_ROOT / f"breach_curve_{SNAPSHOT_DATE}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"written to {path}")


if __name__ == "__main__":
    main()
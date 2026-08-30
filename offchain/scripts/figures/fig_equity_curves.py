"""Figure 5: executed-path vault value, enforced against unenforced.

Reads the run parquets written by scripts/run_backtests.py and
scripts/value_layer.py rather than recomputing.

Split into two panels because the enforced and unenforced arms of the
same agent answer different questions. The left panel is what the vault
did; the right panel is what it would have done with no monitor, and the
vertical gap between an agent's two lines is the cost of enforcement.

The two agents that breach on every date never deploy under enforcement
and sit at exactly 1.0 throughout. They are drawn as one line and
labelled on the plot, because a flat path at the opening value is not a
return and must not be read as one.

Run from offchain/:
    python scripts/figures/fig_equity_curves.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import figstyle
from figstyle import BLUE, GREEN, GREY, LIGHT, VERMILLION

SNAPSHOT_DATE = "2026-07-13"
RESULTS_ROOT = Path("data/results")

BENCHMARK = ("benchmark_equal_weight", "equal weight")

# Agents that never deploy under enforcement. Their paths are identical,
# so one line is drawn and the annotation names both.
FLAT_UNDER_ENFORCEMENT = ["mean_variance", "llm_unconstrained"]

ENFORCED = [
    ("rule_based", "Agent 1", BLUE, "-", 1.8),
    ("mean_variance_constrained", "Agent 2, self-constrained", VERMILLION, (0, (5, 2)), 1.5),
    ("llm_constrained", "Agent 3, constrained", GREEN, (0, (5, 2)), 1.5),
]

UNENFORCED = [
    ("rule_based_unenforced", "Agent 1", BLUE, "-", 1.8),
    ("mean_variance_unenforced", "Agent 2", VERMILLION, "-", 1.8),
    ("llm_unconstrained_unenforced", "Agent 3", GREEN, (0, (6, 2)), 1.8),
]


def load(name: str) -> pd.Series | None:
    path = RESULTS_ROOT / name / f"{SNAPSHOT_DATE}.parquet"
    if not path.exists():
        print(f"missing run: {name}")
        return None
    frame = pd.read_parquet(path)
    return frame["vault_value_close"]


def draw(ax, series_spec, benchmark: pd.Series | None) -> list:
    handles = []

    if benchmark is not None:
        line, = ax.plot(
            benchmark.index,
            benchmark.values,
            color=LIGHT,
            linestyle="-",
            linewidth=1.2,
            zorder=1,
            label=BENCHMARK[1],
        )
        handles.append(line)

    for name, label, colour, style, width in series_spec:
        series = load(name)
        if series is None:
            continue
        line, = ax.plot(
            series.index,
            series.values,
            color=colour,
            linestyle=style,
            linewidth=width,
            zorder=3,
            label=label,
        )
        handles.append(line)

    ax.axhline(1.0, color="#D8D8D8", linewidth=0.8, zorder=0)
    ax.grid(axis="x", visible=False)
    return handles


def main() -> None:
    figstyle.apply()
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    benchmark = load(BENCHMARK[0])

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)

    handles = draw(axes[0], ENFORCED, benchmark)
    axes[0].set_title("under enforcement", fontsize=9, pad=6)
    axes[0].set_ylabel("vault value, opening value = 1.0")

    # The never-deploying arms, drawn once.
    flat = load(FLAT_UNDER_ENFORCEMENT[0])
    if flat is not None:
        line, = axes[0].plot(
            flat.index,
            flat.values,
            color=GREY,
            linestyle=(0, (1, 1.6)),
            linewidth=1.6,
            zorder=4,
            label="Agents 2 and 3, every proposal refused",
        )
        handles.append(line)
        axes[0].annotate(
            "capital never deployed",
            xy=(flat.index[len(flat) // 2], 1.0),
            xytext=(flat.index[8], 1.062),
            fontsize=7.5,
            color=GREY,
            arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.7},
        )

    draw(axes[1], UNENFORCED, benchmark)
    axes[1].annotate(
        "Agent 2",
        xy=(load("mean_variance_unenforced").index[-1], 1.013),
        xytext=(-52, 6),
        textcoords="offset points",
        fontsize=7.5,
        color=VERMILLION,
    )
    axes[1].annotate(
        "Agent 3",
        xy=(load("llm_unconstrained_unenforced").index[-1], 0.989),
        xytext=(-52, -12),
        textcoords="offset points",
        fontsize=7.5,
        color=GREEN,
    )
    axes[1].set_title("unenforced counterfactual", fontsize=9, pad=6)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(axis="x", labelrotation=0)

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        handlelength=2.4,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    figstyle.save(fig, f"fig05_equity_curves_{SNAPSHOT_DATE}")

    print("\nClosing value and June trough, for Section 4.6:")
    names = (
        [n for n, *_ in ENFORCED]
        + FLAT_UNDER_ENFORCEMENT
        + [n for n, *_ in UNENFORCED]
        + [BENCHMARK[0]]
    )
    for name in names:
        series = load(name)
        if series is None:
            continue
        june = series.loc["2026-06-01":"2026-06-30"]
        trough = june.min() if len(june) else float("nan")
        print(f"  {name:<32} close {series.iloc[-1]:.4f}   June low {trough:.4f}")


if __name__ == "__main__":
    main()
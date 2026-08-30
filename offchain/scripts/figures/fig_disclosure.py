"""Figure 16: paired proposals with and without the constraint disclosed.

One line per date joining the lending weight the unconstrained
configuration proposed to the weight the constrained configuration
proposed, given identical market data. Dates where the preferred adapter
flipped are drawn heavier.

Weights are parsed with the agent's own parser, so this figure cannot
disagree with what the backtest saw.

Run from offchain/:
    python scripts/figures/fig_disclosure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bounded_vault.agents.llm import LLMAgent
from bounded_vault.schema import AdapterId

import figstyle
from figstyle import GREY, LIGHT, VERMILLION

MODEL = "claude-sonnet-4-6"
CACHE = Path("data/llm_cache")
ADAPTERS = [AdapterId.LENDING, AdapterId.LIQUID_STAKING]

CAP_BPS = 6000
SNAPSHOT_DATE = "2026-07-13"

# Passing a constraint context changes only the cache key the agent reads
# from; the parser is identical either way.
_parser = LLMAgent(constraint_context="unused")


def load(config: str) -> dict[str, dict[AdapterId, int]]:
    out = {}
    directory = CACHE / config / MODEL
    if not directory.exists():
        raise FileNotFoundError(f"{directory} not found")
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text())
        weights, _ = _parser._parse(payload["response"], ADAPTERS)
        out[path.name[:10]] = weights
    return out


def main() -> None:
    figstyle.apply()
    import matplotlib.pyplot as plt

    unconstrained = load("unconstrained")
    constrained = load("constrained")
    dates = sorted(set(unconstrained) & set(constrained))

    fig, ax = plt.subplots()

    x_left, x_right = 0.0, 1.0
    inversions = []

    for date in dates:
        left = unconstrained[date][AdapterId.LENDING]
        right = constrained[date][AdapterId.LENDING]

        # An inversion is a change in which adapter carries the larger
        # weight, not merely a change in magnitude.
        flipped = (left > 5000) != (right > 5000)

        if flipped:
            inversions.append((date, left, right))
        else:
            ax.plot(
                [x_left, x_right],
                [left, right],
                color=LIGHT,
                linewidth=0.9,
                solid_capstyle="round",
                zorder=2,
            )

    # Drawn last so they sit above the bundle.
    for date, left, right in inversions:
        ax.plot(
            [x_left, x_right],
            [left, right],
            color=VERMILLION,
            linewidth=1.8,
            solid_capstyle="round",
            zorder=4,
        )

    ax.axhline(CAP_BPS, color=GREY, linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    ax.text(
        x_right + 0.04,
        CAP_BPS,
        f"cap, {CAP_BPS:,} bps",
        fontsize=7.5,
        color=GREY,
        va="center",
    )

    ax.axhline(5000, color="#E8E8E8", linewidth=0.9, zorder=0)
    ax.text(
        x_right + 0.04,
        5000,
        "lending and staking\nequally weighted",
        fontsize=7.5,
        color="#9A9A9A",
        va="center",
    )

    ax.set_xlim(-0.22, 1.55)
    ax.set_ylim(0, 10_000)
    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(["constraint not stated", "constraint stated in prompt"])
    ax.set_yticks(range(0, 10_001, 2000))
    ax.set_yticklabels([f"{t:,}" for t in range(0, 10_001, 2000)])
    ax.set_ylabel("proposed weight in lending (bps)")
    ax.grid(visible=False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=8)

    figstyle.save(fig, f"fig16_disclosure_{SNAPSHOT_DATE}")

    print(f"\n{len(dates)} dates in both configurations")
    print(f"{len(inversions)} inversions:\n")
    for date, left, right in inversions:
        print(f"  {date}  lending {left:>5,} -> {right:>5,}")



if __name__ == "__main__":
    main()
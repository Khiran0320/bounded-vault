"""Interactive demonstration of the constraint layer.

Two things are shown here that a static figure cannot. The first tab
animates the gap between what an agent proposed and what the vault
actually held, so the reference monitor is visible as a divergence
rather than described as a mechanism. The second tab exposes the
language model's own reasoning on any chosen date in both
configurations, which is the evidence behind the inversion result and
is unreadable in tabular form.

Everything is read from the parquets written by scripts/value_layer.py.
Nothing is recomputed, so this cannot disagree with the dissertation.

Run from offchain/:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

SNAPSHOT_DATE = "2026-07-13"
RESULTS_ROOT = Path("data/results")
PER_STRATEGY_CAP_BPS = 6000

# Only the enforced arms appear here. The unenforced arms exist to
# measure the cost of the monitor and have nothing to show on a tab
# whose subject is the monitor acting.
AGENTS = {
    "Agent 1: rule-based": "rule_based",
    "Agent 2: mean-variance": "mean_variance",
    "Agent 2: self-constrained": "mean_variance_constrained",
    "Agent 3: unconstrained": "llm_unconstrained",
    "Agent 3: constrained": "llm_constrained",
}

LLM_RUNS = {
    "unconstrained": "llm_unconstrained",
    "constrained": "llm_constrained",
}


@st.cache_data
def load(name: str) -> pd.DataFrame:
    path = RESULTS_ROOT / name / f"{SNAPSHOT_DATE}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/value_layer.py first."
        )
    return pd.read_parquet(path)


def reasoning_of(rationale: str) -> str:
    """Pull the model's own words out of the packed rationale string.

    The agent stores provenance and reasoning in one field so a single
    parquet column carries both. Splitting here rather than at write
    time keeps the stored record exactly as the backtest saw it.
    """
    marker = "model_reasoning="
    if marker not in rationale:
        return rationale
    return rationale.split(marker, 1)[1].strip()


def status_of(rationale: str) -> str:
    for part in rationale.split(", "):
        if part.startswith("status="):
            return part.removeprefix("status=")
    return "unknown"


def proposal_plot(results: pd.DataFrame, label: str):
    """Proposed against executed lending weight, with rejections shaded.

    Lending is plotted rather than both legs because with two adapters
    summing to a constant the second line carries no extra information
    and halves the vertical resolution of the first.
    """
    figure, axis = plt.subplots(figsize=(10, 4.5))

    rejected = results.index[~results["accepted"]]
    for date in rejected:
        axis.axvspan(date, date, color="#d62728", alpha=0.10, linewidth=0)

    axis.plot(
        results.index,
        results["proposed_lending_bps"],
        label="proposed by agent",
        linewidth=1.8,
        color="#1f77b4",
    )
    axis.plot(
        results.index,
        results["weight_lending_bps"],
        label="held by vault after enforcement",
        linewidth=1.8,
        color="#2ca02c",
    )
    axis.axhline(
        PER_STRATEGY_CAP_BPS,
        color="0.3",
        linestyle="--",
        linewidth=1.0,
        label=f"per-strategy cap ({PER_STRATEGY_CAP_BPS} bps)",
    )

    axis.set_ylim(-300, 10_300)
    axis.set_ylabel("lending allocation (bps)")
    axis.set_title(label)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="center left")
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def tab_enforcement() -> None:
    st.subheader("What the agent asked for, and what the vault did")

    choice = st.selectbox("Agent", list(AGENTS))
    results = load(AGENTS[choice])

    rejected = int((~results["accepted"]).sum())
    days = len(results)
    deployed = int((results["turnover"] > 0).sum())

    left, middle, right = st.columns(3)
    left.metric("Decision dates", days)
    middle.metric("Rejected on chain", f"{rejected} ({rejected / days:.0%})")
    right.metric("Dates traded", deployed)

    st.pyplot(proposal_plot(results, choice))

    if deployed == 0:
        st.warning(
            "Every proposal was refused, so the vault never left its opening "
            "allocation and no capital was deployed. The flat line is the "
            "monitor holding, not the agent choosing."
        )
    elif rejected == 0:
        st.info(
            "No proposal breached the cap, so the two lines coincide. This "
            "agent was given the limit and respected it."
        )
    else:
        breached_leg = (
            "liquid staking"
            if (results["proposed_liquid_staking_bps"] > PER_STRATEGY_CAP_BPS).sum()
            > (results["proposed_lending_bps"] > PER_STRATEGY_CAP_BPS).sum()
            else "lending"
        )
        st.info(
            f"Shaded dates were refused. Every breach here was on the "
            f"{breached_leg} leg, so on those dates the cap moved the vault "
            f"away from that adapter rather than towards it."
        )

    reasons = results["rejection_reason"].value_counts()
    if not reasons.empty:
        st.caption(
            "Rejection reasons: "
            + ", ".join(f"{reason} ({count})" for reason, count in reasons.items())
        )


def tab_transcripts() -> None:
    st.subheader("What the language model said, on any date")

    constrained = load(LLM_RUNS["constrained"])
    unconstrained = load(LLM_RUNS["unconstrained"])

    # An inversion is a constrained date on which the agent preferred the
    # volatile adapter. These are the dates where the constraint text
    # changed the reasoning rather than merely truncating the output.
    inversions = constrained.index[
        constrained["proposed_liquid_staking_bps"]
        > constrained["proposed_lending_bps"]
    ]

    st.caption(
        f"{len(inversions)} of {len(constrained)} dates inverted under the "
        "constrained configuration, allocating the cap maximum to liquid "
        "staking. Those dates are marked below."
    )

    def label(date: pd.Timestamp) -> str:
        mark = "  [inversion]" if date in inversions else ""
        return f"{date:%Y-%m-%d}{mark}"

    dates = list(constrained.index)
    default = dates.index(inversions[0]) if len(inversions) else 0
    chosen = st.selectbox("Date", dates, index=default, format_func=label)

    for name, results in (
        ("Unconstrained", unconstrained),
        ("Constrained", constrained),
    ):
        row = results.loc[chosen]
        lending = int(row["proposed_lending_bps"])
        staking = int(row["proposed_liquid_staking_bps"])
        breached = max(lending, staking) > PER_STRATEGY_CAP_BPS

        st.markdown(f"### {name}")

        left, middle, right = st.columns(3)
        left.metric("Lending", f"{lending} bps")
        middle.metric("Liquid staking", f"{staking} bps")
        right.metric(
            "On-chain outcome",
            "rejected" if not row["accepted"] else "accepted",
            delta="breaches cap" if breached else None,
            delta_color="inverse" if breached else "off",
        )

        st.write(reasoning_of(str(row["rationale"])))
        st.caption(f"response served from: {status_of(str(row['rationale']))}")

    if chosen in inversions:
        st.success(
            "On this date the agent reversed its preference once told the "
            "cap existed. The constraint text changed the reasoning, not "
            "just the arithmetic, which is why zero breaches under the "
            "constrained configuration is not evidence of compliance."
        )


def main() -> None:
    st.set_page_config(page_title="Bounded Vault", layout="wide")
    st.title("Bounded Vault")
    st.caption(
        f"On-chain constraint enforcement over agent allocation proposals. "
        f"Snapshot {SNAPSHOT_DATE}, {136} decision dates, "
        f"{PER_STRATEGY_CAP_BPS} bps per-strategy cap."
    )

    enforcement, transcripts = st.tabs(
        ["Proposal against enforcement", "Agent 3 transcripts"]
    )
    with enforcement:
        tab_enforcement()
    with transcripts:
        tab_transcripts()


if __name__ == "__main__":
    main()
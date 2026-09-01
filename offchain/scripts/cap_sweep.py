"""Breach rate and executed performance as a function of the per-strategy cap.

Sweeps max_strategy_bps from the feasibility floor to the denominator and
records, per agent, what was proposed, what the monitor admitted, and what
the vault earned. The 6000 bps headline configuration is one column of the
resulting table rather than the whole result.

Three execution regimes are reported, because an enforced run alone cannot
separate the cost of the constraint from the cost of an operator who does
not respond to a rejection.

  hold     The reference monitor exactly. A refused proposal executes
           nothing and the vault keeps the weights it already held.
  project  The monitor still refuses, but the operator resubmits the
           closest admissible allocation.
  aware    The agent is told the cap and optimises subject to it, so
           nothing is refused. Only Agent 2 can do this without spending
           on live model calls, so it is the only agent swept this way.

Two markets are reported. The realised market is the snapshot as it
happened. The yield-only market replaces every price return with zero and
leaves the APY accrual and the proposals untouched, which isolates the
cost of reallocating away from the higher-yielding adapter from the cost
of being moved out of whichever asset happened to rise over this window.
The realised figure describes the sample; the yield-only figure describes
the adapters, and only the second survives the objection that the result
is a directional bet on one asset.

The vault opens at 5000/5000 rather than in cash. A vault opening at zero
and refused on its first proposal never deploys, so its return is zero at
every cap below its breach threshold and the curve would describe the
opening condition rather than the cap. 5000/5000 is the one allocation
admissible under every cap in the sweep, so the seed favours no agent and
no cap.

Proposals are replayed from data/results rather than regenerated. A
proposal is a property of the agent and does not depend on the cap the
monitor happens to hold, so replaying guarantees the sweep and the
headline run share one set of proposals and that Agent 3 is never called
live. The two self-constrained configurations are excluded for the
opposite reason: their proposals were produced under a disclosed cap of
6000 and replaying them against a different cap would describe neither.

Run from offchain/, after scripts/run_backtests.py:
    python scripts/cap_sweep.py

    This is AI Generated. The prompt used to generate this code is: Sweep the cap. Do this first. Vary max_strategy_bps from 10000 down to about 5000 and record, per agent, breach rate and executed performance
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bounded_vault.agents.base import Agent
from bounded_vault.agents.mean_variance import MeanVarianceAgent
from bounded_vault.backtest.engine import (
    BacktestConfig,
    run_backtest,
    summarise_run,
)
from bounded_vault.constraints import ConstraintConfig
from bounded_vault.feasible import project_to_cap
from bounded_vault.market import MarketView
from bounded_vault.market.snapshots import load_snapshot
from bounded_vault.schema import AdapterId, Proposal, StrategyAllocation

SNAPSHOT_DATE = "2026-07-13"
RESULTS_ROOT = Path("data/results")
FIGURES_ROOT = Path("data/figures")

# With two adapters and exact-sum, no allocation can hold both legs at or
# below a cap under 5000, so the sweep stops there rather than reporting a
# region where the constraint set is empty.
CAP_MIN_BPS = 5000
CAP_MAX_BPS = 10_000
CAP_STEP_BPS = 100
HEADLINE_CAP_BPS = 6000
REPORT_CAPS = [10_000, 8000, HEADLINE_CAP_BPS, 5000]

# Left inert at the denominator so the per-strategy cap is the only
# binding constraint and every rejection in the sweep is attributable to
# the parameter being swept.
MAX_REBALANCE_DELTA_BPS = 10_000

SEED_WEIGHTS = {AdapterId.LENDING: 5000, AdapterId.LIQUID_STAKING: 5000}

# Below this many accepted dates the executed path under the holding
# regime is set by a handful of proposals that happened to slip through,
# and its cumulative return is a draw rather than a measurement. Such
# points are computed and written out, but suppressed from the return
# panel of the figure so they cannot be read as performance.
MIN_ACCEPTED_DAYS = 10

# Agent 2 re-solved at every cap costs one convex program per date per
# cap. Set False to drop it if the sweep is being rerun for a figure tweak.
INCLUDE_CAP_AWARE = True

BACKTEST = BacktestConfig(warmup_days=0, initial_value=1.0, cost_bps=0)

REPLAY_RUNS = {
    "rule_based": "Agent 1: rule-based",
    "mean_variance": "Agent 2: mean-variance",
    "llm_unconstrained": "Agent 3: unconstrained",
}

PROPOSAL_COLUMNS = {
    f"proposed_{adapter.name.lower()}_bps": adapter for adapter in AdapterId
}


class ReplayAgent(Agent):
    """Re-emits the proposals a previous run recorded.

    Driving the sweep through the engine rather than reimplementing the
    executed-path recursion means the sweep and the headline results are
    produced by identical code, so they cannot disagree. A missing date
    raises rather than falling back, since a silent gap would surface only
    as an unexplained kink in the curve.
    """

    def __init__(self, proposals: pd.DataFrame, name: str) -> None:
        missing = [c for c in PROPOSAL_COLUMNS if c not in proposals.columns]
        if missing:
            raise ValueError(f"{name} is missing proposal columns: {missing}")
        self.name = name
        self._weights = {
            pd.Timestamp(date): {
                adapter: int(row[column])
                for column, adapter in PROPOSAL_COLUMNS.items()
            }
            for date, row in proposals.iterrows()
        }

    def propose(self, market: MarketView) -> Proposal:
        as_of = pd.Timestamp(market.as_of)
        if as_of not in self._weights:
            raise KeyError(
                f"{self.name} has no recorded proposal for {as_of:%Y-%m-%d}"
            )
        return Proposal(
            agent_name=self.name,
            as_of=market.as_of,
            allocations=[
                StrategyAllocation(adapter=adapter, weight_bps=weight)
                for adapter, weight in self._weights[as_of].items()
            ],
            rationale="replayed",
        )


class ProjectingAgent(Agent):
    """Submits the closest admissible allocation to what the agent wanted.

    The projection sits between the agent and the monitor, not inside
    either. The agent is unchanged and the monitor still rejects rather
    than rewrites; this only models an operator who retries.
    """

    def __init__(self, inner: Agent, cap: int, name: str) -> None:
        self.name = name
        self._inner = inner
        self._cap = cap

    def propose(self, market: MarketView) -> Proposal:
        proposal = self._inner.propose(market)
        proposed = {a.adapter: a.weight_bps for a in proposal.allocations}
        projected = project_to_cap(proposed, self._cap)
        return Proposal(
            agent_name=self.name,
            as_of=proposal.as_of,
            allocations=[
                StrategyAllocation(adapter=adapter, weight_bps=weight)
                for adapter, weight in projected.items()
            ],
            rationale=f"projected onto cap={self._cap}",
        )


class ConstantWeightAgent(Agent):
    """Proposes one fixed allocation on every date."""

    def __init__(self, weights_bps: dict[AdapterId, int], name: str) -> None:
        total = sum(weights_bps.values())
        if total != 10_000:
            raise ValueError(f"{name} weights sum to {total}, expected 10000")
        self.name = name
        self._weights = weights_bps

    def propose(self, market: MarketView) -> Proposal:
        return Proposal(
            agent_name=self.name,
            as_of=market.as_of,
            allocations=[
                StrategyAllocation(adapter=adapter, weight_bps=bps)
                for adapter, bps in self._weights.items()
            ],
            rationale="constant-weight benchmark",
        )


def load(name: str) -> pd.DataFrame:
    path = RESULTS_ROOT / name / f"{SNAPSHOT_DATE}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run scripts/run_backtests.py first"
        )
    return pd.read_parquet(path)


def constraints_at(cap: int) -> ConstraintConfig:
    return ConstraintConfig(
        allowed_adapters=frozenset({AdapterId.LENDING, AdapterId.LIQUID_STAKING}),
        max_strategy_bps=cap,
        total_cap_bps=10_000,
        max_rebalance_delta_bps=MAX_REBALANCE_DELTA_BPS,
        require_exact_sum=True,
    )


def breach_rate(proposals: pd.DataFrame, cap: int) -> float:
    """Share of dates whose proposal exceeded the cap, before enforcement.

    Read off the proposed columns, so the figure stays a property of the
    agent. Under the projecting regime the executed weights are admissible
    by construction, so the breach rate must come from the source frame
    rather than from the run.

    As a function of the cap this is one minus the empirical distribution
    function of the agent's largest proposed weight, which is what makes
    the breach curve a complete description of the agent's concentration
    rather than a single summary of it.
    """
    columns = list(PROPOSAL_COLUMNS)
    return float((proposals[columns].max(axis=1) > cap).mean())


def staleness(results: pd.DataFrame) -> float:
    """Mean days since the vault last executed a rebalance.

    Zero under the projecting regime, where every submission is admissible
    by construction. Under the holding regime this is the quantity the
    cumulative return cannot express: a vault refused on almost every date
    is not earning a return from an allocation it chose, it is carrying an
    allocation it has been unable to revise.
    """
    age = 0
    ages = []
    for accepted in results["accepted"]:
        age = 0 if accepted else age + 1
        ages.append(age)
    return sum(ages) / len(ages)


def evaluate(agent: Agent, snapshot, returns, yields, cap: int):
    results = run_backtest(
        agent,
        returns=returns,
        yields=yields,
        constraints=constraints_at(cap),
        config=BACKTEST,
        history=snapshot.returns,
        initial_weights=SEED_WEIGHTS,
    )
    return results, summarise_run(results, BACKTEST)


def record(market, regime, label, cap, results, metrics, breach) -> dict:
    return {
        "market": market,
        "regime": regime,
        "run": label,
        "cap_bps": cap,
        "breach_rate": round(breach, 4),
        "accepted": int(results["accepted"].sum()),
        "rejected": int((~results["accepted"]).sum()),
        "staleness_days": round(staleness(results), 2),
        "cumulative_return": round(metrics["cumulative_return"], 6),
        "annualised_return": round(metrics["annualised_return"], 6),
        "annualised_vol": round(metrics["annualised_volatility"], 6),
        # Named to make the artefact explicit. With a zero risk-free rate
        # and a portfolio held almost entirely in a yield-bearing stable
        # asset, this quantity diverges and ranks by stablecoin share
        # rather than by skill. It is retained for completeness and is not
        # reported in the thesis.
        "sharpe_rf0": round(metrics["sharpe_ratio"], 4),
        "max_drawdown": round(metrics["max_drawdown"], 6),
        "mean_turnover": round(metrics["mean_daily_turnover"], 6),
        "exec_lending_bps": round(results["weight_lending_bps"].mean()),
    }


def main() -> None:
    snapshot = load_snapshot(SNAPSHOT_DATE)
    yields = snapshot.yields
    returns = snapshot.returns.loc[yields.index]
    caps = list(range(CAP_MIN_BPS, CAP_MAX_BPS + 1, CAP_STEP_BPS))

    # Price returns zeroed, APY untouched. The lookback history handed to
    # the agents is left as the real series, so proposals are identical
    # across the two markets and only the accounting differs.
    flat = returns.copy()
    flat.loc[:, :] = 0.0
    markets = {"realised": returns, "yield_only": flat}

    print(f"Snapshot {SNAPSHOT_DATE}, {len(yields) - 1} decision dates")
    print(f"Caps {CAP_MIN_BPS} to {CAP_MAX_BPS} step {CAP_STEP_BPS} "
          f"({len(caps)} points)")
    print(f"Vault seeded at {SEED_WEIGHTS[AdapterId.LENDING]}/"
          f"{SEED_WEIGHTS[AdapterId.LIQUID_STAKING]} bps")

    benchmarks = {}
    for market, market_returns in markets.items():
        _, metrics = evaluate(
            ConstantWeightAgent(SEED_WEIGHTS, "benchmark_equal_weight"),
            snapshot, market_returns, yields, CAP_MAX_BPS,
        )
        benchmarks[market] = metrics["cumulative_return"]
        print(f"Equal-weight benchmark [{market}]: {benchmarks[market]:.5f}")
    print()

    proposals = {name: load(name) for name in REPLAY_RUNS}
    rows: list[dict] = []

    for market, market_returns in markets.items():
        for name, label in REPLAY_RUNS.items():
            print(f"sweeping {name} [{market}]", flush=True)
            source = proposals[name]

            for cap in caps:
                breach = breach_rate(source, cap)

                # The holding regime is the reference monitor and is
                # reported on the realised market only. Under a flat price
                # series it degenerates into a comparison of which dates
                # happened to be admissible, which the accepted counts
                # already report.
                if market == "realised":
                    held, metrics = evaluate(
                        ReplayAgent(source, f"{name}_hold"),
                        snapshot, market_returns, yields, cap,
                    )
                    rows.append(
                        record(market, "hold", label, cap, held, metrics, breach)
                    )

                projected, metrics = evaluate(
                    ProjectingAgent(
                        ReplayAgent(source, name), cap, f"{name}_project"
                    ),
                    snapshot, market_returns, yields, cap,
                )
                rows.append(
                    record(market, "project", label, cap, projected, metrics, breach)
                )

    if INCLUDE_CAP_AWARE:
        print("sweeping mean_variance re-solved at each cap", flush=True)
        for cap in caps:
            aware, metrics = evaluate(
                MeanVarianceAgent(
                    max_strategy_bps=cap, name=f"mean_variance_aware_{cap}"
                ),
                snapshot, returns, yields, cap,
            )
            rows.append(
                record("realised", "aware", "Agent 2: cap-aware", cap,
                       aware, metrics, 0.0)
            )

    table = pd.DataFrame(rows)
    table["excess_bps"] = (
        (table["cumulative_return"] - table["market"].map(benchmarks)) * 10_000
    ).round(1)

    check(table, benchmarks)
    report(table)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    path = RESULTS_ROOT / f"cap_sweep_{SNAPSHOT_DATE}.csv"
    table.to_csv(path, index=False)
    print(f"\nwritten to {path}")

    plot(table)


def check(table: pd.DataFrame, benchmarks: dict) -> None:
    """Properties the sweep must satisfy if it is wired correctly.

    At the denominator no proposal can breach, so holding and projecting
    must coincide. At the feasibility floor the constraint set is the
    single point 5000/5000, so every projected path must equal the
    equal-weight benchmark. A failure of either is a wiring fault, not a
    result. The third is not a wiring check but a finding, reported here
    because it is the cleanest place to see it: for an agent whose
    unconstrained optimum is a corner, projecting a refused proposal and
    re-optimising subject to the cap give the same portfolio.
    """
    real = table[table["market"] == "realised"]

    top = real[real["cap_bps"] == CAP_MAX_BPS]
    held = top[top["regime"] == "hold"].set_index("run")["cumulative_return"]
    proj = top[top["regime"] == "project"].set_index("run")["cumulative_return"]
    gap = (held - proj).abs().max()
    print(f"check: hold equals project at cap {CAP_MAX_BPS}, max gap {gap:.2e}")

    for market, benchmark in benchmarks.items():
        floor = table[
            (table["market"] == market)
            & (table["cap_bps"] == CAP_MIN_BPS)
            & (table["regime"] == "project")
        ]
        gap = (floor["cumulative_return"] - benchmark).abs().max()
        print(f"check: project equals equal weight at cap {CAP_MIN_BPS} "
              f"[{market}], max gap {gap:.2e}")

    aware = real[real["regime"] == "aware"].set_index("cap_bps")
    mv = real[(real["regime"] == "project")
              & (real["run"] == "Agent 2: mean-variance")].set_index("cap_bps")
    if len(aware) and len(mv):
        gap = (aware["cumulative_return"] - mv["cumulative_return"]).abs().max()
        print(f"check: Agent 2 project equals Agent 2 cap-aware, "
              f"max gap {gap:.2e}")


def report(table: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    real = table[table["market"] == "realised"]

    subset = real[real["cap_bps"].isin(REPORT_CAPS)]
    print("\nExecuted cumulative return by cap, realised market")
    print(
        subset.pivot_table(
            index=["regime", "run"], columns="cap_bps", values="cumulative_return"
        ).to_string()
    )

    print("\nExcess over equal weight, basis points of cumulative return")
    print(
        subset.pivot_table(
            index=["regime", "run"], columns="cap_bps", values="excess_bps"
        ).to_string()
    )

    # Positive means tightening the cap cost return on this sample.
    # Negative means the constraint acted as a regulariser and paid for
    # itself. The realised column is a property of the window; the
    # yield-only column is a property of the two adapters.
    loose = table[table["cap_bps"] == CAP_MAX_BPS]
    tight = table[table["cap_bps"] == HEADLINE_CAP_BPS]
    merged = loose.merge(
        tight, on=["market", "regime", "run"], suffixes=("_loose", "_tight")
    )
    merged["price_bps"] = (
        (merged["cumulative_return_loose"] - merged["cumulative_return_tight"])
        * 10_000
    ).round(1)
    print(f"\nPrice of safety, cap {CAP_MAX_BPS} to {HEADLINE_CAP_BPS}, "
          "in basis points of cumulative return")
    print(
        merged.pivot_table(
            index=["regime", "run"], columns="market", values="price_bps"
        ).to_string()
    )

    hold = real[(real["regime"] == "hold") & real["cap_bps"].isin(REPORT_CAPS)]
    print("\nHolding regime: dates the vault was able to act on, and mean "
          "days since the last executed rebalance")
    print(
        hold.pivot_table(index="run", columns="cap_bps",
                         values=["accepted", "staleness_days"]).to_string()
    )

    # The cap at which two agents' executed paths first differ. Below it
    # the cap binds on every date for both, the executed allocation is the
    # corner of the feasible set, and the agents are indistinguishable.
    proj = real[real["regime"] == "project"].pivot_table(
        index="cap_bps", columns="run", values="cumulative_return"
    )
    print("\nLowest cap at which two agents' executed paths differ")
    names = list(proj.columns)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            differs = proj.index[(proj[left] - proj[right]).abs() > 1e-9]
            first = int(differs.min()) if len(differs) else None
            print(f"  {left} vs {right}: "
                  f"{first if first is not None else 'never'}")


def plot(table: pd.DataFrame) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; figure skipped")
        return

    real = table[table["market"] == "realised"]
    flat = table[table["market"] == "yield_only"]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))

    # Common vertical scale for the two return panels on the realised
    # market, set from the projecting panel. Points on the holding panel
    # that fall outside it are the near-total-rejection region, which is
    # suppressed below in any case.
    reference = real[real["regime"].isin({"project", "aware"})]["excess_bps"]
    margin = 0.12 * (reference.max() - reference.min())
    limits = (reference.min() - margin, reference.max() + margin)

    ax = axes[0][0]
    for label, group in real[real["regime"] == "hold"].groupby("run"):
        group = group.sort_values("cap_bps")
        ax.plot(group["cap_bps"], group["excess_bps"],
                color="0.86", linewidth=1.0, zorder=1)
        usable = group[group["accepted"] >= MIN_ACCEPTED_DAYS]
        ax.plot(usable["cap_bps"], usable["excess_bps"],
                linewidth=1.6, label=label, zorder=2)
    ax.set_ylim(*limits)
    ax.set_title(f"Reject and hold (dates with under {MIN_ACCEPTED_DAYS} "
                 "accepted rebalances suppressed)", fontsize=9)
    ax.set_ylabel("excess over equal weight (bps)")
    ax.legend(fontsize=7, loc="best")

    ax = axes[0][1]
    for label, group in real[real["regime"] == "project"].groupby("run"):
        group = group.sort_values("cap_bps")
        ax.plot(group["cap_bps"], group["excess_bps"], linewidth=1.6, label=label)
    for label, group in real[real["regime"] == "aware"].groupby("run"):
        group = group.sort_values("cap_bps")
        ax.plot(group["cap_bps"], group["excess_bps"],
                linestyle="--", linewidth=1.4, label=label)
    ax.set_ylim(*limits)
    ax.set_title("Reject and project, realised market", fontsize=9)
    ax.legend(fontsize=7, loc="best")

    ax = axes[1][0]
    for label, group in real[real["regime"] == "hold"].groupby("run"):
        group = group.sort_values("cap_bps")
        ax.plot(group["cap_bps"], group["accepted"], linewidth=1.6, label=label)
    ax.axhline(MIN_ACCEPTED_DAYS, color="0.6", linestyle=":", linewidth=1.0)
    ax.set_title("Reject and hold: dates the vault was able to act on",
                 fontsize=9)
    ax.set_ylabel("accepted rebalances")

    ax = axes[1][1]
    for label, group in flat[flat["regime"] == "project"].groupby("run"):
        group = group.sort_values("cap_bps")
        ax.plot(group["cap_bps"], group["excess_bps"], linewidth=1.6, label=label)
    ax.set_title("Reject and project, yield-only market "
                 "(price returns set to zero)", fontsize=9)
    ax.set_ylabel("excess over equal weight (bps)")

    for row in axes:
        for ax in row:
            ax.axvline(HEADLINE_CAP_BPS, color="0.85", linewidth=1.0, zorder=0)
            ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
            ax.set_xlabel("per-strategy cap (bps), tighter to the left")
            ax.grid(alpha=0.3)

    fig.suptitle(
        f"Price of safety, {SNAPSHOT_DATE} snapshot, no rebalance cost",
        fontsize=11,
    )

    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    path = FIGURES_ROOT / f"cap_sweep_{SNAPSHOT_DATE}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"written to {path}")


if __name__ == "__main__":
    main()
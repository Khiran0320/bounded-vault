"""Benchmarks and turnover cost sensitivity for the executed path.

Adds three passive benchmarks to the result set and reports what each
agent delivered against them, then sweeps the rebalance cost to check
whether any ranking depends on the frictionless assumption.

The benchmarks run unenforced. Lending-only proposes the full allocation
to one adapter and would be refused under the cap, leaving it stuck at its
opening weights and returning zero, which measures the monitor rather than
the benchmark. Equal weight is admissible either way and is unaffected.

Run from offchain/:
    python scripts/value_layer.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bounded_vault.agents.base import Agent
from bounded_vault.agents.llm import LLMAgent
from bounded_vault.agents.mean_variance import MeanVarianceAgent
from bounded_vault.agents.rule_based import RuleBasedAgent
from bounded_vault.backtest.engine import (
    BacktestConfig,
    run_backtest,
    save_results,
    summarise_run,
)
from bounded_vault.constraints import ConstraintConfig
from bounded_vault.market.snapshots import load_snapshot
from bounded_vault.schema import AdapterId, Proposal, StrategyAllocation

SNAPSHOT_DATE = "2026-07-13"
RESULTS_ROOT = Path("data/results")
FIGURES_ROOT = Path("data/figures")

PER_STRATEGY_CAP_BPS = 6000
MAX_REBALANCE_DELTA_BPS = 10_000

# Basis points charged on turnover. Zero is the headline configuration;
# the others test whether the ranking survives friction. A daily-rebalanced
# vault paying 20 bps a side on Solana would be at the pessimistic end.
COST_SWEEP_BPS = [0, 5, 20]

CONSTRAINTS = ConstraintConfig(
    allowed_adapters=frozenset({AdapterId.LENDING, AdapterId.LIQUID_STAKING}),
    max_strategy_bps=PER_STRATEGY_CAP_BPS,
    total_cap_bps=10_000,
    max_rebalance_delta_bps=MAX_REBALANCE_DELTA_BPS,
    require_exact_sum=True,
)

UNENFORCED = ConstraintConfig(
    allowed_adapters=frozenset({AdapterId.LENDING, AdapterId.LIQUID_STAKING}),
    max_strategy_bps=10_000,
    total_cap_bps=10_000,
    max_rebalance_delta_bps=MAX_REBALANCE_DELTA_BPS,
    require_exact_sum=True,
)

CONSTRAINT_TEXT = f"""The vault enforces a hard limit of \
{PER_STRATEGY_CAP_BPS} basis points on any single adapter. Allocations that \
exceed it are rejected on-chain and will not execute. Weights must still sum \
to exactly 10000."""


class ConstantWeightAgent(Agent):
    """Proposes one fixed allocation on every date.

    Passive benchmarks are agents that ignore the market, so they need no
    separate execution path: running them through the same engine means
    their returns, turnover and costs are computed by identical code.
    """

    def __init__(self, weights_bps: dict[AdapterId, int], name: str) -> None:
        total = sum(weights_bps.values())
        if total != 10_000:
            raise ValueError(f"{name} weights sum to {total}, expected 10000")
        self.name = name
        self._weights = weights_bps

    def propose(self, market) -> Proposal:
        return Proposal(
            agent_name=self.name,
            as_of=market.as_of,
            allocations=[
                StrategyAllocation(adapter=adapter, weight_bps=bps)
                for adapter, bps in self._weights.items()
            ],
            rationale="constant-weight benchmark",
        )


def build_runs():
    """Every configuration to evaluate, in reporting order."""
    return [
        (
            ConstantWeightAgent(
                {AdapterId.LENDING: 5_000, AdapterId.LIQUID_STAKING: 5_000},
                "benchmark_equal_weight",
            ),
            UNENFORCED,
        ),
        (
            ConstantWeightAgent(
                {AdapterId.LENDING: 10_000, AdapterId.LIQUID_STAKING: 0},
                "benchmark_lending_only",
            ),
            UNENFORCED,
        ),
        (
            ConstantWeightAgent(
                {AdapterId.LENDING: 0, AdapterId.LIQUID_STAKING: 10_000},
                "benchmark_liquid_staking_only",
            ),
            UNENFORCED,
        ),
        (RuleBasedAgent(), CONSTRAINTS),
        (RuleBasedAgent(name="rule_based_unenforced"), UNENFORCED),
        (MeanVarianceAgent(), CONSTRAINTS),
        (MeanVarianceAgent(name="mean_variance_unenforced"), UNENFORCED),
        (
            MeanVarianceAgent(
                max_strategy_bps=PER_STRATEGY_CAP_BPS,
                name="mean_variance_constrained",
            ),
            CONSTRAINTS,
        ),
        (LLMAgent(), CONSTRAINTS),
        (LLMAgent(name="llm_unconstrained_unenforced"), UNENFORCED),
        (LLMAgent(constraint_context=CONSTRAINT_TEXT), CONSTRAINTS),
    ]


def main() -> None:
    snapshot = load_snapshot(SNAPSHOT_DATE)
    yields = snapshot.yields
    returns = snapshot.returns.loc[yields.index]

    print(f"Snapshot {SNAPSHOT_DATE}, {len(yields) - 1} decision dates")

    rows = []
    curves: dict[str, pd.Series] = {}

    for agent, constraints in build_runs():
        record = {"run": agent.name}

        for cost_bps in COST_SWEEP_BPS:
            config = BacktestConfig(
                warmup_days=0, initial_value=1.0, cost_bps=cost_bps
            )
            results = run_backtest(
                agent,
                returns=returns,
                yields=yields,
                constraints=constraints,
                config=config,
                history=snapshot.returns,
            )
            metrics = summarise_run(results, config)

            if cost_bps == 0:
                # Only the frictionless run is canonical. The cost sweep is
                # a sensitivity check and is not written to disk.
                curves[agent.name] = results["vault_value_close"]
                deployed = int((results["turnover"] > 0).sum())
                record["deployed"] = deployed > 0
                record["ann_vol"] = round(metrics["annualised_volatility"], 4)
                record["sortino"] = round(metrics["sortino_ratio"], 3)
                record["max_dd"] = round(metrics["max_drawdown"], 4)
                record["turnover"] = round(metrics["mean_daily_turnover"], 5)
                record["rejected"] = metrics["violation_count"]

                if agent.name.startswith("benchmark_"):
                    save_results(results, SNAPSHOT_DATE)

            record[f"ret_{cost_bps}bps"] = round(metrics["cumulative_return"], 5)

        rows.append(record)

    table = pd.DataFrame(rows).set_index("run")
    pd.set_option("display.width", 220)
    print()
    print(table.to_string())
    print("\ndeployed=False means every proposal was refused and the vault")
    print("held its opening zero allocation, so its return reflects the")
    print("monitor rather than the agent.")

    plot(curves)


def plot(curves: dict[str, pd.Series]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; figure skipped")
        return

    # Benchmarks dashed and grey-ish so the agent curves read as the
    # subject and the benchmarks as the backdrop.
    styles = {
        "benchmark_equal_weight": ("--", "0.45"),
        "benchmark_lending_only": ("--", "0.65"),
        "benchmark_liquid_staking_only": ("--", "0.25"),
    }

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    for name, series in curves.items():
        style, colour = styles.get(name, ("-", None))
        ax.plot(
            series.index,
            series.values,
            label=name,
            linestyle=style,
            color=colour,
            linewidth=1.6,
        )

    ax.axhline(1.0, color="0.8", linewidth=0.8, zorder=0)
    ax.set_xlabel("date")
    ax.set_ylabel("vault value (opening value = 1.0)")
    ax.set_title(
        "Executed-path vault value, 136 decision dates, no rebalance cost"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.autofmt_xdate()

    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    path = FIGURES_ROOT / f"equity_curves_{SNAPSHOT_DATE}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"written to {path}")


if __name__ == "__main__":
    main()
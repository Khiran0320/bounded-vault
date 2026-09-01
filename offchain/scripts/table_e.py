"""Table E and Table F source figures for Section 4.6.3.

Reads only the saved result parquets and reduces each through the same
summarise_run the harness uses, so every figure printed here has the same
provenance as the figures already in Chapter 4. No agent is re-run, no
network call is made, and nothing under src/ is touched.

The turnover cost columns are reconstructed rather than re-simulated.
Cost enters the engine as an additive drag on the daily return and never
touches the weight path, so the value at c basis points is the product
over dates of (1 + portfolio_return - turnover * c / 10000). The 0 bps
column is printed twice by two different routes as a check on that claim.

Run from offchain/:
    python scripts/table_e.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bounded_vault.backtest.engine import BacktestConfig, summarise_run

SNAPSHOT_DATE = "2026-07-13"
RESULTS_ROOT = Path("data/results")
COST_SWEEP_BPS = [0, 5, 20]

LABELS = {
    "rule_based": "C1 Agent 1, enforced",
    "rule_based_unenforced": "C2 Agent 1, unenforced",
    "mean_variance": "C3 Agent 2, enforced",
    "mean_variance_unenforced": "C4 Agent 2, unenforced",
    "mean_variance_constrained": "C5 Agent 2, self constrained",
    "llm_unconstrained": "C6 Agent 3, enforced, cap not disclosed",
    "llm_unconstrained_unenforced": "C7 Agent 3, unenforced",
    "llm_constrained": "C8 Agent 3, cap disclosed",
    "benchmark_equal_weight": "B1 Equal weight",
    "benchmark_lending_only": "B2 Lending only",
    "benchmark_liquid_staking_only": "B3 Liquid staking only",
}


def cost_adjusted_return(results: pd.DataFrame, cost_bps: int) -> float:
    drag = results["turnover"] * (cost_bps / 10_000.0)
    net = results["portfolio_return"] - drag
    return float((1.0 + net).prod() - 1.0)


def main() -> None:
    config = BacktestConfig(warmup_days=0, initial_value=1.0, cost_bps=0)
    rows = []

    for name, label in LABELS.items():
        path = RESULTS_ROOT / name / f"{SNAPSHOT_DATE}.parquet"
        if not path.exists():
            print(f"missing: {path}")
            continue

        results = pd.read_parquet(path)
        m = summarise_run(results, config)

        row = {
            "config": label,
            "days": m["days"],
            "deployed": bool((results["turnover"] > 0).any()),
            "cum_ret_pct": round(100 * m["cumulative_return"], 2),
            "ann_ret_pct": round(100 * m["annualised_return"], 2),
            "ann_vol_pct": round(100 * m["annualised_volatility"], 2),
            "max_dd_pct": round(100 * m["max_drawdown"], 2),
            "sharpe": round(m["sharpe_ratio"], 3),
            "turnover": round(m["mean_daily_turnover"], 5),
            "rejections": m["violation_count"],
        }

        for cost_bps in COST_SWEEP_BPS:
            row[f"ret_{cost_bps}bps_pct"] = round(
                100 * cost_adjusted_return(results, cost_bps), 2
            )

        rows.append(row)

    table = pd.DataFrame(rows).set_index("config")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print(table.to_string())

    mismatch = (table["cum_ret_pct"] - table["ret_0bps_pct"]).abs().max()
    print(f"\ncum_ret_pct vs ret_0bps_pct, max absolute difference: {mismatch}")
    print("Non-zero would mean the cost reconstruction is not an identity.")

    print("\nRows with deployed=False belong in Table F, not Table E.")
    print("Their zero return is the monitor refusing every proposal, not")
    print("a performance figure, and must not enter any ranking.")


if __name__ == "__main__":
    main()
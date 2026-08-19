"""Regime-control analysis over the completed backtest results.

The research proposal committed to testing whether agents differ
systematically between calm and volatile conditions. A single 137 day
window cannot supply two independent market regimes, so this partitions
the one window two ways instead, each answering a different question,
and neither requiring the agents to be rerun.

The directional split cuts the window in two at the peak preceding the
deepest drawdown of the most volatile benchmark, giving a rising
sub-period and a falling one. Because the parts are contiguous, every
path-dependent metric remains well defined.

The volatility split labels each date by the trailing realised
volatility of that same benchmark, above or below its median. The
labels are non-contiguous, so only path-independent statistics are
reported for them: a cumulative return over scattered dates would
describe no strategy anyone could have held.

Both partitions are defined from a passive benchmark rather than from
any agent's own returns, so no agent is measured against a partition
its own behaviour helped draw.

Run from offchain/:
    python scripts/regime_split.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bounded_vault.backtest.metrics import summarise

SNAPSHOT_DATE = "2026-07-13"
RESULTS_ROOT = Path("data/results")

# Liquid-staking-only carries the undiluted market signal, so its path
# defines the regimes for every run. Using a blend would smear the break.
REGIME_BENCHMARK = "benchmark_liquid_staking_only"

# Trailing window for the volatility classifier. Seven days separates the
# two classes most cleanly on this sample and costs the fewest dates to
# warmup; the window is backward-looking, so the labelling uses no
# information unavailable on the date it labels.
VOLATILITY_WINDOW = 7

DAYS_PER_YEAR = 365

ORDER = [
    "benchmark_lending_only",
    "benchmark_equal_weight",
    "benchmark_liquid_staking_only",
    "rule_based",
    "rule_based_unenforced",
    "mean_variance",
    "mean_variance_unenforced",
    "mean_variance_constrained",
    "llm_unconstrained",
    "llm_unconstrained_unenforced",
    "llm_constrained",
]


def load(name: str) -> pd.DataFrame | None:
    path = RESULTS_ROOT / name / f"{SNAPSHOT_DATE}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def find_break(benchmark: pd.DataFrame) -> pd.Timestamp:
    """Date of the running peak preceding the deepest drawdown.

    Defined from the realised path rather than chosen by inspection, so
    the partition is reproducible and cannot be accused of having been
    positioned to flatter a result.
    """
    values = benchmark["vault_value_close"]
    drawdown = 1.0 - values / values.cummax()
    trough = drawdown.idxmax()
    return values.loc[:trough].idxmax()


def classify_volatility(benchmark: pd.DataFrame, window: int) -> pd.Series:
    """Label each date calm or volatile by trailing realised volatility.

    The median is taken over the labelled dates themselves, which splits
    the sample evenly by construction. That is a deliberate choice: an
    absolute threshold would be arbitrary on a single window, whereas a
    median split makes the comparison one of relative conditions within
    the sample, which is what the available data can support.
    """
    returns = benchmark["portfolio_return"]
    realised = returns.rolling(window).std()
    median = realised.median()

    labels = pd.Series(pd.NA, index=returns.index, dtype="object")
    labels[realised.notna() & (realised > median)] = "volatile"
    labels[realised.notna() & (realised <= median)] = "calm"
    return labels


def period_metrics(results: pd.DataFrame, mask) -> dict | None:
    """Full metrics over a contiguous slice, rebased to its own start."""
    part = results[mask]
    if len(part) < 2:
        return None
    values = [part["vault_value_open"].iloc[0], *part["vault_value_close"]]
    return summarise(
        values=values,
        daily_returns=part["portfolio_return"].tolist(),
        turnovers=part["turnover"].tolist(),
        violations=int((~part["accepted"]).sum()),
        days_per_year=DAYS_PER_YEAR,
    )


def annualised_stats(returns: pd.Series) -> tuple[float, float]:
    """Mean return and volatility, annualised, for a set of scattered dates.

    Both are path-independent, so they remain meaningful over dates that
    do not form a continuous holding period.
    """
    mean_daily = float(returns.mean())
    annual_return = (1.0 + mean_daily) ** DAYS_PER_YEAR - 1.0
    annual_vol = float(returns.std()) * (DAYS_PER_YEAR**0.5)
    return annual_return, annual_vol


def directional_table(runs: dict[str, pd.DataFrame], brk: pd.Timestamp):
    rows = []
    for name, results in runs.items():
        record = {"run": name}
        for label, mask in (
            ("rising", results.index <= brk),
            ("falling", results.index > brk),
        ):
            metrics = period_metrics(results, mask)
            if metrics is None:
                continue
            record[f"{label}_days"] = metrics["days"]
            record[f"{label}_ret"] = round(metrics["cumulative_return"], 4)
            record[f"{label}_vol"] = round(metrics["annualised_volatility"], 4)
            record[f"{label}_dd"] = round(metrics["max_drawdown"], 4)
        rows.append(record)
    return pd.DataFrame(rows).set_index("run")


def volatility_table(runs: dict[str, pd.DataFrame], labels: pd.Series):
    rows = []
    for name, results in runs.items():
        record = {"run": name}
        aligned = labels.reindex(results.index)
        for regime in ("calm", "volatile"):
            selected = results.loc[aligned == regime, "portfolio_return"]
            annual_return, annual_vol = annualised_stats(selected)
            record[f"{regime}_n"] = len(selected)
            record[f"{regime}_ret"] = round(annual_return, 4)
            record[f"{regime}_vol"] = round(annual_vol, 4)
        rows.append(record)
    return pd.DataFrame(rows).set_index("run")


def main() -> None:
    benchmark = load(REGIME_BENCHMARK)
    if benchmark is None:
        raise FileNotFoundError(
            f"{REGIME_BENCHMARK} results not found; run scripts/value_layer.py first"
        )

    runs = {name: load(name) for name in ORDER}
    runs = {name: df for name, df in runs.items() if df is not None}

    brk = find_break(benchmark)
    labels = classify_volatility(benchmark, VOLATILITY_WINDOW)
    counts = labels.value_counts().to_dict()

    print(f"Snapshot {SNAPSHOT_DATE}, {len(runs)} runs")
    print(f"directional break at {brk:%Y-%m-%d}")
    print(
        f"volatility labels: {counts.get('calm', 0)} calm, "
        f"{counts.get('volatile', 0)} volatile, "
        f"{int(labels.isna().sum())} unlabelled (classifier warmup)"
    )

    pd.set_option("display.width", 220)

    directional = directional_table(runs, brk)
    print("\nDirectional split (contiguous, path-dependent metrics valid)")
    print(directional.to_string())

    volatility = volatility_table(runs, labels)
    print("\nVolatility split (scattered dates, annualised statistics only)")
    print(volatility.to_string())

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    directional.to_csv(RESULTS_ROOT / f"regime_directional_{SNAPSHOT_DATE}.csv")
    volatility.to_csv(RESULTS_ROOT / f"regime_volatility_{SNAPSHOT_DATE}.csv")
    print(f"\nwritten to {RESULTS_ROOT}/regime_*_{SNAPSHOT_DATE}.csv")


if __name__ == "__main__":
    main()
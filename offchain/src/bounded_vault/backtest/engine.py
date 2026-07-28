"""Backtest engine.

Strict no-lookahead. On day t the agent receives a MarketView built only
from data up to and including t. The weights it sets then earn the return
realised between t and t+1, which the agent never saw.

Convention: returns.loc[d, adapter] is the price return earned from d-1 to
d. If your returns frame is built the other way round, the engine leaks
future data with no visible symptom, so check this before trusting output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bounded_vault.backtest.accounting import (
    adapter_daily_return,
    portfolio_daily_return,
    rebalance_cost,
    turnover,
)
from bounded_vault.backtest.metrics import summarise
from bounded_vault.constraints import ConstraintConfig, validate_proposal
from bounded_vault.market.view import MarketView
from bounded_vault.schema import AdapterId


@dataclass(frozen=True)
class BacktestConfig:
    """Run parameters.

    warmup_days is applied to every agent, not just those that need it, so
    that all agents are evaluated over an identical window.
    """

    warmup_days: int = 60
    initial_value: float = 1.0
    cost_bps: int = 0
    days_per_year: int = 365


def _to_adapter(label) -> AdapterId:
    """Recover an AdapterId from a DataFrame column label.

    pandas stores IntEnum keys as plain integers, so column labels come back
    as ints rather than enum members. Normalising here keeps the rest of the
    engine working with the enum, and fails loudly on an unknown column
    rather than silently dropping it.
    """
    if isinstance(label, AdapterId):
        return label
    try:
        return AdapterId(int(label))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"column {label!r} is not a known AdapterId") from exc


def _proposal_to_weights(proposal) -> dict[AdapterId, int]:
    """Extract a weight vector from a Proposal.

    This is the only place the engine depends on the Proposal shape. If the
    schema changes, change this function and nothing else.
    """
    return {
        allocation.adapter: allocation.weight_bps
        for allocation in proposal.allocations
    }


def _build_view(
    returns_to_date: pd.DataFrame,
    yields_row: pd.Series,
    as_of,
) -> MarketView:
    """Construct the MarketView handed to the agent on a single day.

    This is the only place the engine depends on the MarketView shape. The
    returns frame passed in is already truncated at as_of, so the agent is
    structurally unable to see the future rather than merely trusted not to.
    Yield keys are restored to AdapterId so agents can index the dict with
    the enum, matching the declared type on MarketView.
    """
    return MarketView(
        as_of=as_of,
        returns=returns_to_date,
        yields={_to_adapter(key): float(value) for key, value in yields_row.items()},
    )


def run_backtest(
    agent,
    returns: pd.DataFrame,
    yields: pd.DataFrame,
    constraints: ConstraintConfig,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Run one agent over one dataset and return the per day record.

    returns and yields are both indexed by date with AdapterId columns.
    yields holds APY as a fraction, not a percent.

    A rejected proposal leaves the previous weights in place, mirroring the
    on-chain behaviour where a failing transaction reverts and the vault is
    unchanged. Rejections are recorded rather than raised: the violation
    count is a result of the experiment, not an error condition.
    """
    config = config or BacktestConfig()

    if not returns.index.equals(yields.index):
        raise ValueError("returns and yields must share an identical date index")

    if set(returns.columns) != set(yields.columns):
        raise ValueError("returns and yields must cover the same adapters")

    # Maps each adapter to the label pandas actually stored, so lookups do
    # not depend on whether the enum survived frame construction.
    columns = {_to_adapter(col): col for col in returns.columns}
    if len(columns) != len(returns.columns):
        raise ValueError("returns frame has duplicate adapter columns")

    adapters = list(columns)
    dates = returns.index
    last_decision = len(dates) - 1

    if config.warmup_days >= last_decision:
        raise ValueError(
            f"warmup of {config.warmup_days} days leaves no evaluation period "
            f"in a {len(dates)} day series"
        )

    current_weights: dict[AdapterId, int] | None = None
    value = config.initial_value
    rows = []

    for i in range(config.warmup_days, last_decision):
        today = dates[i]
        tomorrow = dates[i + 1]

        view = _build_view(returns.loc[:today], yields.loc[today], today)
        proposal = agent.propose(view)

        if pd.Timestamp(proposal.as_of) != pd.Timestamp(today):
            raise ValueError(
                f"agent stamped proposal {proposal.as_of} while simulating {today}; "
                "agents must take as_of from view.as_of, never from wall clock time"
            )

        proposed = _proposal_to_weights(proposal)

        accepted, reason = validate_proposal(proposed, current_weights, constraints)
        previous = current_weights or {adapter: 0 for adapter in adapters}
        new_weights = proposed if accepted else previous

        traded = turnover(previous, new_weights)
        cost = rebalance_cost(previous, new_weights, config.cost_bps)

        # APY known at decision time, price return realised overnight.
        adapter_returns = {
            adapter: adapter_daily_return(
                price_return=float(returns.at[tomorrow, col]),
                apy=float(yields.at[today, col]),
            )
            for adapter, col in columns.items()
        }

        gross = portfolio_daily_return(new_weights, adapter_returns)
        net = gross - cost
        closing = value * (1.0 + net)

        rows.append(
            {
                "date": today,
                "agent_name": proposal.agent_name,
                "vault_value_open": value,
                "vault_value_close": closing,
                "portfolio_return": net,
                "accepted": accepted,
                "rejection_reason": reason.value if reason else None,
                "turnover": traded,
                "cost": cost,
                **{
                    f"weight_{adapter.name.lower()}_bps": new_weights.get(adapter, 0)
                    for adapter in adapters
                },
            }
        )

        current_weights = new_weights
        value = closing

    return pd.DataFrame(rows).set_index("date")


def summarise_run(
    results: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> dict:
    """Reduce a per day record to a single row of headline metrics."""
    config = config or BacktestConfig()
    values = [results["vault_value_open"].iloc[0], *results["vault_value_close"]]
    return summarise(
        values=values,
        daily_returns=results["portfolio_return"].tolist(),
        turnovers=results["turnover"].tolist(),
        violations=int((~results["accepted"]).sum()),
        days_per_year=config.days_per_year,
    )


def save_results(
    results: pd.DataFrame,
    snapshot_date: str,
    root: Path | str = "data/results",
    agent_name: str | None = None,
) -> Path:
    """Write a run to data/results/{agent_name}/{snapshot_date}.parquet.

    The agent name is taken from the run itself unless overridden, so the
    output path cannot drift from the agent that actually produced it.
    Spaces become underscores; no other transformation is applied.
    """
    if agent_name is None:
        if "agent_name" not in results.columns:
            raise ValueError(
                "results carry no agent_name column; pass agent_name explicitly"
            )

        names = results["agent_name"].unique()
        if len(names) != 1:
            raise ValueError(f"results mix multiple agents: {list(names)}")
        agent_name = str(names[0])

    directory = Path(root) / agent_name.strip().replace(" ", "_")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{snapshot_date}.parquet"
    results.to_parquet(path)
    return path
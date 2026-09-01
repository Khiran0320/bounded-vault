"""Contribution of the five inversion dates to the disclosure shortfall.

Section 4.5 claims the inversion dates account for a stated number of
percentage points of the gap between Agent 3's undisclosed and disclosed
configurations. This computes that number.

Inversion dates are identified from the proposal columns, which are a
property of the agent and invariant to the constraint configuration a run
executed under. The return difference is taken from the executed paths of
C7 and C8.

Two decompositions are printed. The additive one sums the daily return
differences on the inversion dates. The counterfactual one recomputes the
disclosed path with the undisclosed daily return substituted on those
dates only, which is path consistent and is the figure to quote. They
differ because returns compound, which is why the sentence says roughly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SNAPSHOT_DATE = "2026-07-13"
RESULTS_ROOT = Path("data/results")
MIDPOINT_BPS = 5000


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS_ROOT / name / f"{SNAPSHOT_DATE}.parquet")


def cumulative(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def main() -> None:
    undisclosed = load("llm_unconstrained_unenforced")
    disclosed = load("llm_constrained")

    # An inversion is a change in which adapter the model preferred, not a
    # reduction in how much it wanted. Both configurations saw identical
    # market data, so a flip across the midpoint cannot be a projection
    # onto the feasible set.
    mask = (
        (undisclosed["proposed_lending_bps"] > MIDPOINT_BPS)
        & (disclosed["proposed_lending_bps"] < MIDPOINT_BPS)
    )

    print(f"inversion dates: {int(mask.sum())}")
    print()
    print(
        pd.DataFrame(
            {
                "undisc_lend": undisclosed.loc[mask, "proposed_lending_bps"],
                "undisc_lst": undisclosed.loc[mask, "proposed_liquid_staking_bps"],
                "disc_lend": disclosed.loc[mask, "proposed_lending_bps"],
                "disc_lst": disclosed.loc[mask, "proposed_liquid_staking_bps"],
            }
        ).to_string()
    )

    u_ret = undisclosed["portfolio_return"]
    d_ret = disclosed["portfolio_return"]

    total_gap = 100 * (cumulative(d_ret) - cumulative(u_ret))
    daily_diff = d_ret - u_ret

    additive_inversion = 100 * float(daily_diff[mask].sum())
    additive_other = 100 * float(daily_diff[~mask].sum())

    # Path consistent attribution: hold the disclosed path fixed except on
    # the inversion dates, where the undisclosed return is substituted.
    counterfactual = d_ret.copy()
    counterfactual[mask] = u_ret[mask]
    counterfactual_gap = 100 * (cumulative(d_ret) - cumulative(counterfactual))

    print()
    print(f"C7 undisclosed cumulative return: {100 * cumulative(u_ret):.2f} pct")
    print(f"C8 disclosed cumulative return:   {100 * cumulative(d_ret):.2f} pct")
    print(f"total gap:                        {total_gap:.2f} pp")
    print()
    print(f"additive, inversion dates:        {additive_inversion:.2f} pp")
    print(f"additive, remaining 131 dates:    {additive_other:.2f} pp")
    print(f"counterfactual attribution:       {counterfactual_gap:.2f} pp")
    print()
    print("Quote the counterfactual figure. The additive split is printed")
    print("as a cross-check and will differ slightly because returns")
    print("compound, which is why the sentence in 4.5 says roughly.")


if __name__ == "__main__":
    main()
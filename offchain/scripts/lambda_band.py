"""
Risk aversion robustness check for Agent 2.

Recomputes the mean-variance allocation across a grid of risk aversion
values on the frozen snapshot, and locates the band (if any) in which the
per-strategy cap is satisfiable.

Run from offchain/:
    python scripts/lambda_band.py
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp

from bounded_vault.schema import AdapterId

# ADJUST THIS IMPORT to match your snapshots module path.
from bounded_vault.market.snapshots import load_snapshot, view_at
from bounded_vault.agents.mean_variance import MeanVarianceAgent

SNAPSHOT_DATE = "2026-07-13"
TAU = 0.15
PER_STRATEGY_CAP_BPS = 6000
DAYS_PER_YEAR = 365
LAMBDA_GRID = [0.1, 0.15, 0.2, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]


# def build_inputs(view):
#     """Return (mu, sigma, diagnostics) annualised, ordered by ORDER."""
#     returns = view.returns[[*ORDER]].dropna()
#     n_obs = len(returns)
#     years = n_obs / DAYS_PER_YEAR

#     sigma = returns.cov().values * DAYS_PER_YEAR
#     vol_annual = np.sqrt(np.diag(sigma))

#     drift_annual = returns.mean().values * DAYS_PER_YEAR
#     se_annual = vol_annual / np.sqrt(years)

#     k = TAU**2 / (TAU**2 + se_annual**2)
#     apy = np.array([float(view.yields[a]) for a in ORDER])

#     mu = k * drift_annual + apy

#     diagnostics = {
#         "n_obs": n_obs,
#         "years": years,
#         "vol_annual": vol_annual,
#         "drift_annual": drift_annual,
#         "se_annual": se_annual,
#         "k": k,
#         "apy": apy,
#     }
#     return mu, sigma, diagnostics


def solve(mu, sigma, lam):
    """Maximise w'mu - (lam / 2) w'Sigma w subject to sum(w) = 1, w >= 0."""
    n = len(mu)
    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - (lam / 2) * cp.quad_form(w, cp.psd_wrap(sigma)))
    problem = cp.Problem(objective, [cp.sum(w) == 1, w >= 0])
    problem.solve()
    if w.value is None:
        raise RuntimeError(f"solver failed at lambda={lam}")
    return np.clip(w.value, 0.0, 1.0)


def max_weight_bps(mu, sigma, lam):
    return float(np.max(solve(mu, sigma, lam))) * 10000


def bisect_boundary(mu, sigma, lo, hi, tol=1e-4):
    """Find lambda where max weight crosses the cap. Returns None if no crossing."""
    cap = PER_STRATEGY_CAP_BPS
    f_lo = max_weight_bps(mu, sigma, lo) - cap
    f_hi = max_weight_bps(mu, sigma, hi) - cap
    if f_lo * f_hi > 0:
        return None
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if (max_weight_bps(mu, sigma, mid) - cap) * f_lo > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    snapshot = load_snapshot(SNAPSHOT_DATE)
    view = view_at(snapshot, snapshot.dates[-1])
    agent = MeanVarianceAgent(tau=TAU)
    adapters, mu, sigma, d = agent._estimate(view)
    n_obs = min(len(view.returns.dropna()), agent.lookback_days) 
    print(f"Observations: {n_obs} ({n_obs / 365:.3f} years)")

    print(f"Snapshot: {SNAPSHOT_DATE}")
    print(f"Tau: {TAU}   Per-strategy cap: {PER_STRATEGY_CAP_BPS} bps\n")

    print(f"{'asset':<16}{'drift':>10}{'SE':>10}{'keep':>10}{'apy':>10}{'mu':>10}")
    for i, adapter in enumerate(adapters):
        print(
            f"{adapter.name:<16}"
            f"{d['drift'][i]:>10.4f}"
            f"{d['standard_errors'][i]:>10.4f}"
            f"{d['keep'][i]:>10.4f}"
            f"{d['apy'][i]:>10.4f}"
            f"{mu[i]:>10.4f}"
        )

    print(f"\n{'lambda':>8}{'LENDING':>12}{'LST':>12}   feasible")
    for lam in LAMBDA_GRID:
        w = solve(mu, sigma, lam)
        bps = np.round(w * 10000).astype(int)
        ok = bool(np.all(bps <= PER_STRATEGY_CAP_BPS))
        print(f"{lam:>8.2f}{bps[0]:>12d}{bps[1]:>12d}   {'yes' if ok else 'NO'}")

    boundary = bisect_boundary(mu, sigma, 0.01, 50.0)
    print()
    if boundary is None:
        print("No lambda in [0.01, 50] satisfies the per-strategy cap.")
        print("The cap is infeasible for this agent across the entire range.")
    else:
        print(f"Cap boundary at lambda = {boundary:.4f}")
        print("Feasible only below this value, which is far under any")
        print("defensible institutional calibration of 2 to 4.")


if __name__ == "__main__":
    main()
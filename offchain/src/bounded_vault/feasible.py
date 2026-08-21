"""Projection of a proposal onto the constraint layer's feasible set.

The reference monitor rejects; it never rewrites. Nothing here models
on-chain behaviour. This models the operator standing in front of the
monitor: an allocator whose proposal is refused and who resubmits the
nearest admissible one rather than abandoning the rebalance entirely.

Reporting both paths separates two costs the enforced run conflates.
Reject-and-hold measures what an agent loses by proposing something the
vault cannot execute and not responding. Reject-and-project measures what
the constraint costs once the operator has adapted, which is the quantity
a designer choosing a cap actually wants.

The projection is Euclidean: among all weight vectors summing to the
denominator and respecting the cap, it returns the closest to the proposal
in squared distance. With two adapters this reduces to clipping the larger
leg and giving the remainder to the other, but the general form is
implemented so the result need not be re-derived if a third adapter is
added.
"""

from __future__ import annotations

from bounded_vault.schema import AdapterId
from bounded_vault.weights import to_basis_points

BPS_DENOMINATOR = 10_000

# Bisection on a bounded, monotone function over a range of 20000 basis
# points. Sixty steps resolve it far below one basis point, which is the
# smallest quantity the rounding step can represent anyway.
_BISECTION_STEPS = 60


def is_feasible(n_adapters: int, cap: int) -> bool:
    """Whether any allocation across n adapters can sum to the denominator.

    Mirrors the reachability test on ConstraintConfig. Below this bound the
    constraint set is empty and a breach is a property of the arithmetic
    rather than of any agent.
    """
    return n_adapters * cap >= BPS_DENOMINATOR


def enforce_cap(bps: dict[AdapterId, int], cap: int) -> dict[AdapterId, int]:
    """Push any post-rounding overshoot onto adapters with headroom.

    Largest-remainder rounding can lift a weight one basis point above the
    cap when the continuous solution sat exactly on the boundary, which is
    routine here because projection puts it there by construction.
    Redistributing preserves the exact sum.
    """
    adjusted = dict(bps)
    excess = sum(max(0, w - cap) for w in adjusted.values())
    if excess == 0:
        return adjusted

    for adapter, weight in adjusted.items():
        if weight > cap:
            adjusted[adapter] = cap

    for adapter in sorted(adjusted, key=lambda a: adjusted[a]):
        if excess == 0:
            break
        headroom = cap - adjusted[adapter]
        moved = min(headroom, excess)
        adjusted[adapter] += moved
        excess -= moved

    if excess != 0:
        raise ValueError(
            f"cannot satisfy cap of {cap} bps across {len(adjusted)} adapters"
        )
    return adjusted


def project_to_cap(bps: dict[AdapterId, int], cap: int) -> dict[AdapterId, int]:
    """Nearest allocation that sums to the denominator and respects the cap.

    The projection has the form w_i = clip(x_i - theta, 0, cap) for a single
    scalar theta, which is the Lagrange multiplier on the sum constraint.
    theta is negative whenever a leg is clipped, since removing weight from
    the capped leg must be given back to the others. The sum is monotone
    decreasing in theta, so a bisection recovers it.

    Returns the input unchanged when it is already admissible, so an
    accepted proposal is never perturbed by rounding.
    """
    adapters = sorted(bps, key=int)
    n = len(adapters)

    if not is_feasible(n, cap):
        raise ValueError(
            f"cap of {cap} bps across {n} adapters cannot reach {BPS_DENOMINATOR}"
        )

    total = sum(bps.values())
    if total != BPS_DENOMINATOR:
        raise ValueError(f"proposal sums to {total}, expected {BPS_DENOMINATOR}")

    if all(0 <= bps[adapter] <= cap for adapter in adapters):
        return dict(bps)

    x = [float(bps[adapter]) for adapter in adapters]

    # At the lower bracket every leg clips to the cap and the sum is
    # n * cap, which feasibility guarantees is at least the denominator.
    # At the upper bracket every leg floors to zero. The root lies between.
    low, high = -float(BPS_DENOMINATOR), float(BPS_DENOMINATOR)
    for _ in range(_BISECTION_STEPS):
        theta = (low + high) / 2.0
        shifted = sum(min(max(v - theta, 0.0), float(cap)) for v in x)
        if shifted > BPS_DENOMINATOR:
            low = theta
        else:
            high = theta

    theta = (low + high) / 2.0
    projected = [min(max(v - theta, 0.0), float(cap)) for v in x]
    scale = sum(projected)

    fractions = [p / scale for p in projected]
    rounded = dict(zip(adapters, to_basis_points(fractions)))
    return enforce_cap(rounded, cap)
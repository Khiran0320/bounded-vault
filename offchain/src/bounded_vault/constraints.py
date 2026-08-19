"""Off-chain mirror of the on-chain constraint layer.

This module must stay behaviourally identical to validate_proposal in the
Anchor program. The backtest uses it to determine whether a proposal would
have been accepted on chain, so any divergence here silently invalidates
the violation counts that the dissertation reports.

Two deliberate divergences remain, both documented at their check site:
NEGATIVE_WEIGHT has no on-chain counterpart because weights are u16, and
ADAPTER_NOT_ALLOWED approximates a whitelist of CPI target program keys
that has no representation in this layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from bounded_vault.schema import AdapterId

BPS_DENOMINATOR = 10_000


class ViolationReason(str, Enum):
    """Mirrors the on-chain error variants."""

    ADAPTER_NOT_ALLOWED = "adapter_not_allowed"
    NEGATIVE_WEIGHT = "negative_weight"
    STRATEGY_CAP_EXCEEDED = "strategy_cap_exceeded"
    TOTAL_CAP_EXCEEDED = "total_cap_exceeded"
    WEIGHTS_NOT_EXACT = "weights_not_exact"
    REBALANCE_DELTA_EXCEEDED = "rebalance_delta_exceeded"


@dataclass(frozen=True)
class ConstraintConfig:
    """Vault safety parameters.

    These must match the values stored in the on-chain Vault account, where
    max_strategy_bps appears as per_strategy_cap_bps and allowed_adapters
    corresponds loosely to whitelisted_programs.

    require_exact_sum reflects the on-chain rule that weights sum to
    exactly BPS_DENOMINATOR. Setting it False is not an on-chain option;
    it exists so the backtest can run the counterfactual described below.
    """

    allowed_adapters: frozenset[AdapterId]
    max_strategy_bps: int
    total_cap_bps: int = BPS_DENOMINATOR
    max_rebalance_delta_bps: int = BPS_DENOMINATOR
    require_exact_sum: bool = True

    def is_reachable(self) -> bool:
        """Whether any allocation can satisfy every constraint at once.

        Returns False when exact-sum and a sub-total cap are both active,
        since no vector can sum to BPS_DENOMINATOR and stay under a lower
        ceiling. This is the known constraint composition conflict.

        Note that this tests reachability from an unconstrained starting
        point only. A second conflict exists that this cannot express: with
        exact-sum active and a rebalance delta below BPS_DENOMINATOR / n,
        no first rebalance away from a flat zero allocation is admissible,
        even though steady-state adjustments remain legal.
        """
        if self.require_exact_sum and self.total_cap_bps < BPS_DENOMINATOR:
            return False
        n = len(self.allowed_adapters)
        return n * self.max_strategy_bps >= BPS_DENOMINATOR


def validate_proposal(
    proposed_bps: dict[AdapterId, int],
    current_bps: dict[AdapterId, int] | None,
    config: ConstraintConfig,
) -> tuple[bool, ViolationReason | None]:
    """Decide whether a proposal would be accepted on chain.

    current_bps is the vault's existing allocation, used for the rebalance
    delta check. Pass None for the first rebalance, when there is no prior
    position to move away from and the delta check does not apply. The
    on-chain program has no equivalent of None and always compares against
    the stored weights, so the two agree only once a position exists.

    Checks run in the order the on-chain implementation uses, so the
    reported reason matches what the program would emit when a proposal
    breaches more than one rule at once. Adapters are visited in enum
    order, mirroring the fixed strategy array the program indexes over.
    """
    adapters = sorted(proposed_bps, key=int)

    # No on-chain counterpart: weights are u16 and the pydantic schema
    # already rejects negatives. Retained as a structural guard for
    # proposals built outside the schema, and checked first so it cannot
    # be masked by a later rule.
    for adapter in adapters:
        if proposed_bps[adapter] < 0:
            return False, ViolationReason.NEGATIVE_WEIGHT

    total = sum(proposed_bps.values())

    if config.require_exact_sum and total != BPS_DENOMINATOR:
        return False, ViolationReason.WEIGHTS_NOT_EXACT

    # Cap and delta share one pass, as they do on chain. A proposal that
    # breaches the cap on a later adapter and the delta on an earlier one
    # is therefore reported as a delta breach, matching the program.
    for adapter in adapters:
        bps = proposed_bps[adapter]

        if bps > config.max_strategy_bps:
            return False, ViolationReason.STRATEGY_CAP_EXCEEDED

        if current_bps is not None:
            delta = abs(bps - current_bps.get(adapter, 0))
            if delta > config.max_rebalance_delta_bps:
                return False, ViolationReason.REBALANCE_DELTA_EXCEEDED

    if total > config.total_cap_bps:
        return False, ViolationReason.TOTAL_CAP_EXCEEDED

    # Approximates the on-chain whitelist of CPI target program keys, which
    # this layer has no representation of. Checked last, as on chain.
    for adapter in adapters:
        if adapter not in config.allowed_adapters:
            return False, ViolationReason.ADAPTER_NOT_ALLOWED

    return True, None
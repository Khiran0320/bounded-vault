"""Off-chain mirror of the on-chain constraint layer.

This module must stay behaviourally identical to validate_proposal in the
Anchor program. The backtest uses it to determine whether a proposal would
have been accepted on chain, so any divergence here silently invalidates
the violation counts that the dissertation reports.
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

    These must match the values stored in the on-chain Vault account.
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
    position to move away from and the delta check does not apply.

    Checks run in the same order as the on-chain implementation so that
    the reported reason matches what the program would emit.
    """
    for adapter, bps in proposed_bps.items():
        if adapter not in config.allowed_adapters:
            return False, ViolationReason.ADAPTER_NOT_ALLOWED
        if bps < 0:
            return False, ViolationReason.NEGATIVE_WEIGHT
        if bps > config.max_strategy_bps:
            return False, ViolationReason.STRATEGY_CAP_EXCEEDED

    total = sum(proposed_bps.values())

    if config.require_exact_sum and total != BPS_DENOMINATOR:
        return False, ViolationReason.WEIGHTS_NOT_EXACT

    if total > config.total_cap_bps:
        return False, ViolationReason.TOTAL_CAP_EXCEEDED

    if current_bps is not None:
        adapters = set(current_bps) | set(proposed_bps)
        for adapter in adapters:
            delta = abs(proposed_bps.get(adapter, 0) - current_bps.get(adapter, 0))
            if delta > config.max_rebalance_delta_bps:
                return False, ViolationReason.REBALANCE_DELTA_EXCEEDED

    return True, None
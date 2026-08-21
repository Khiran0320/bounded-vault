"""Conformance of the off-chain mirror against the shared fixture.

The dissertation's breach figures come from this module rather than from
the deployed program. This test and its Rust counterpart read one fixture
and assert the same verdicts on every case in the shared section, which
is what licenses reporting those figures as measurements of the on-chain
enforcement logic.

Run from offchain/:
    pytest tests/test_constraint_parity.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bounded_vault.constraints import (
    ConstraintConfig,
    ViolationReason,
    validate_proposal,
)
from bounded_vault.schema import AdapterId

FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "constraint_cases.json"
)

ADAPTERS = {
    "LENDING": AdapterId.LENDING,
    "LIQUID_STAKING": AdapterId.LIQUID_STAKING,
}

# The fixture token vocabulary, mapped to this implementation's reasons.
# Tokens with no entry are ones the mirror cannot produce, and a case
# using one belongs in rust_only rather than in both.
EXPECTED = {
    "weights_not_exact": ViolationReason.WEIGHTS_NOT_EXACT,
    "strategy_cap_exceeded": ViolationReason.STRATEGY_CAP_EXCEEDED,
    "rebalance_delta_exceeded": ViolationReason.REBALANCE_DELTA_EXCEEDED,
    "total_cap_exceeded": ViolationReason.TOTAL_CAP_EXCEEDED,
    "adapter_not_allowed": ViolationReason.ADAPTER_NOT_ALLOWED,
    "negative_weight": ViolationReason.NEGATIVE_WEIGHT,
}


def load():
    if not FIXTURE.exists():
        raise FileNotFoundError(f"shared fixture not found at {FIXTURE}")
    return json.loads(FIXTURE.read_text())


FIXTURE_DATA = load()
CASES = [
    (section, case)
    for section in ("both", "python_only")
    for case in FIXTURE_DATA[section]
]


@pytest.mark.parametrize(
    "section,case", CASES, ids=[f"{s}:{c['id']}" for s, c in CASES]
)
def test_matches_the_shared_fixture(section, case):
    spec = case.get("constraints", FIXTURE_DATA["constraints"])
    adapters = [ADAPTERS[name] for name in case["adapters"]]

    # The whitelist of CPI target program keys has no representation in
    # this layer, so the per-position flag is collapsed into the set of
    # adapters the config permits. This is the documented approximation.
    allowed = frozenset(
        adapter
        for adapter, ok in zip(adapters, case["whitelisted"])
        if ok
    )

    config = ConstraintConfig(
        allowed_adapters=allowed,
        max_strategy_bps=spec["per_strategy_cap_bps"],
        total_cap_bps=spec["total_cap_bps"],
        max_rebalance_delta_bps=spec["max_rebalance_delta_bps"],
        require_exact_sum=True,
    )

    proposed = dict(zip(adapters, case["proposed"]))
    current = (
        None
        if case["current"] is None
        else dict(zip(adapters, case["current"]))
    )

    accepted, reason = validate_proposal(proposed, current, config)

    if case["expect"] == "ok":
        assert accepted, f"{case['id']}: expected ok, refused with {reason}"
        assert reason is None
    else:
        assert not accepted, f"{case['id']}: expected refusal, was accepted"
        assert reason == EXPECTED[case["expect"]], case["id"]


def test_every_shared_token_is_mapped_in_both_directions():
    """Guards the vocabulary itself.

    A case in the shared section using a token this implementation cannot
    produce would otherwise fail as a KeyError deep inside a parametrised
    run, which reads as a broken test rather than as a misfiled case.
    """
    for case in FIXTURE_DATA["both"]:
        token = case["expect"]
        assert token == "ok" or token in EXPECTED, (
            f"{case['id']} expects {token}, which the mirror cannot produce; "
            "it belongs in rust_only"
        )
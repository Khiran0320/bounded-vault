//! Conformance of the on-chain constraint layer against the shared fixture.
//!
//! Every breach figure reported in the dissertation is produced by the
//! Python mirror rather than by this program. This test and its Python
//! counterpart read one fixture and assert the same verdicts, which is
//! what licenses treating those figures as measurements of the deployed
//! enforcement logic rather than of a second implementation that merely
//! resembles it.
//!
//! Run in isolation, since the other integration test needs a built
//! program binary:
//!     cargo test --package bounded-vault --test constraint_parity

use anchor_lang::prelude::Pubkey;
use bounded_vault::error::VaultError;
use bounded_vault::instructions::constraints::validate_proposal;
use bounded_vault::state::{AdapterId, SafetyConstraints, MAX_WHITELISTED_PROGRAMS};
use serde::Deserialize;

// Compile-time include, so a moved fixture fails the build rather than
// the run, and the path does not depend on the working directory.
const FIXTURE: &str = include_str!("../../../fixtures/constraint_cases.json");
#[derive(Deserialize, Clone)]
struct Constraints {
    per_strategy_cap_bps: u16,
    total_cap_bps: u16,
    max_rebalance_delta_bps: u16,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    adapters: Vec<String>,
    proposed: Vec<i64>,
    current: Option<Vec<i64>>,
    whitelisted: Vec<bool>,
    expect: String,
    #[serde(default)]
    constraints: Option<Constraints>,
}

#[derive(Deserialize)]
struct Fixture {
    constraints: Constraints,
    both: Vec<Case>,
    rust_only: Vec<Case>,
}

fn adapter(name: &str) -> AdapterId {
    match name {
        "LENDING" => AdapterId::Lending,
        "LIQUID_STAKING" => AdapterId::LiquidStaking,
        other => panic!("unknown adapter in fixture: {other}"),
    }
}

/// The fixture token vocabulary, mapped to this implementation's errors.
/// Tokens with no entry here are ones the program cannot produce, and a
/// case using one belongs in python_only rather than in both.
fn expected_error(token: &str) -> anchor_lang::error::Error {
    match token {
        "weights_not_exact" => anchor_lang::error!(VaultError::InvalidWeightSum),
        "strategy_cap_exceeded" => {
            anchor_lang::error!(VaultError::PerStrategyCapBreached)
        }
        "rebalance_delta_exceeded" => {
            anchor_lang::error!(VaultError::RebalanceDeltaExceeded)
        }
        "total_cap_exceeded" => anchor_lang::error!(VaultError::TotalCapBreached),
        "adapter_not_allowed" => {
            anchor_lang::error!(VaultError::ProgramNotWhitelisted)
        }
        "duplicate_adapter" => anchor_lang::error!(VaultError::DuplicateAdapter),
        other => panic!("no on-chain error is mapped to fixture token {other}"),
    }
}

fn run(case: &Case, default: &Constraints) {
    let spec = case.constraints.clone().unwrap_or_else(|| default.clone());

    let allowed = Pubkey::new_unique();
    let rogue = Pubkey::new_unique();
    let mut programs = [Pubkey::default(); MAX_WHITELISTED_PROGRAMS];
    programs[0] = allowed;

    let constraints = SafetyConstraints {
        per_strategy_cap_bps: spec.per_strategy_cap_bps,
        total_cap_bps: spec.total_cap_bps,
        max_rebalance_delta_bps: spec.max_rebalance_delta_bps,
        whitelisted_program_count: 1,
        whitelisted_programs: programs,
    };

    let adapter_ids: Vec<AdapterId> =
        case.adapters.iter().map(|n| adapter(n)).collect();

    // python_only cases carry negative weights, which u16 cannot hold.
    // Those cases never reach this function, so the cast is total here.
    let weights: Vec<u16> = case.proposed.iter().map(|w| *w as u16).collect();

    let current: Vec<u16> = case
        .current
        .as_ref()
        .expect("a case with no prior position is python_only")
        .iter()
        .map(|w| *w as u16)
        .collect();

    let targets: Vec<Pubkey> = case
        .whitelisted
        .iter()
        .map(|ok| if *ok { allowed } else { rogue })
        .collect();

    let result = validate_proposal(
        &constraints,
        &adapter_ids,
        &weights,
        &current,
        &targets,
    );

    if case.expect == "ok" {
        assert!(result.is_ok(), "{}: expected ok, got {:?}", case.id, result);
    } else {
        let actual = result.expect_err(&format!(
            "{}: expected {}, got ok",
            case.id, case.expect
        ));
        assert_eq!(actual, expected_error(&case.expect), "case {}", case.id);
    }
}

#[test]
fn matches_the_shared_fixture() {
    let fixture: Fixture =
        serde_json::from_str(FIXTURE).expect("fixture is not valid JSON");

    for case in &fixture.both {
        run(case, &fixture.constraints);
    }
    for case in &fixture.rust_only {
        run(case, &fixture.constraints);
    }

    println!(
        "{} shared cases and {} rust-only cases passed",
        fixture.both.len(),
        fixture.rust_only.len()
    );
}
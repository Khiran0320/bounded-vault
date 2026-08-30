use anchor_lang::prelude::*;
use crate::{
    error::VaultError,
    state::{AdapterId, SafetyConstraints, MAX_STRATEGIES},
    constants::BPS_DENOMINATOR,
};

/// The reference monitor. Every rebalance proposal passes through this
/// function before any adapter is called, and a failure here reverts the
/// whole transaction, leaving the vault in the allocation it already had.
///
/// The function is deliberately small and total: it reads no accounts,
/// performs no CPI, and has no failure mode other than returning one of
/// the declared errors. That is what makes it verifiable by inspection,
/// which is the third of Anderson's reference monitor criteria.
///
/// adapter_ids: which strategy each position refers to
/// weights: proposed allocation in basis points, one entry per strategy
/// current_weights: current allocation in basis points
/// target_programs: the program each strategy routes to
///
/// Check order is load bearing. A proposal that breaches more than one
/// rule is reported against the first rule reached, and the off-chain
/// mirror reproduces this order so that the two agree on the reason and
/// not merely on the verdict.
pub fn validate_proposal(
    constraints: &SafetyConstraints,
    adapter_ids: &[AdapterId],
    weights: &[u16],
    current_weights: &[u16],
    target_programs: &[Pubkey],
) -> Result<()> {
    require!(
        weights.len() <= MAX_STRATEGIES,
        VaultError::InvalidWeightSum
    );
    require!(
        weights.len() == adapter_ids.len(),
        VaultError::InvalidWeightSum
    );
    require!(
        weights.len() == current_weights.len(),
        VaultError::InvalidWeightSum
    );
    require!(
        weights.len() == target_programs.len(),
        VaultError::InvalidWeightSum
    );

    // No adapter may appear twice. Without this, a proposal of
    // [Lending 6000, Lending 4000] clears the per-strategy cap on both
    // entries, sums to the denominator, passes the whitelist, and puts
    // the entire vault in one strategy. The property the cap protects is
    // concentration, and a per-entry test alone does not protect it.
    //
    // Checked before the sum so a duplicate is reported as such rather
    // than masked by a rule it happens to breach as well. The off-chain
    // mirror keys weights by adapter in a dict and cannot represent a
    // proposal of this shape at all, so no fixture case for it appears in
    // the shared section.
    for i in 0..adapter_ids.len() {
        for j in (i + 1)..adapter_ids.len() {
            require!(
                adapter_ids[i] != adapter_ids[j],
                VaultError::DuplicateAdapter
            );
        }
    }

    // Weights must sum to exactly 10000 bps. Accumulated in u32 because
    // four u16 weights can exceed u16::MAX before this check rejects them.
    let total: u32 = weights.iter().map(|w| *w as u32).sum();
    require!(
        total == BPS_DENOMINATOR as u32,
        VaultError::InvalidWeightSum
    );

    // Per-strategy cap and rebalance delta share a single pass over the
    // strategies, so a proposal breaching the cap on a later strategy and
    // the delta on an earlier one is reported as a delta breach.
    for (i, &weight) in weights.iter().enumerate() {
        require!(
            weight <= constraints.per_strategy_cap_bps,
            VaultError::PerStrategyCapBreached
        );

        let current = current_weights[i];
        let delta = if weight > current {
            weight - current
        } else {
            current - weight
        };
        require!(
            delta <= constraints.max_rebalance_delta_bps,
            VaultError::RebalanceDeltaExceeded
        );
    }

    // Total cap. This check can never fire: the exact-sum rule above has
    // already fixed the total at the denominator, so a total_cap_bps below
    // the denominator makes every proposal unsatisfiable at the sum check,
    // and one equal to the denominator makes this comparison vacuous.
    //
    // It is retained rather than deleted because the unreachability is a
    // finding rather than an oversight. Two independently reasonable
    // safety rules compose into an empty feasible set, and nothing in this
    // program detects that at configuration time. Reuses the u32 total,
    // since summing into u16 here would overflow for vectors the earlier
    // check rejects first.
    require!(
        total <= constraints.total_cap_bps as u32,
        VaultError::TotalCapBreached
    );

    // Target programs must be whitelisted. Checked last, so a proposal
    // that is both economically inadmissible and pointed at an unknown
    // program is reported against the economic rule.
    //
    // Note the residual gap: this binds each target program to the
    // whitelist, but nothing binds adapter_id to a particular program. A
    // proposal naming AdapterId::Lending with the staking program's key
    // validates here and dispatches to the lending adapter.
    let whitelisted = &constraints.whitelisted_programs
        [..constraints.whitelisted_program_count as usize];

    for program in target_programs {
        require!(
            whitelisted.contains(program),
            VaultError::ProgramNotWhitelisted
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::SafetyConstraints;
    use anchor_lang::prelude::Pubkey;

    const LENDING: AdapterId = AdapterId::Lending;
    const STAKING: AdapterId = AdapterId::LiquidStaking;

    fn default_constraints() -> SafetyConstraints {
        let program_a = Pubkey::new_unique();
        let mut programs = [Pubkey::default(); 8];
        programs[0] = program_a;

        SafetyConstraints {
            per_strategy_cap_bps: 6000,
            total_cap_bps: 10000,
            max_rebalance_delta_bps: 3000,
            whitelisted_program_count: 1,
            whitelisted_programs: programs,
        }
    }

    #[test]
    fn test_valid_proposal() {
        let c = default_constraints();
        let program_a = c.whitelisted_programs[0];
        let result = validate_proposal(
            &c,
            &[LENDING, STAKING],
            &[5000, 5000],
            &[5000, 5000],
            &[program_a, program_a],
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_per_strategy_cap_breach() {
        let c = default_constraints();
        let program_a = c.whitelisted_programs[0];
        // 7000 bps exceeds per_strategy_cap_bps of 6000
        let result = validate_proposal(
            &c,
            &[LENDING, STAKING],
            &[7000, 3000],
            &[5000, 5000],
            &[program_a, program_a],
        );
        assert_eq!(
            result.unwrap_err(),
            anchor_lang::error!(VaultError::PerStrategyCapBreached)
        );
    }

    #[test]
    fn test_invalid_weight_sum() {
        let c = default_constraints();
        let program_a = c.whitelisted_programs[0];
        // Weights sum to 9000, not 10000
        let result = validate_proposal(
            &c,
            &[LENDING, STAKING],
            &[4500, 4500],
            &[5000, 5000],
            &[program_a, program_a],
        );
        assert_eq!(
            result.unwrap_err(),
            anchor_lang::error!(VaultError::InvalidWeightSum)
        );
    }

    #[test]
    fn test_rebalance_delta_exceeded() {
        let c = default_constraints();
        let program_a = c.whitelisted_programs[0];
        // First leg moves 4000 bps against a limit of 3000
        let result = validate_proposal(
            &c,
            &[LENDING, STAKING],
            &[1000, 9000],
            &[5000, 5000],
            &[program_a, program_a],
        );
        assert_eq!(
            result.unwrap_err(),
            anchor_lang::error!(VaultError::RebalanceDeltaExceeded)
        );
    }

    #[test]
    fn test_program_not_whitelisted() {
        let c = default_constraints();
        let rogue_program = Pubkey::new_unique();
        let result = validate_proposal(
            &c,
            &[LENDING, STAKING],
            &[5000, 5000],
            &[5000, 5000],
            &[c.whitelisted_programs[0], rogue_program],
        );
        assert_eq!(
            result.unwrap_err(),
            anchor_lang::error!(VaultError::ProgramNotWhitelisted)
        );
    }

    #[test]
    fn test_duplicate_adapter_is_refused() {
        // Both entries clear the cap and the pair sums to the denominator,
        // so every other rule is satisfied. Only the uniqueness check
        // stands between this proposal and full concentration in lending.
        let c = default_constraints();
        let program_a = c.whitelisted_programs[0];
        let result = validate_proposal(
            &c,
            &[LENDING, LENDING],
            &[6000, 4000],
            &[5000, 5000],
            &[program_a, program_a],
        );
        assert_eq!(
            result.unwrap_err(),
            anchor_lang::error!(VaultError::DuplicateAdapter)
        );
    }

    #[test]
    fn test_cap_is_reported_before_delta_on_the_same_leg() {
        // Pins the documented check order, which the off-chain mirror
        // reproduces. A change here silently invalidates the rejection
        // reasons reported throughout the results chapter.
        let c = default_constraints();
        let program_a = c.whitelisted_programs[0];
        let result = validate_proposal(
            &c,
            &[LENDING, STAKING],
            &[6500, 3500],
            &[1000, 9000],
            &[program_a, program_a],
        );
        assert_eq!(
            result.unwrap_err(),
            anchor_lang::error!(VaultError::PerStrategyCapBreached)
        );
    }
}
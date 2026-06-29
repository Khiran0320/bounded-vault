use anchor_lang::prelude::*;
use crate::{
    error::VaultError,
    state::{SafetyConstraints, MAX_STRATEGIES},
    constants::BPS_DENOMINATOR,
};

/// weights: proposed allocation in basis points, one entry per strategy
/// current_weights: current allocation in basis points
/// target_programs: the program each strategy routes to
pub fn validate_proposal(
    constraints: &SafetyConstraints,
    weights: &[u16],
    current_weights: &[u16],
    target_programs: &[Pubkey],
) -> Result<()> {
    require!(
        weights.len() <= MAX_STRATEGIES,
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

    // Weights must sum to exactly 10000 bps
    let total: u32 = weights.iter().map(|w| *w as u32).sum();
    require!(
        total == BPS_DENOMINATOR as u32,
        VaultError::InvalidWeightSum
    );

    // Per-strategy cap
    for (i, &weight) in weights.iter().enumerate() {
        require!(
            weight <= constraints.per_strategy_cap_bps,
            VaultError::PerStrategyCapBreached
        );

        // Check 3: max rebalance delta per strategy
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

    // Total cap (sum of weights already validated to 10000,
    // but total_cap_bps may be set below 10000 to keep a cash buffer)
    let total_u16 = weights.iter().sum::<u16>();
    require!(
        total_u16 <= constraints.total_cap_bps,
        VaultError::TotalCapBreached
    );

    // Target programs must be whitelisted
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
        let result = validate_proposal(
            &c,
            &[1000, 3000, 6000],
            &[5000, 3000, 2000],
            &[program_a, program_a, program_a],
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
            &[5000, 5000],
            &[5000, 5000],
            &[c.whitelisted_programs[0], rogue_program],
        );
        assert_eq!(
            result.unwrap_err(),
            anchor_lang::error!(VaultError::ProgramNotWhitelisted)
        );
    }
}
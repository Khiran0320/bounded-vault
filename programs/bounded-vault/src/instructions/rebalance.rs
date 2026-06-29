use anchor_lang::prelude::*;
use crate::{
    constants::VAULT_SEED,
    error::VaultError,
    state::{Vault, AdapterId},
    instructions::constraints::validate_proposal,
};
use super::adapters;

#[derive(AnchorSerialize, AnchorDeserialize)]
pub struct StrategyInput {
    pub adapter_id: AdapterId,
    pub program_id: Pubkey,
    pub proposed_weight_bps: u16,
    pub current_weight_bps: u16,
}

#[derive(Accounts)]
pub struct Rebalance<'info> {
    pub authority: Signer<'info>,

    #[account(
        mut,
        seeds = [VAULT_SEED, vault.asset_mint.as_ref()],
        bump = vault.bump,
        has_one = authority,
    )]
    pub vault: Account<'info, Vault>,
}

pub fn handler(
    ctx: Context<Rebalance>,
    strategies: Vec<StrategyInput>,
) -> Result<()> {
    require!(!ctx.accounts.vault.paused, VaultError::VaultPaused);
    require!(!strategies.is_empty(), VaultError::InvalidWeightSum);

    let proposed_weights: Vec<u16> = strategies.iter()
        .map(|s| s.proposed_weight_bps)
        .collect();

    let current_weights: Vec<u16> = strategies.iter()
        .map(|s| s.current_weight_bps)
        .collect();

    let target_programs: Vec<Pubkey> = strategies.iter()
        .map(|s| s.program_id)
        .collect();

    // Constraint validation runs atomically before any adapter is called.
    // If any check fails the entire transaction reverts here.
    validate_proposal(
        &ctx.accounts.vault.constraints,
        &proposed_weights,
        &current_weights,
        &target_programs,
    )?;

    let total_assets = ctx.accounts.vault.total_assets;

    // Dispatch to the appropriate adapter for each strategy.
    for strategy in &strategies {
        let amount = (strategy.proposed_weight_bps as u64)
            .checked_mul(total_assets)
            .ok_or(VaultError::MathOverflow)?
            .checked_div(10_000)
            .ok_or(VaultError::MathOverflow)?;

        match strategy.adapter_id {
            AdapterId::Lending => {
                adapters::lending::deposit(&strategy.program_id, amount)?;
            }
            AdapterId::LiquidStaking => {
                adapters::liquid_staking::deposit(&strategy.program_id, amount)?;
            }
        }
    }

    msg!(
        "Rebalance complete. {} strategies dispatched. Total assets: {}",
        strategies.len(),
        total_assets,
    );

    Ok(())
}
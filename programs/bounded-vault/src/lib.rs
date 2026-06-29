use anchor_lang::prelude::*;

pub mod constants;
pub mod error;
pub mod state;
pub mod instructions;

use instructions::initialize_vault::*;
use instructions::deposits::*;
use instructions::withdraw::*;
use instructions::rebalance::*;


declare_id!("5UmkjGjzK4hFv2Z5Vo52iB4auwBgHX91NpZg5Pp483xP");

#[program]
pub mod bounded_vault {
    use super::*;

    pub fn initialize_vault(
        ctx: Context<InitializeVault>,
        asset_mint: Pubkey,
        per_strategy_cap_bps: u16,
        total_cap_bps: u16,
        max_rebalance_delta_bps: u16,
        whitelisted_programs: Vec<Pubkey>,
    ) -> Result<()> {
        instructions::initialize_vault::handler(
            ctx,
            asset_mint,
            per_strategy_cap_bps,
            total_cap_bps,
            max_rebalance_delta_bps,
            whitelisted_programs,
        )
    }

    pub fn rebalance(
        ctx: Context<Rebalance>,
        strategies: Vec<StrategyInput>,
    ) -> Result<()> {
        instructions::rebalance::handler(ctx, strategies)
    }
        

    pub fn deposits(ctx: Context<Deposit>, amount: u64) -> Result<()> {
        instructions::deposits::handler(ctx, amount)
    }

    pub fn withdraw(ctx: Context<Withdraw>, shares: u64) -> Result<()> {
        instructions::withdraw::handler(ctx, shares)
    }
}
use anchor_lang::prelude::*;

/// Liquid staking adapter: allocates tokens to a liquid staking protocol.
pub fn deposit(
    program_id: &Pubkey,
    amount: u64,
) -> Result<()> {
    msg!(
        "LiquidStakingAdapter: staking {} tokens via program {}",
        amount,
        program_id,
    );
    // CPI to liquid staking protocol would go here.
    Ok(())
}

pub fn withdraw(
    program_id: &Pubkey,
    amount: u64,
) -> Result<()> {
    msg!(
        "LiquidStakingAdapter: unstaking {} tokens via instant-unstake pool, program {}",
        amount,
        program_id,
    );
    // Instant-unstake routes through an LST liquidity pool
    Ok(())
}
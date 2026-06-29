use anchor_lang::prelude::*;

/// Lending adapter: allocates tokens to a lending protocol.
/// On devnet this logs the intended action.
pub fn deposit(
    program_id: &Pubkey,
    amount: u64,
) -> Result<()> {
    msg!(
        "LendingAdapter: depositing {} tokens into lending program {}",
        amount,
        program_id,
    );
    Ok(())
}

pub fn withdraw(
    program_id: &Pubkey,
    amount: u64,
) -> Result<()> {
    msg!(
        "LendingAdapter: withdrawing {} tokens from lending program {}",
        amount,
        program_id,
    );
    Ok(())
}
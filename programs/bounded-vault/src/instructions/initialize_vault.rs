use anchor_lang::prelude::*;
use anchor_spl::token_interface::{Mint, TokenAccount, TokenInterface};
use crate::{
    constants::{VAULT_SEED, VAULT_TOKEN_SEED, SHARE_MINT_SEED},
    state::{Vault, SafetyConstraints, MAX_WHITELISTED_PROGRAMS},
};

#[derive(Accounts)]
#[instruction(asset_mint: Pubkey)]
pub struct InitializeVault<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    /// The vault state account, derived from VAULT_SEED + asset_mint
    #[account(
        init,
        payer = authority,
        space = Vault::LEN,
        seeds = [VAULT_SEED, asset_mint.as_ref()],
        bump,
    )]
    pub vault: Account<'info, Vault>,

    /// The underlying asset mint (e.g. USDC devnet mint)
    pub asset_mint_account: InterfaceAccount<'info, Mint>,

    /// Token account that holds custody of deposited assets
    #[account(
        init,
        payer = authority,
        token::mint = asset_mint_account,
        token::authority = vault,
        seeds = [VAULT_TOKEN_SEED, asset_mint.as_ref()],
        bump,
    )]
    pub vault_token_account: InterfaceAccount<'info, TokenAccount>,

    /// Share mint: vault issues these to depositors
    #[account(
        init,
        payer = authority,
        mint::decimals = 6,
        mint::authority = vault,
        seeds = [SHARE_MINT_SEED, asset_mint.as_ref()],
        bump,
    )]
    pub share_mint: InterfaceAccount<'info, Mint>,

    pub token_program: Interface<'info, TokenInterface>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

pub fn handler(
    ctx: Context<InitializeVault>,
    asset_mint: Pubkey,
    per_strategy_cap_bps: u16,
    total_cap_bps: u16,
    max_rebalance_delta_bps: u16,
    whitelisted_programs: Vec<Pubkey>,
) -> Result<()> {
    require!(
        whitelisted_programs.len() <= MAX_WHITELISTED_PROGRAMS,
        crate::error::VaultError::ProgramNotWhitelisted
    );

    let vault = &mut ctx.accounts.vault;

    vault.authority = ctx.accounts.authority.key();
    vault.asset_mint = asset_mint;
    vault.vault_token_account = ctx.accounts.vault_token_account.key();
    vault.share_mint = ctx.accounts.share_mint.key();
    vault.total_assets = 0;
    vault.total_shares = 0;
    vault.paused = false;
    vault.bump = ctx.bumps.vault;

    let mut programs_array = [Pubkey::default(); MAX_WHITELISTED_PROGRAMS];
    for (i, pk) in whitelisted_programs.iter().enumerate() {
        programs_array[i] = *pk;
    }

    vault.constraints = SafetyConstraints {
        per_strategy_cap_bps,
        total_cap_bps,
        max_rebalance_delta_bps,
        whitelisted_program_count: whitelisted_programs.len() as u8,
        whitelisted_programs: programs_array,
    };

    msg!(
        "Vault initialized. Asset: {}. Per-strategy cap: {} bps. Total cap: {} bps.",
        asset_mint,
        per_strategy_cap_bps,
        total_cap_bps,
    );

    Ok(())
}
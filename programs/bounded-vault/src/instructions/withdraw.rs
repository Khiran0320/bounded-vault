use anchor_lang::prelude::*;
use anchor_spl::token_interface::{
    Mint, TokenAccount, TokenInterface,
    TransferChecked, transfer_checked,
    Burn, burn,
};
use crate::{
    constants::{VAULT_SEED, VAULT_TOKEN_SEED, SHARE_MINT_SEED},
    error::VaultError,
    state::Vault,
};

#[derive(Accounts)]
pub struct Withdraw<'info> {
    #[account(mut)]
    pub user: Signer<'info>,

    #[account(
        mut,
        seeds = [VAULT_SEED, vault.asset_mint.as_ref()],
        bump = vault.bump,
    )]
    pub vault: Account<'info, Vault>,

    #[account(
        mut,
        token::mint = asset_mint,
        token::authority = user,
    )]
    pub user_token_account: Box<InterfaceAccount<'info, TokenAccount>>,

    #[account(
        mut,
        seeds = [VAULT_TOKEN_SEED, vault.asset_mint.as_ref()],
        bump,
        token::mint = asset_mint,
        token::authority = vault,
    )]
    pub vault_token_account: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(
        mut,
        seeds = [SHARE_MINT_SEED, vault.asset_mint.as_ref()],
        bump,
        mint::authority = vault,
    )]
    pub share_mint: Box<InterfaceAccount<'info, Mint>>,

    #[account(
        mut,
        token::mint = share_mint,
        token::authority = user,
    )]
    pub user_share_account: Box<InterfaceAccount<'info, TokenAccount>>,

    pub asset_mint: Box<InterfaceAccount<'info, Mint>>,

    pub token_program: Interface<'info, TokenInterface>,
    pub system_program: Program<'info, System>,
}

pub fn handler(ctx: Context<Withdraw>, shares: u64) -> Result<()> {
    require!(shares > 0, VaultError::ZeroWithdraw);
    require!(!ctx.accounts.vault.paused, VaultError::VaultPaused);
    require!(
        ctx.accounts.user_share_account.amount >= shares,
        VaultError::InsufficientShares
    );

    let vault = &ctx.accounts.vault;
    let decimals = ctx.accounts.asset_mint.decimals;

    let tokens_out = (shares as u128)
        .checked_mul(vault.total_assets as u128)
        .ok_or(VaultError::MathOverflow)?
        .checked_div(vault.total_shares as u128)
        .ok_or(VaultError::MathOverflow)? as u64;

    let asset_mint_key = vault.asset_mint;
    let vault_bump = vault.bump;
    let signer_seeds: &[&[&[u8]]] = &[&[
        VAULT_SEED,
        asset_mint_key.as_ref(),
        &[vault_bump],
    ]];

    burn(
        CpiContext::new(
            ctx.accounts.token_program.key(),
            Burn {
                mint: ctx.accounts.share_mint.to_account_info(),
                from: ctx.accounts.user_share_account.to_account_info(),
                authority: ctx.accounts.user.to_account_info(),
            },
        ),
        shares,
    )?;

    transfer_checked(
        CpiContext::new_with_signer(
            ctx.accounts.token_program.key(),
            TransferChecked {
                from: ctx.accounts.vault_token_account.to_account_info(),
                mint: ctx.accounts.asset_mint.to_account_info(),
                to: ctx.accounts.user_token_account.to_account_info(),
                authority: ctx.accounts.vault.to_account_info(),
            },
            signer_seeds,
        ),
        tokens_out,
        decimals,
    )?;

    let vault = &mut ctx.accounts.vault;
    vault.total_assets = vault.total_assets
        .checked_sub(tokens_out)
        .ok_or(VaultError::MathOverflow)?;
    vault.total_shares = vault.total_shares
        .checked_sub(shares)
        .ok_or(VaultError::MathOverflow)?;

    msg!(
        "Withdraw: {} shares burned, {} tokens returned. Total assets: {}, total shares: {}",
        shares,
        tokens_out,
        vault.total_assets,
        vault.total_shares,
    );

    Ok(())
}
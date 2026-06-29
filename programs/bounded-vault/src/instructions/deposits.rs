use anchor_lang::prelude::*;
use anchor_spl::token_interface::{
    Mint, TokenAccount, TokenInterface,
    TransferChecked, transfer_checked,
    MintTo, mint_to,
};
use crate::{
    constants::{VAULT_SEED, VAULT_TOKEN_SEED, SHARE_MINT_SEED},
    error::VaultError,
    state::Vault,
};

#[derive(Accounts)]
pub struct Deposit<'info> {
    #[account(mut)]
    pub user: Signer<'info>,

    #[account(
        mut,
        seeds = [VAULT_SEED, vault.asset_mint.as_ref()],
        bump = vault.bump,
    )]
    pub vault: Account<'info, Vault>,

    /// User's token account (source of deposit)
    #[account(
        mut,
        token::mint = asset_mint,
        token::authority = user,
    )]
    pub user_token_account: Box<InterfaceAccount<'info, TokenAccount>>,

    /// Vault's custody token account (destination)
    #[account(
        mut,
        seeds = [VAULT_TOKEN_SEED, vault.asset_mint.as_ref()],
        bump,
        token::mint = asset_mint,
        token::authority = vault,
    )]
    pub vault_token_account: Box<InterfaceAccount<'info, TokenAccount>>,

    /// Share mint - vault will mint shares to the user
    #[account(
        mut,
        seeds = [SHARE_MINT_SEED, vault.asset_mint.as_ref()],
        bump,
        mint::authority = vault,
    )]
    pub share_mint: Box<InterfaceAccount<'info, Mint>>,

    /// User's share token account (receives minted shares)
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

pub fn handler(ctx: Context<Deposit>, amount: u64) -> Result<()> {
    require!(amount > 0, VaultError::ZeroDeposit);
    require!(!ctx.accounts.vault.paused, VaultError::VaultPaused);

    let vault = &ctx.accounts.vault;
    let decimals = ctx.accounts.asset_mint.decimals;

    // Calculate shares to mint
    let shares_to_mint = if vault.total_shares == 0 || vault.total_assets == 0 {
        // First deposit: 1:1
        amount
    } else {
        // shares = (amount * total_shares) / total_assets
        (amount as u128)
            .checked_mul(vault.total_shares as u128)
            .ok_or(VaultError::MathOverflow)?
            .checked_div(vault.total_assets as u128)
            .ok_or(VaultError::MathOverflow)? as u64
    };

    // Transfer tokens from user to vault
        transfer_checked(
            CpiContext::new(
                ctx.accounts.token_program.key(),
                TransferChecked {
                    from: ctx.accounts.user_token_account.to_account_info(),
                    mint: ctx.accounts.asset_mint.to_account_info(),
                    to: ctx.accounts.vault_token_account.to_account_info(),
                    authority: ctx.accounts.user.to_account_info(),
                },
            ),
            amount,
            decimals,
        )?;
    // Define signer seeds for vault PDA
        let asset_mint_key = vault.asset_mint;
        let vault_bump = vault.bump;
        let signer_seeds: &[&[&[u8]]] = &[&[
            VAULT_SEED,
            asset_mint_key.as_ref(),
            &[vault_bump],
        ]];

    // Mint shares to user - vault PDA signs
        mint_to(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.key(),
                MintTo {
                    mint: ctx.accounts.share_mint.to_account_info(),
                    to: ctx.accounts.user_share_account.to_account_info(),
                    authority: ctx.accounts.vault.to_account_info(),
                },
                signer_seeds,
            ),
            shares_to_mint,
        )?;

    // Update vault state
    let vault = &mut ctx.accounts.vault;
    vault.total_assets = vault.total_assets
        .checked_add(amount)
        .ok_or(VaultError::MathOverflow)?;
    vault.total_shares = vault.total_shares
        .checked_add(shares_to_mint)
        .ok_or(VaultError::MathOverflow)?;

    msg!(
        "Deposit: {} tokens, {} shares minted. Total assets: {}, total shares: {}",
        amount,
        shares_to_mint,
        vault.total_assets,
        vault.total_shares,
    );

    Ok(())
}
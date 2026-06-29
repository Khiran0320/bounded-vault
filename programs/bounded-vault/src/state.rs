use anchor_lang::prelude::*;

pub const MAX_STRATEGIES: usize = 4;
pub const MAX_WHITELISTED_PROGRAMS: usize = 8;

#[account]
pub struct Vault {
    /// Authority that can update constraints and pause the vault
    pub authority: Pubkey,
    /// Mint of the underlying asset (e.g. USDC)
    pub asset_mint: Pubkey,
    /// PDA token account holding custody of deposited assets
    pub vault_token_account: Pubkey,
    /// Mint for vault shares issued to depositors
    pub share_mint: Pubkey,
    /// Total underlying assets under management (in token decimals)
    pub total_assets: u64,
    /// Total share tokens in circulation
    pub total_shares: u64,
    /// Whether the vault is paused (blocks deposits, withdraws, rebalance)
    pub paused: bool,
    /// Bump seed for PDA derivation
    pub bump: u8,
    /// Safety constraints enforced on every rebalance proposal
    pub constraints: SafetyConstraints,
}

impl Vault {
    /// Account space: discriminator + fields + constraints
    pub const LEN: usize = 8          // discriminator
        + 32                          // authority
        + 32                          // asset_mint
        + 32                          // vault_token_account
        + 32                          // share_mint
        + 8                           // total_assets
        + 8                           // total_shares
        + 1                           // paused
        + 1                           // bump
        + SafetyConstraints::LEN;
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct SafetyConstraints {
    /// Max allocation to any single strategy, in basis points (e.g. 6000 = 60%)
    pub per_strategy_cap_bps: u16,
    /// Max total allocation across all strategies, in basis points (must be <= 10000)
    pub total_cap_bps: u16,
    /// Max movement allowed in a single rebalance, in basis points
    pub max_rebalance_delta_bps: u16,
    /// Number of active whitelisted programs
    pub whitelisted_program_count: u8,
    /// Pubkeys of programs the vault is permitted to CPI into
    pub whitelisted_programs: [Pubkey; MAX_WHITELISTED_PROGRAMS],
}

impl SafetyConstraints {
    pub const LEN: usize =
        2                                      // per_strategy_cap_bps
        + 2                                    // total_cap_bps
        + 2                                    // max_rebalance_delta_bps
        + 1                                    // whitelisted_program_count
        + (32 * MAX_WHITELISTED_PROGRAMS);     // whitelisted_programs
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq)]
pub enum AdapterId {
    Lending,
    LiquidStaking,
}
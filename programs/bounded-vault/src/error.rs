use anchor_lang::prelude::*;

#[error_code]
pub enum VaultError {
    #[msg("Vault is currently paused")]
    VaultPaused,

    #[msg("Per-strategy allocation exceeds cap")]
    PerStrategyCapBreached,

    #[msg("Total allocation across strategies exceeds cap")]
    TotalCapBreached,

    #[msg("Rebalance delta exceeds maximum allowed movement")]
    RebalanceDeltaExceeded,

    #[msg("Target program is not whitelisted")]
    ProgramNotWhitelisted,

    #[msg("Allocation weights do not sum to 10000 basis points")]
    InvalidWeightSum,

    #[msg("Arithmetic overflow")]
    MathOverflow,

    #[msg("Zero deposit amount")]
    ZeroDeposit,

    #[msg("Zero withdraw amount")]
    ZeroWithdraw,

    #[msg("Insufficient shares to withdraw")]
    InsufficientShares,
}
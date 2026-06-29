use {
    anchor_lang::{solana_program::instruction::Instruction, InstructionData, ToAccountMetas},
    anchor_lang::prelude::Pubkey,
    litesvm::LiteSVM,
    solana_message::{Message, VersionedMessage},
    solana_signer::Signer,
    solana_keypair::Keypair,
    solana_transaction::versioned::VersionedTransaction,
};

#[test]
fn test_initialize_vault() {
    let program_id = bounded_vault::id();
    let payer = Keypair::new();
    let mut svm = LiteSVM::new();

    let bytes = include_bytes!("../../../target/deploy/bounded_vault.so");
    svm.add_program(program_id, bytes).unwrap();
    svm.airdrop(&payer.pubkey(), 1_000_000_000).unwrap();

    // Use a fake mint pubkey for PDA derivation in this smoke test
    let asset_mint = Pubkey::new_unique();

    // Derive the expected PDAs
    let (vault_pda, _) = Pubkey::find_program_address(
        &[b"vault", asset_mint.as_ref()],
        &program_id,
    );
    let (vault_token_pda, _) = Pubkey::find_program_address(
        &[b"vault_token", asset_mint.as_ref()],
        &program_id,
    );
    let (share_mint_pda, _) = Pubkey::find_program_address(
        &[b"share_mint", asset_mint.as_ref()],
        &program_id,
    );

    let instruction = Instruction::new_with_bytes(
        program_id,
        &bounded_vault::instruction::InitializeVault {
            asset_mint,
            per_strategy_cap_bps: 6000,
            total_cap_bps: 10000,
            max_rebalance_delta_bps: 2000,
            whitelisted_programs: vec![],
        }
        .data(),
        bounded_vault::accounts::InitializeVault {
            authority: payer.pubkey(),
            vault: vault_pda,
            asset_mint_account: asset_mint,
            vault_token_account: vault_token_pda,
            share_mint: share_mint_pda,
            token_program: anchor_spl::token::ID,
            system_program: anchor_lang::solana_program::system_program::ID,
            rent: "SysvarRent111111111111111111111111111111111".parse::<Pubkey>().unwrap(),        }
        .to_account_metas(None),
    );

    let blockhash = svm.latest_blockhash();
    let msg = Message::new_with_blockhash(&[instruction], Some(&payer.pubkey()), &blockhash);
    let tx = VersionedTransaction::try_new(VersionedMessage::Legacy(msg), &[payer]).unwrap();

    let res = svm.send_transaction(tx);
    assert!(res.is_ok(), "initialize_vault failed: {:?}", res);
}
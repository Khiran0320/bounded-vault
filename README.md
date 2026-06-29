# Bounded-Vault

A bounded autonomy architecture for decentralized capital allocation.

## Project Overview

Bounded-Vault investigates whether intelligent capital allocation decisions can be safely delegated to autonomous agents when a separate, rigid enforcement layer retains sole authority over fund movement.

The system separates two concerns that are usually conflated in DeFi vault design:

- **Flexible strategy intelligence**, handled off-chain by Python agents.
- **Hard safety enforcement**, handled on-chain by a Rust/Anchor constraint layer.

This separation, and the argument that constraints belong structurally on-chain rather than inside the agent, forms the core academic contribution.

The vault is implemented and tested on Solana devnet only. Mainnet deployment and reinforcement learning are explicitly out of scope.

## Architecture Summary

The system has two halves that communicate through a single proposal interface.

**Off-chain agents** (rising in complexity): a rule-based agent, a mean-variance optimisation agent, and an LLM-augmented agent. Each agent produces a proposal, an allocation across yield strategies, based only on information available up to that point in time.

**On-chain constraint layer** (Anchor program on Solana): the sole authority that can move funds. It receives proposals and validates them against hard safety limits before any rebalance is executed.

Because every proposal passes through the same validation regardless of which agent produced it, the on-chain program is agent-agnostic: agents can be swapped or upgraded without touching the contract. This is the architectural property the dissertation argues for, using the reference monitor pattern from the security literature as its conceptual grounding.

## Progress to Date

### On-chain program (Stages 1 to 4): complete and tested

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Vault state initialisation: three PDAs (vault state, custody token account, share mint) | Complete |
| 2 | Deposit and withdraw with share-based accounting (ERC-4626-style math) | Complete |
| 3 | Constraint module, with unit tests covering all violation cases | Complete, 5/5 tests passing |
| 4 | Rebalance instruction: `StrategyInput` struct, `AdapterId` enum dispatch, lending and liquid staking adapter stubs | Complete |

### Off-chain system (Stages 5 to 6): scaffolding complete, first agent live

- Monorepo Python structure under `offchain/`, installable as the `bounded_vault` package.
- Shared schema (`Proposal`, `AdapterId`, `StrategyAllocation`) and weight conversion utilities using basis points for scale invariance.
- Abstract base class for all agents, ensuring a consistent interface.
- `MarketView`: an immutable, point-in-time snapshot that structurally enforces a no-lookahead invariant.
- Live data loader pulling price history from CoinGecko and current APY from DeFiLlama.
- Agent 1 (rule-based, yield-proportional baseline): fully implemented and unit tested with pytest.

## Key Technical and Academic Decisions

**Constraints belong on-chain, not in the agent.** Agents only ever hold an opinion. The on-chain program is the single point of enforcement, which means no agent, however sophisticated or flawed, can move funds outside the defined safety limits.

**No-lookahead is enforced structurally, not by convention.** `MarketView` is immutable, so an agent cannot accidentally access future data during backtesting, removing a common source of invalid results in this kind of research.

**Two separate price data paths.** CoinGecko and DeFiLlama are used off-chain for agent training and evaluation; Pyth is used on-chain for live vault validation. These are kept deliberately separate so off-chain research data never leaks into on-chain decision-making.

**Adapters are stubbed with `msg!` logs rather than live CPIs.** This keeps the on-chain program testable without depending on the uptime or behaviour of external protocols, while leaving a clear extension point for the optional stretch goal.

**Basis points throughout for allocation weights,** for scale invariance across strategies.

## Testing and Verification

The constraint module has full unit test coverage of all violation cases (5/5 passing), verifying that the enforcement layer rejects any proposal outside the defined safety bounds.

Agent 1 is unit tested with pytest and has been validated against live market data, not just synthetic fixtures, confirming that the data loading and proposal pipeline work end to end before further agents are built on top of it.

## Current Status and Next Steps

**Immediate next step:** build out Agent 2 before extending the schema further.

**Planned through September:**

- Agent 2: mean-variance optimisation (scipy/cvxpy).
- Agent 3: LLM-augmented agent using a frontier API.
- Offline backtest harness with strict no-lookahead discipline across all three agents.
- Sharpe ratio and empirical evaluation layer comparing agent performance.
- Python mirror of the on-chain constraint logic, for cross-validation against the Rust implementation.
- Stretch goal (timeboxed, attempted last): connect one adapter to a real devnet protocol via live CPI, replacing the `msg!` stub. Not required for the dissertation's core thesis.

## Repository Navigation Guide

For a reader cloning the repository for the first time:

- `onchain/`: the Anchor program (Rust). Contains vault state, deposit/withdraw logic, the constraint module, and the rebalance instruction with adapter dispatch.
- `offchain/`: the Python package `bounded_vault`. Contains shared schema and weight utilities, the agent base class, `MarketView`, the data loaders, and Agent 1.

Tests live alongside their respective implementations: Rust unit tests in `onchain/`, the pytest suite in `offchain/`.

# Bounded-Vault

A two-layer architecture for constrained autonomous capital allocation in DeFi,
in which an untrusted off-chain agent proposes portfolio allocations and a
deterministic on-chain reference monitor decides whether they execute.

## Research question

Can an autonomous agent be given discretion over capital allocation while a
separate enforcement layer retains sole authority over fund movement, and what
does enforcement cost when the agent is told about the constraint rather than
merely subjected to it?

The answer the system supports has three parts:

1. Constraints placed in the agent are advisory. Constraints placed in the
   settlement path are binding. The separation is architectural, not a matter
   of agent quality.
2. Disclosing a constraint to an LLM agent does not make it project its
   preferred allocation onto the feasible set. It perturbs the objective, so
   the agent can arrive at a different and sometimes inverted preference.
3. Complete mediation over an action is not complete mediation over a decision
   when the mediated party supplies the operands the decision is computed from.

## Prerequisites

The on-chain half of this repository builds and runs on macOS or Linux only.
The Solana and Anchor toolchains have no native Windows support. On Windows,
work inside WSL2 (Ubuntu 22.04 or later).

The off-chain Python package is platform independent. Every result, table and
figure reported in the dissertation can be reproduced on any operating system
with Python 3.12 and the committed snapshot, without building the Rust program.

| Component   | Version                                | Platform       |
|-------------|----------------------------------------|----------------|
| Rust        | 1.89.0 (pinned in `rust-toolchain.toml`) | macOS or Linux |
| Anchor CLI  | 1.0.1                                  | macOS or Linux |
| Solana CLI  | 3.1.15                                 | macOS or Linux |
| Node / Yarn | 1.22.22                                | any            |
| Python      | 3.12 (3.10 minimum)                    | any            |

The LiteSVM test suite loads a compiled program binary from
`target/deploy/bounded_vault.so`, so `anchor build` must succeed before
`cargo test` will run. That is why the platform requirement is hard rather
than advisory.

## Installation

On-chain:

```
anchor build
```

Off-chain, from `offchain/`:

```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install cvxpy requests
```

`cvxpy` and `requests` are imported by the mean-variance agent and the snapshot
fetcher respectively but are not declared in `pyproject.toml`. They are
installed explicitly above rather than by editing the frozen package metadata.

An `ANTHROPIC_API_KEY` is required only to populate the Agent 3 response cache
from scratch. The committed cache under `offchain/data/llm_cache/` makes every
reported run reproducible offline, and no key is needed to regenerate the
results below.

## Reproducing the results

All commands run from `offchain/` against the frozen snapshot `2026-07-13`,
which covers 136 decision dates.

```
python scripts/run_backtests.py        # every agent and arm, writes parquet
python scripts/summarise_results.py    # cross-run allocation and breach table
python scripts/breach_curve.py         # cap sweep, writes CSV and figure
python scripts/value_layer.py          # benchmarks, cost sweep, equity curves
python scripts/lambda_band.py          # Agent 2 feasibility across risk aversion
python scripts/compare_llm_configs.py  # constrained vs unconstrained, date by date
python scripts/inspect_llm_inversions.py  # the inverted dates, with reasoning
```

Outputs land in `offchain/data/results/` and `offchain/data/figures/`.

`scripts/fetch_snapshot.py` refetches market data from live APIs. It is not
part of reproduction. The snapshot is frozen deliberately, and the script
refuses to overwrite an existing snapshot directory.

## Repository layout

```
programs/bounded-vault/        Anchor program (Rust)
  src/state.rs                 Vault account and SafetyConstraints
  src/error.rs                 VaultError variants
  src/instructions/
    initialize_vault.rs        Three PDAs: vault state, custody, share mint
    deposits.rs                Share-based deposit accounting
    withdraw.rs                Share-based withdrawal accounting
    constraints.rs             validate_proposal, the reference monitor
    rebalance.rs               Proposal entry point and adapter dispatch
    adapters.rs                Lending and liquid staking adapter stubs
  tests/                       LiteSVM integration tests

offchain/
  src/bounded_vault/
    schema.py                  Proposal, AdapterId, StrategyAllocation
    constraints.py             Python mirror of validate_proposal
    agents/                    base, rule_based, mean_variance, llm
    market/                    MarketView, snapshot loading, data loaders
    backtest/engine.py         Point-in-time simulation loop
  scripts/                     Reproduction scripts, listed above
  tests/                       pytest suite
  data/
    snapshots/2026-07-13/      Frozen market data and manifest
    results/                   Per-run parquet output
    figures/                   PNG and PDF figures
    llm_cache/                 Cached Agent 3 responses

fixtures/constraint_cases.json Shared conformance cases for both languages
Anchor.toml, Cargo.toml, rust-toolchain.toml
```

## Architecture

Two halves communicate through a single proposal interface.

**Off-chain agents** produce an allocation across adapters in basis points,
using only information available at the decision date. Three agents of rising
sophistication are compared:

- Agent 1, rule-based, allocating in proportion to observed yield.
- Agent 2, mean-variance optimisation with Bayesian shrinkage on the drift
  estimate, solved with cvxpy.
- Agent 3, an LLM agent using Claude Sonnet, run in two configurations: one
  told nothing about the vault's limits, one told the per-strategy cap in the
  prompt.

**The on-chain program** is the sole authority over fund movement. Every
proposal passes through `validate_proposal` before any adapter is called, and
validation and dispatch occur in one transaction, so a failing check reverts the
whole rebalance and leaves the vault unchanged. The program is agent-agnostic:
nothing in it depends on which agent produced the proposal.

Two adapters are configured: Jupiter Lend USDC (lending) and JitoSOL via Jito
(liquid staking).

## Constraints enforced on chain

| Rule | Error | Code | Predicate over |
|------|-------|------|----------------|
| Weights sum to exactly 10000 bps | `InvalidWeightSum` | 6005 | Submitted weights |
| No adapter exceeds the per-strategy cap | `PerStrategyCapBreached` | 6001 | Submitted weights |
| Total across adapters within total cap | `TotalCapBreached` | 6002 | Submitted weights |
| Each adapter appears at most once | `DuplicateAdapter` | 6010 | Submitted weights |
| Movement per adapter within delta limit | `RebalanceDeltaExceeded` | 6003 | Submitted weights and caller-supplied current weights |
| CPI target is whitelisted | `ProgramNotWhitelisted` | 6004 | Submitted program keys |

The first four are predicates over the proposal alone, so the monitor decides
them from data it fully controls. The delta rule is the exception and is
discussed under Limitations.


## Verification

Verification rests on three layers, none of which depends on a deployed
network.

**Unit tests on the constraint module.** `validate_proposal` is exercised
directly against the `SafetyConstraints` struct, with one test per violation
class plus an accepting case. Run with `cargo test`.

**LiteSVM integration tests.** The compiled program is loaded into an in-process
Solana VM and driven through real transactions, so instruction encoding, PDA
derivation, account constraints and revert behaviour are exercised as the
runtime would exercise them. This is the basis for the claim that enforcement is
atomic: a rejected proposal fails the transaction, and no adapter dispatch
occurs.

**Cross-language conformance.** `fixtures/constraint_cases.json` holds a shared
set of proposals and expected verdicts. `constraint_parity.rs` and
`test_constraint_parity.py` read the same file, so the Python mirror used by the
backtest and the Rust implementation that would run on chain are checked against
one common specification rather than against each other informally.


## Results summary

Against the frozen snapshot, at a per-strategy cap of 6000 bps over 136 decision
dates:

- Agent 2 breaches the cap on every date. The breach curve is flat at 100
  percent for every cap value up to 9575 bps, so the result is not an artefact
  of the cap chosen. Mean-variance optimisation on price covariance cannot
  represent protocol or depeg risk, so it concentrates in the leg whose price
  series looks riskless.
- Agent 3, given no constraint information, also breaches on every date. It is
  told about depeg and smart contract risk in plain language, so the breach is
  not an information failure.
- Agent 3, given the cap in its prompt, breaches on no date. On five dates,
  however, it does not clip its unconstrained preference to the boundary. It
  inverts, preferring the other adapter outright, at a cost of roughly two
  percentage points.

That last finding is the primary empirical contribution. Constraint disclosure
perturbs the agent's objective rather than projecting its answer onto the
feasible set, which is precisely the behaviour that makes self-imposed
constraints an unsafe substitute for enforced ones.

## Scope and limitations

**No mainnet or devnet deployment.** The program is verified under LiteSVM and
by unit test, not by deployment. Nothing here has custody of real funds.

**Adapters are stubs.** Both adapters log via `msg!` rather than performing a
real CPI into Jupiter Lend or Jito. The constraint layer is the object of study,
and stubbing keeps the tests independent of external protocol uptime. Connecting
a live adapter is identified as future work, not as completed work.

**The delta rule is self-attested.** The caller supplies `current_weight_bps`
alongside the proposal, so the monitor computes the delta from an operand the
mediated party controls. An agent that misreports its current position can
satisfy the check while moving further than the rule permits. The other rules
were upgraded to predicates over vault state; this one was not. It is retained
and documented rather than removed, because it is the concrete instance of the
general point the dissertation makes: mediating an action is not the same as
mediating a decision when the mediated party supplies the inputs.

**A latent integer overflow exists and is unreachable.** The total-cap check
sums weights into a `u16` after an exact-sum check has already run over `u32`.
The earlier check dominates, so no input reaches the narrower sum in a state
that could overflow it. It is a defect in defence in depth, not an exploitable
vulnerability.

**The Python mirror is not the on-chain implementation.** It has two documented
divergences: a negative-weight check with no on-chain counterpart, since weights
are `u16`, and an adapter allowlist that approximates a whitelist of CPI target
program keys it has no representation of. The conformance fixture bounds the
divergence but does not eliminate it.

**Performance figures are descriptive only.** The evaluation window is a single
market regime. Cumulative return, Sharpe and Sortino are reported to
characterise what happened, not to argue that any agent allocates better. The
enforcement-cost comparison between an agent's enforced and unenforced arms is
the meaningful contrast; cross-agent return comparisons are not.

**Two constraint composition conflicts exist and are documented.** Exact-sum
combined with a total cap below 10000 bps admits no allocation at all. Exact-sum
combined with a delta limit below 10000/n admits no first rebalance away from a
flat zero position, even though steady-state adjustment remains legal.
`ConstraintConfig.is_reachable` detects the first and cannot express the second.

## Data provenance

APY history from DeFiLlama, daily price history from CoinGecko, both fetched on
13 July 2026 and frozen thereafter. `offchain/data/snapshots/2026-07-13/manifest.json`
records the pool identifiers, row counts and date ranges for each series, so the
provenance of every reported figure is traceable to a specific fetch.

Returns are stored backward-looking: `returns.loc[d]` is the return earned from
`d-1` to `d`. `offchain/tests/test_returns_convention.py` asserts this, because
the off-by-one that a forward-looking frame would introduce raises no error and
is invisible in the output.

No-lookahead is enforced structurally rather than by convention. `MarketView` is
immutable and is constructed per decision date from a truncated history, and the
backtest engine rejects any proposal whose `as_of` does not match the date being
simulated.

## Use of AI tools

Claude Sonnet is used as an object of study in Agent 3, invoked through the
Anthropic API. Its role, prompts and caching are described above and in the
dissertation methodology chapter. Separately, AI assistance was used during
development and drafting; that use is declared in full in the dissertation in
the form required by the programme handbook. This README is not a substitute for
that declaration.

"""Exploratory live-data smoke test. Hits CoinGecko + DeFiLlama. Not a unit test."""

from bounded_vault.market.loaders import fetch_current_apy, fetch_daily_returns


def main() -> None:
    # yield fetch first, it is the fragile one.
    # project=None means: any pool matching chain+symbol, highest TVL wins.
    print("lending apy:", fetch_current_apy(chain="Solana", symbol="USDC", project=None))
    print("staking apy:", fetch_current_apy(chain="Solana", symbol="SOL", project=None))

    # then the price history
    print("usdc returns:")
    print(fetch_daily_returns("usd-coin").tail())
    print("sol returns:")
    print(fetch_daily_returns("solana").tail())


if __name__ == "__main__":
    main()
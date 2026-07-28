# Market data snapshots

Frozen point-in-time captures of DeFiLlama APY history and CoinGecko price
returns, keyed by the ISO date of fetch.

Regenerate a snapshot:

    python scripts/fetch_snapshot.py

Each snapshot directory contains one parquet per adapter for APY history,
one parquet per underlying asset for price returns, and a manifest.json
recording pool IDs, row counts, and date ranges.

Parquet files are gitignored (regeneratable from live APIs given the manifest).
Manifests are committed as reproducibility evidence.

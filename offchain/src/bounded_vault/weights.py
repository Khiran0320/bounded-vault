def to_basis_points(weights: list[float]) -> list[int]:
    """Convert fractional weights summing to ~1.0 into integer basis points
    that sum to exactly 10_000, using largest-remainder rounding."""
    raw = [w * 10_000 for w in weights]
    floored = [int(x) for x in raw]
    leftover = 10_000 - sum(floored)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floored[i], reverse=True)
    for i in order[:leftover]:
        floored[i] += 1
    return floored
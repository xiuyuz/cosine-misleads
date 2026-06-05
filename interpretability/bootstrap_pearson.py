"""Bootstrap CIs for the cross-variant Pearson correlations in Table 1.

Each variant contributes one (probe_a, V*B, MMVP, BLINK) data point.
We bootstrap by resampling each variant's V*B from a binomial sampling
model over its n=191 V*Bench questions, then recomputing the cross-variant
Pearson on each replicate (B=5000 by default).

Usage:
    python interpretability/bootstrap_pearson.py
"""
import numpy as np


# Snapshot of Table values from the paper, primary-5 variants.
VARIANT_NAMES   = ["LVR", "N-LVR", "D-LVR", "P-LVR-2", "P-LVR-3"]
PROBE_A         = np.array([69.1, 66.0, 64.9, 50.2, 48.7])
PROBE_B         = np.array([32.5, 41.9, 33.0, 35.6, 34.6])
COSINE          = np.array([0.555, 0.556, 0.464, 0.777, 0.769])
VSB_S8          = np.array([70.2, 71.7, 69.6, 57.1, 57.1])
MMVP            = np.array([49.7, 50.0, 49.3, 48.7, 48.0])
BLINK           = np.array([53.4, 52.9, 51.4, 47.3, 48.5])
TRUNC_DELTA     = np.array([-2.6, -2.6, -1.1,  0.0, +2.1])
NOISE01_DELTA   = np.array([-1.6, -0.5,  0.0, +2.1, +2.6])
SWAP_DELTA      = np.array([+1.6, -0.5, +2.1, +2.1, +2.6])
GAP_G           = PROBE_A - PROBE_B

N_VSTAR, N_MMVP, N_BLINK = 191, 300, 697


def _pearson(x, y):
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def bootstrap_pearson(x, y, n_bench, B=5000, seed=42):
    """Bootstrap CI for r(x, y_bootstrap) where y is a percent accuracy
    over n_bench items per variant. Returns (point, ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y)
    p = y / 100.0
    rs = np.empty(B)
    for b in range(B):
        yb = rng.binomial(n_bench, p) / n_bench * 100.0
        rs[b] = np.corrcoef(x, yb)[0, 1]
    return _pearson(x, y), float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def main():
    # Bootstrap CIs make sense only when the bench-axis is a 0-100 accuracy
    # (we resample from Binomial(n, p)). For corruption-Δ targets we report
    # the point estimate only.
    boot_rows = [
        ("probe-(a) vs V*B",  PROBE_A,       VSB_S8, N_VSTAR),
        ("probe-(a) vs MMVP", PROBE_A,       MMVP,   N_MMVP),
        ("probe-(a) vs BLINK",PROBE_A,       BLINK,  N_BLINK),
        ("cosine vs V*B",     COSINE,        VSB_S8, N_VSTAR),
        ("trunc Δ vs V*B",    TRUNC_DELTA,   VSB_S8, N_VSTAR),
        ("swap Δ vs V*B",     SWAP_DELTA,    VSB_S8, N_VSTAR),
        ("G vs V*B",          GAP_G,         VSB_S8, N_VSTAR),
    ]
    point_rows = [
        ("G vs σ=.1 Δ",       GAP_G, NOISE01_DELTA),
        ("G vs trunc Δ",      GAP_G, TRUNC_DELTA),
    ]
    print(f"{'measure':24s}  point     95% CI")
    print("-" * 52)
    for name, x, y, n in boot_rows:
        pt, lo, hi = bootstrap_pearson(x, y, n)
        print(f"{name:24s}  {pt:+.3f}   [{lo:+.3f}, {hi:+.3f}]")
    print()
    for name, x, y in point_rows:
        print(f"{name:24s}  {_pearson(x, y):+.3f}")


if __name__ == "__main__":
    main()

"""Small NumPy-only statistics helpers shared by the analysis scripts."""
import numpy as np


def paired_bootstrap(diff, n_boot=10000, seed=0, ci=95.0):
    """Bootstrap CI of the mean of paired differences (NaNs dropped).

    `diff` is one value per paired unit (a split for Option A, a test shape for Option B).
    """
    d = np.asarray(diff, dtype=float)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0, wins=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), n=int(d.size), wins=int((d > 0).sum()))


def fmt_ci(r, scale=100.0, nd=2):
    """'+0.52 [-0.10, +1.12]' (in points when scale=100)."""
    if r["n"] == 0:
        return "n/a"
    f = lambda v: f"{v * scale:+.{nd}f}"
    star = "*" if (r["lo"] > 0 or r["hi"] < 0) else ""
    return f"{f(r['mean'])} [{f(r['lo'])}, {f(r['hi'])}]{star}"


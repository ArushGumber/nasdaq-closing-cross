import numpy as np
from scipy import stats

from .config import ALPHA


def interval_comparison(x, bound):
    """Three 95% intervals for a mean: CLT (t), Chebyshev (finite variance), Hoeffding (|x| <= bound)."""
    x = np.asarray(x, dtype="float64")
    n, mean, sd = len(x), x.mean(), x.std(ddof=1)
    t_half = stats.t.ppf(1 - ALPHA / 2, n - 1) * sd / np.sqrt(n)
    cheb_half = sd / np.sqrt(n * ALPHA)
    hoeff_half = 2 * bound * np.sqrt(np.log(2 / ALPHA) / (2 * n))
    rows = {"t": t_half, "chebyshev": cheb_half, "hoeffding": hoeff_half}
    return {
        "n": n, "mean": float(mean), "sd": float(sd), "bound": float(bound),
        "intervals": {k: {"low": float(mean - h), "high": float(mean + h), "half_width": float(h),
                          "width_vs_t": float(h / t_half)} for k, h in rows.items()},
    }


def tail_shares(x):
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    z = np.abs(x - x.mean()) / x.std()
    out = {}
    for k in (2, 3, 4):
        out[str(k)] = {"observed": float((z > k).mean()), "normal": float(2 * stats.norm.sf(k)),
                       "chebyshev_max": float(1 / k**2)}
    return out

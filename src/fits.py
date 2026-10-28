import numpy as np
from scipy import stats

BINS = 20


def gof(x, dist, n_params):
    edges = dist.ppf(np.linspace(0, 1, BINS + 1))
    observed = np.histogram(x, bins=edges)[0]
    expected = len(x) / BINS
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    dof = BINS - 1 - n_params
    return {"chi2": chi2, "df": dof, "p": float(stats.chi2.sf(chi2, dof))}


def target_fits(x):
    x = np.asarray(x, dtype="float64")
    mu, sd = stats.norm.fit(x)
    nu, loc, scale = stats.t.fit(x)
    normal = stats.norm(mu, sd)
    student = stats.t(nu, loc, scale)
    return {
        "normal": {"mean": float(mu), "sd": float(sd), "loglik": float(normal.logpdf(x).sum()),
                   "gof": gof(x, normal, 2)},
        "student_t": {"df": float(nu), "loc": float(loc), "scale": float(scale),
                      "loglik": float(student.logpdf(x).sum()), "gof": gof(x, student, 3)},
        "loglik_gain_per_row": float((student.logpdf(x).sum() - normal.logpdf(x).sum()) / len(x)),
    }


def side_flips(df):
    ordered = df.sort_values(["stock_id", "date_id", "seconds_in_bucket"])
    flag = ordered["imbalance_buy_sell_flag"]
    changed = flag.ne(flag.groupby([ordered["stock_id"], ordered["date_id"]]).shift()).astype(int)
    first = ordered.groupby(["stock_id", "date_id"]).cumcount() == 0
    counts = changed[~first].groupby([ordered["stock_id"], ordered["date_id"]]).sum().to_numpy()
    lam = counts.mean()
    k_max = 0
    while len(counts) * stats.poisson.sf(k_max, lam) >= 5:
        k_max += 1
    observed = np.array([np.sum(counts == k) for k in range(k_max)] + [np.sum(counts >= k_max)])
    probs = np.append(stats.poisson.pmf(np.arange(k_max), lam), stats.poisson.sf(k_max - 1, lam))
    expected = len(counts) * probs
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    dof = len(observed) - 1 - 1
    return {"stock_days": int(len(counts)), "mean": float(lam), "variance": float(counts.var(ddof=1)),
            "dispersion": float(counts.var(ddof=1) / lam), "zero_share": float((counts == 0).mean()),
            "poisson_zero_share": float(stats.poisson.pmf(0, lam)), "chi2": chi2, "df": dof,
            "p": float(stats.chi2.sf(chi2, dof)), "bins": int(len(observed)),
            "observed": observed.tolist(), "expected": expected.tolist()}

import numpy as np
import pandas as pd
from scipy import stats

from .config import TEST_SECONDS

DESCRIBE = ["target", "signed_imbalance", "imbalance_size", "matched_size", "spread_bps", "near_gap_bps"]


def describe(df):
    out = {}
    for c in DESCRIBE:
        x = df[c].astype("float64").dropna()
        out[c] = {"n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()), "sd": float(x.std()),
                  "q1": float(x.quantile(0.25)), "q3": float(x.quantile(0.75)),
                  "skew": float(stats.skew(x)), "kurtosis": float(stats.kurtosis(x))}
    return out


def normal_gof(x, bins=20):
    """Chi-square goodness of fit of x to a normal with its own mean and sd, on equiprobable bins."""
    x = np.asarray(x, dtype="float64")
    edges = stats.norm.ppf(np.linspace(0, 1, bins + 1), loc=x.mean(), scale=x.std(ddof=1))
    observed = np.histogram(x, bins=edges)[0]
    expected = np.full(bins, len(x) / bins)
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    dof = bins - 1 - 2
    return {"chi2": chi2, "df": dof, "p": float(stats.chi2.sf(chi2, dof)), "n": int(len(x)), "bins": bins,
            "centre_share": float(observed[bins // 2 - 1: bins // 2 + 1].sum() / len(x)),
            "centre_expected": 2 / bins}


def autocorrelation(df, max_lag=12):
    ordered = df.sort_values(["stock_id", "date_id", "seconds_in_bucket"])
    y = ordered["target"].astype("float64")
    groups = ordered.groupby(["stock_id", "date_id"])["target"]
    return {lag: float(y.corr(groups.shift(lag).astype("float64"))) if lag else 1.0
            for lag in range(max_lag + 1)}


def by_second(df):
    g = df.groupby("seconds_in_bucket")
    out = pd.DataFrame({
        "abs_target": g["target"].apply(lambda t: t.abs().mean()),
        "median_spread_bps": g["spread_bps"].median(),
        "flag_share": g["imbalance_buy_sell_flag"].apply(lambda f: (f != 0).mean()),
        "median_abs_imbalance": g["signed_imbalance"].apply(lambda v: v.abs().median()),
    })
    return out


def decile_profile(df, n_bins=10):
    s = df[df["seconds_in_bucket"].isin(TEST_SECONDS)].copy()
    s["bin"] = pd.qcut(s["signed_imbalance"].rank(method="first"), n_bins, labels=False)
    day_means = s.groupby(["post", "bin", "date_id"])["target_w"].mean()
    g = day_means.groupby(["post", "bin"])
    out = pd.DataFrame({"mean": g.mean(), "se": g.std() / np.sqrt(g.count()), "days": g.count()}).reset_index()
    centres = s.groupby("bin")["signed_imbalance"].median()
    out["centre"] = out["bin"].map(centres)
    return out


def flag_shares(df):
    shares = df["imbalance_buy_sell_flag"].value_counts(normalize=True)
    return {str(k): float(v) for k, v in shares.items()}


def coverage(present):
    missing = ~present
    return {"possible_stock_days": int(present.size), "present": int(present.to_numpy().sum()),
            "missing": int(missing.to_numpy().sum()),
            "stocks_with_gaps": int(missing.any(axis=1).sum()),
            "max_missing_days_one_stock": int(missing.sum(axis=1).max())}

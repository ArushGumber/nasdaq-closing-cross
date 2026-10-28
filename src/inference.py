import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .config import (ALPHA, BOUNDARY, MATTERS_CRAMER_V, MATTERS_DELTA_IC, MATTERS_HIT_RATE,
                     MATTERS_VAR_RATIO, N_BOOT, SEED, T2_SECOND, TEST_SECONDS)


def test_sample(df):
    return df[df["seconds_in_bucket"].isin(TEST_SECONDS)].copy()


def smallest_difference(df):
    return float(df["half_spread_bps"].median())


def verdict(ci, low, high=None):
    """'matters' if the whole CI lies beyond the threshold, 'does not matter' if inside it."""
    if high is None:
        if ci[0] > low:
            return "matters"
        if ci[1] < low:
            return "does not matter"
        return "inconclusive"
    if ci[0] > high or ci[1] < low:
        return "matters"
    if ci[0] >= low and ci[1] <= high:
        return "does not matter"
    return "inconclusive"


def t_summary(x, alternative="two-sided", null=0.0):
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    n = len(x)
    res = stats.ttest_1samp(x, null, alternative=alternative)
    se = x.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(1 - ALPHA / 2, n - 1) * se
    return {"n": n, "estimate": float(x.mean()), "se": float(se), "t": float(res.statistic),
            "df": n - 1, "p": float(res.pvalue), "ci": [float(x.mean() - half), float(x.mean() + half)]}


def bootstrap_mean_ci(x, rng):
    x = np.asarray(x, dtype="float64")
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def naive_z(x, alternative="greater"):
    x = np.asarray(x, dtype="float64")
    z = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    p = stats.norm.sf(z) if alternative == "greater" else 2 * stats.norm.sf(abs(z))
    return {"n": len(x), "z": float(z), "p": float(p)}


def test1_direction(s, sdm, full, rng):
    flagged = s[s["imbalance_buy_sell_flag"] != 0]
    edge_w = flagged["imbalance_buy_sell_flag"] * flagged["target_w"]
    edge_raw = flagged["imbalance_buy_sell_flag"] * flagged["target"]
    daily = edge_w.groupby(flagged["date_id"]).mean()
    daily_raw = edge_raw.groupby(flagged["date_id"]).mean()
    primary = t_summary(daily, alternative="greater")
    primary["verdict"] = verdict(primary["ci"], sdm)

    buy = flagged.loc[flagged["imbalance_buy_sell_flag"] == 1, "target_w"]
    sell = flagged.loc[flagged["imbalance_buy_sell_flag"] == -1, "target_w"]
    welch = stats.ttest_ind(buy, sell, equal_var=False, alternative="greater")
    pooled = stats.ttest_ind(buy, sell, equal_var=True, alternative="greater")
    var_f = float(buy.var() / sell.var())

    every = full[full["imbalance_buy_sell_flag"] != 0]
    overlapping = naive_z(every["imbalance_buy_sell_flag"] * every["target_w"])

    return {
        "primary": primary,
        "raw_target": t_summary(daily_raw, alternative="greater"),
        "bootstrap_ci": bootstrap_mean_ci(daily.to_numpy(), rng),
        "naive_row_z": naive_z(edge_w),
        "naive_overlapping_z": overlapping,
        "welch_buy_vs_sell": {"t": float(welch.statistic), "p": float(welch.pvalue),
                              "diff": float(buy.mean() - sell.mean()), "n_buy": len(buy), "n_sell": len(sell)},
        "pooled_buy_vs_sell": {"t": float(pooled.statistic), "p": float(pooled.pvalue)},
        "variance_ratio_buy_sell": var_f,
        "positive_days": int((daily > 0).sum()),
        "daily": daily.to_numpy().tolist(),
    }


def cramers_v(chi2, n, shape):
    return float(np.sqrt(chi2 / (n * (min(shape) - 1))))


def test2_independence(df, s):
    first = df[(df["seconds_in_bucket"] == T2_SECOND) & (df["target"] != 0)]
    table = pd.crosstab(first["imbalance_buy_sell_flag"], first["target"] > 0)
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    g, g_p, _, _ = stats.chi2_contingency(table, correction=False, lambda_="log-likelihood")
    n = int(table.to_numpy().sum())
    v = cramers_v(chi2, n, table.shape)

    up = first["target"] > 0
    buy, sell = first["imbalance_buy_sell_flag"] == 1, first["imbalance_buy_sell_flag"] == -1
    p_buy, p_sell = up[buy].mean(), up[sell].mean()
    n_buy, n_sell = int(buy.sum()), int(sell.sum())
    pooled = up[buy | sell].mean()
    z = (p_buy - p_sell) / np.sqrt(pooled * (1 - pooled) * (1 / n_buy + 1 / n_sell))
    se = np.sqrt(p_buy * (1 - p_buy) / n_buy + p_sell * (1 - p_sell) / n_sell)
    diff = p_buy - p_sell
    zc = stats.norm.ppf(1 - ALPHA / 2)

    daily = first.assign(up=up).groupby("date_id").apply(
        lambda d: d.loc[d["imbalance_buy_sell_flag"] == 1, "up"].mean()
        - d.loc[d["imbalance_buy_sell_flag"] == -1, "up"].mean(), include_groups=False)

    rows = s[s["target"] != 0]
    all_table = pd.crosstab(rows["imbalance_buy_sell_flag"], rows["target"] > 0)
    all_chi2, all_p, _, _ = stats.chi2_contingency(all_table, correction=False)

    return {
        "primary": {"chi2": float(chi2), "df": int(dof), "p": float(p), "n": n, "cramers_v": v,
                    "min_expected": float(expected.min()),
                    "verdict": "matters" if v >= MATTERS_CRAMER_V else "does not matter"},
        "table": {str(k): {str(c): int(x) for c, x in row.items()} for k, row in table.iterrows()},
        "up_share": {"buy": float(p_buy), "sell": float(p_sell), "none": float(up[first["imbalance_buy_sell_flag"] == 0].mean())},
        "two_proportion": {"z": float(z), "p": float(stats.norm.sf(z)), "diff": float(diff),
                           "ci": [float(diff - zc * se), float(diff + zc * se)]},
        "g_test": {"g": float(g), "p": float(g_p)},
        "day_level": t_summary(daily.dropna(), alternative="greater"),
        "all_rows": {"chi2": float(all_chi2), "p": float(all_p), "n": int(all_table.to_numpy().sum()),
                     "cramers_v": cramers_v(all_chi2, int(all_table.to_numpy().sum()), all_table.shape)},
    }


def daily_ic(s):
    ranked = s[["date_id", "post"]].copy()
    keys = [s["date_id"], s["post"]]
    ranked["x"] = s.groupby(keys)["signed_imbalance"].rank()
    ranked["y"] = s.groupby(keys)["target"].rank()
    ic = ranked.groupby(["date_id", "post"]).apply(lambda d: d["x"].corr(d["y"]), include_groups=False)
    return ic.unstack("post").rename(columns={0: "pre", 1: "post"})


def fisher_z(r1, n1, r2, n2):
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return float(z), float(2 * stats.norm.sf(abs(z)))


def corr_t(r, n):
    t = r * np.sqrt((n - 2) / (1 - r**2))
    return {"r": float(r), "n": int(n), "t": float(t), "df": int(n - 2), "p": float(2 * stats.t.sf(abs(t), n - 2))}


def test3_boundary(s):
    ic = daily_ic(s).dropna()
    delta = ic["post"] - ic["pre"]
    primary = t_summary(delta)
    primary["verdict"] = verdict(primary["ci"], -MATTERS_DELTA_IC, MATTERS_DELTA_IC)

    pooled = {}
    for phase, name in [(0, "pre"), (1, "post")]:
        part = s[s["post"] == phase]
        r = stats.spearmanr(part["signed_imbalance"], part["target"]).statistic
        pooled[name] = corr_t(r, len(part))
    fz, fz_p = fisher_z(pooled["post"]["r"], pooled["post"]["n"], pooled["pre"]["r"], pooled["pre"]["n"])
    sign = stats.binomtest(int((delta > 0).sum()), len(delta), 0.5)
    wilcoxon = stats.wilcoxon(delta)

    return {
        "primary": primary,
        "ic_pre": t_summary(ic["pre"]),
        "ic_post": t_summary(ic["post"]),
        "pooled_correlation": pooled,
        "fisher_z": {"z": fz, "p": fz_p},
        "sign_test": {"positive_days": int((delta > 0).sum()), "n": len(delta), "p": float(sign.pvalue)},
        "wilcoxon": {"w": float(wilcoxon.statistic), "p": float(wilcoxon.pvalue)},
        "daily_pre": ic["pre"].tolist(),
        "daily_post": ic["post"].tolist(),
    }


def f_test(post, pre):
    f = post.var(ddof=1) / pre.var(ddof=1)
    d1, d2 = len(post) - 1, len(pre) - 1
    p = 2 * min(stats.f.cdf(f, d1, d2), stats.f.sf(f, d1, d2))
    ci = [f / stats.f.ppf(1 - ALPHA / 2, d1, d2), f / stats.f.ppf(ALPHA / 2, d1, d2)]
    return {"f": float(f), "df1": d1, "df2": d2, "p": float(p), "ci": [float(c) for c in ci]}


def test4_variance(s):
    pre = s.loc[s["post"] == 0, "target"].astype("float64")
    post = s.loc[s["post"] == 1, "target"].astype("float64")
    primary = f_test(post, pre)
    primary["verdict"] = verdict(primary["ci"], *MATTERS_VAR_RATIO)
    bf = stats.levene(post, pre, center="median")

    by_day = s.groupby(["date_id", "post"])["target"].var().unstack()
    log_ratio = np.log(by_day[1] / by_day[0]).dropna()
    day = t_summary(log_ratio)
    day["ratio"] = float(np.exp(day["estimate"]))
    day["ratio_ci"] = [float(np.exp(c)) for c in day["ci"]]

    w_pre = s.loc[s["post"] == 0, "target_w"].astype("float64")
    w_post = s.loc[s["post"] == 1, "target_w"].astype("float64")
    return {
        "primary": primary,
        "sd_pre": float(pre.std()), "sd_post": float(post.std()),
        "kurtosis_pre": float(stats.kurtosis(pre)), "kurtosis_post": float(stats.kurtosis(post)),
        "brown_forsythe": {"w": float(bf.statistic), "p": float(bf.pvalue)},
        "day_level_log_ratio": day,
        "winsorised_f": f_test(w_post, w_pre),
    }


def test5_hit_rate(s):
    flagged = s[s["imbalance_buy_sell_flag"] != 0]
    gross = flagged["imbalance_buy_sell_flag"] * flagged["target"]
    net = gross - flagged["half_spread_bps"]
    win = (net > 0).to_numpy()
    n, k = len(win), int(win.sum())
    p_hat = k / n
    z = (p_hat - 0.5) / np.sqrt(0.25 / n)
    zc = stats.norm.ppf(1 - ALPHA / 2)
    half = zc * np.sqrt(p_hat * (1 - p_hat) / n)
    ci = [p_hat - half, p_hat + half]
    exact = stats.binomtest(k, n, 0.5, alternative="greater")
    daily = pd.Series(win, index=flagged.index).groupby(flagged["date_id"]).mean()
    return {
        "primary": {"n": n, "wins": k, "estimate": float(p_hat), "z": float(z), "p": float(stats.norm.sf(z)),
                    "ci": [float(c) for c in ci], "verdict": verdict(ci, MATTERS_HIT_RATE)},
        "gross_hit_rate": float((gross > 0).mean()),
        "mean_net_bps": float(net.mean()),
        "exact_binomial_p": float(exact.pvalue),
        "day_level": t_summary(daily, alternative="greater", null=0.5),
    }


def holm(results):
    names = list(results)
    pvals = [results[k]["primary"]["p"] for k in names]
    reject, adjusted, _, _ = multipletests(pvals, alpha=ALPHA, method="holm")
    return {k: {"p": p, "p_holm": float(a), "reject_holm": bool(r)}
            for k, p, a, r in zip(names, pvals, adjusted, reject)}


def holm_family(pvals):
    names = list(pvals)
    reject, adjusted, _, _ = multipletests([pvals[k] for k in names], alpha=ALPHA, method="holm")
    return {k: {"p": float(pvals[k]), "p_holm": float(a), "reject_holm": bool(r)}
            for k, a, r in zip(names, adjusted, reject)}


def run_tests(df):
    rng = np.random.default_rng(SEED)
    s = test_sample(df)
    sdm = smallest_difference(df)
    results = {
        "T1": test1_direction(s, sdm, df, rng),
        "T2": test2_independence(df, s),
        "T3": test3_boundary(s),
        "T4": test4_variance(s),
        "T5": test5_hit_rate(s),
    }
    for k in results:
        results[k]["primary"]["reject"] = results[k]["primary"]["p"] < ALPHA
    return {"sdm_bps": sdm, "n_test_rows": int(len(s)), "boundary_second": BOUNDARY,
            "tests": results, "holm": holm(results)}

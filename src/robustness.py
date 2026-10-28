import numpy as np
import pandas as pd

from .config import BOUNDARY, SEED, TEST_SECONDS
from .inference import daily_ic, t_summary

N_PERM_T1 = 2000
N_FLIP_T3 = 10000


def terciles(values):
    return pd.qcut(values, 3, labels=["low", "mid", "high"])


def subgroup_delta_ic(s, df):
    stock_size = terciles(df.groupby("stock_id")["matched_size"].median())
    stock_spread = terciles(df.groupby("stock_id")["spread_bps"].median())
    day_vol = terciles(df.groupby("date_id")["target"].apply(lambda t: t.abs().mean()))
    half = s["date_id"].max() // 2 + 1
    groups = {
        "first half of days": s["date_id"] < half,
        "second half of days": s["date_id"] >= half,
    }
    for label, by, key in [("size", stock_size, "stock_id"), ("spread", stock_spread, "stock_id"),
                           ("day volatility", day_vol, "date_id")]:
        for level in ["low", "mid", "high"]:
            groups[f"{label}: {level}"] = s[key].map(by).eq(level)
    out = {}
    for name, mask in groups.items():
        ic = daily_ic(s[mask]).dropna()
        res = t_summary(ic["post"] - ic["pre"])
        res["pre"] = float(ic["pre"].mean())
        res["post"] = float(ic["post"].mean())
        out[name] = res
    positive = sum(r["ci"][0] > 0 for r in out.values())
    status = "robust" if positive == len(out) else "partly robust" if positive >= len(out) - 2 else "not robust"
    return {"groups": out, "positive": positive, "total": len(out), "status": status}


def snapshot_ic(frame):
    keys = [frame["date_id"], frame["seconds_in_bucket"]]
    x = frame.groupby(keys)["signed_imbalance"].rank().to_numpy()
    y = frame.groupby(keys)["target"].rank().to_numpy()
    parts = pd.DataFrame({"date_id": frame["date_id"].to_numpy(), "sec": frame["seconds_in_bucket"].to_numpy(),
                          "x": x, "y": y, "xy": x * y, "xx": x * x, "yy": y * y})
    g = parts.groupby(["date_id", "sec"]).agg(n=("x", "size"), x=("x", "sum"), y=("y", "sum"), xy=("xy", "sum"),
                                              xx=("xx", "sum"), yy=("yy", "sum"))
    cov = g["xy"] - g["x"] * g["y"] / g["n"]
    var_x = g["xx"] - g["x"] ** 2 / g["n"]
    var_y = g["yy"] - g["y"] ** 2 / g["n"]
    return (cov / np.sqrt(var_x * var_y)).unstack("sec")


def placebo_boundaries(df):
    ic = snapshot_ic(df[df["seconds_in_bucket"].isin(TEST_SECONDS)])
    jumps = {}
    for b in TEST_SECONDS[1:]:
        jumps[int(b)] = t_summary((ic[b] - ic[b - 60]).dropna())
    real = jumps[BOUNDARY]
    others = [v for k, v in jumps.items() if k != BOUNDARY]
    largest = all(real["estimate"] > o["estimate"] for o in others)
    separated = all(real["ci"][0] > o["ci"][1] for o in others)
    status = "boundary-specific" if largest and separated else "largest but overlapping" if largest else "not largest"

    pre = ic[[c for c in ic.columns if c < BOUNDARY]].mean(axis=1)
    post = ic[[c for c in ic.columns if c >= BOUNDARY]].mean(axis=1)
    post_early = ic[[c for c in ic.columns if BOUNDARY <= c < TEST_SECONDS[-1]]].mean(axis=1)
    sensitivity = {"with_last_minute": t_summary(post - pre), "without_last_minute": t_summary(post_early - pre)}

    profile = snapshot_ic(df)
    n = profile.notna().sum()
    curve = {"seconds": [int(c) for c in profile.columns], "mean": profile.mean().tolist(),
             "se": (profile.std() / np.sqrt(n)).tolist()}
    return {"jumps": jumps, "status": status, "sensitivity": sensitivity, "profile": curve}


def permutation_t1(s):
    rng = np.random.default_rng(SEED)
    frame = s.sort_values(["date_id", "seconds_in_bucket"])
    group = frame.groupby(["date_id", "seconds_in_bucket"]).ngroup().to_numpy()
    day = frame["date_id"].to_numpy()
    flag = frame["imbalance_buy_sell_flag"].to_numpy().astype("float64")
    target = frame["target_w"].to_numpy().astype("float64")
    flagged_per_day = np.bincount(day, weights=(flag != 0).astype("float64"))
    valid = flagged_per_day > 0

    def statistic(f):
        return (np.bincount(day, weights=f * target)[valid] / flagged_per_day[valid]).mean()

    observed = statistic(flag)
    null = np.empty(N_PERM_T1)
    for i in range(N_PERM_T1):
        order = np.argsort(group + rng.random(len(group)), kind="stable")
        null[i] = statistic(flag[order])
    p = (1 + np.sum(null >= observed)) / (N_PERM_T1 + 1)
    return {"observed": float(observed), "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "null_max": float(null.max()), "p": float(p), "permutations": N_PERM_T1}


def sign_flip_t3(delta):
    rng = np.random.default_rng(SEED)
    delta = np.asarray(delta, dtype="float64")
    observed = abs(delta.mean())
    signs = rng.choice([-1.0, 1.0], size=(N_FLIP_T3, len(delta)))
    null = np.abs((signs * delta).mean(axis=1))
    p = (1 + np.sum(null >= observed)) / (N_FLIP_T3 + 1)
    return {"observed": float(delta.mean()), "null_max": float(null.max()), "p": float(p), "draws": N_FLIP_T3}


def mid_price_checks(frame):
    rows = frame[frame["sec"].isin(TEST_SECONDS) & frame["target_mid"].notna() & (frame["signed_imbalance"] != 0)]
    side = np.sign(rows["signed_imbalance"].to_numpy())
    out = {}
    for column in ["target", "target_mid"]:
        edge = side * rows[column].to_numpy()
        daily = pd.Series(edge).groupby(rows["date_id"].to_numpy()).mean()
        win = edge - rows["half_spread"].to_numpy() > 0
        p_hat = win.mean()
        out[column] = {"edge": t_summary(daily, alternative="greater"), "hit_rate": float(p_hat),
                       "z": float((p_hat - 0.5) / np.sqrt(0.25 / len(win))), "rows": int(len(rows))}
    out["edge_retained"] = out["target_mid"]["edge"]["estimate"] / out["target"]["edge"]["estimate"]
    return out

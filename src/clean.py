import numpy as np

from .config import BOUNDARY, CHEBYSHEV_K, WINSOR
from .load import KEYS

BOOK = ["imbalance_size", "reference_price", "matched_size", "bid_price", "ask_price", "wap"]
PRICES = ["reference_price", "far_price", "near_price", "bid_price", "ask_price", "wap"]


def snapshot(df):
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 2**20, 1),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_by_column": {c: int(n) for c, n in df.isna().sum().items() if n},
        "stock_days": int(df.groupby(["stock_id", "date_id"]).ngroups),
    }


def fence_counts(x):
    x = x[np.isfinite(x)]
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    mean, sd = x.mean(), x.std()
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    rules = {
        "iqr_1.5": (x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr),
        "iqr_3": (x < q1 - 3 * iqr) | (x > q3 + 3 * iqr),
        "mad_5": np.abs(x - med) > 5 * mad,
        "sigma_3": np.abs(x - mean) > 3 * sd,
        "chebyshev_10": np.abs(x - mean) > CHEBYSHEV_K * sd,
    }
    return {k: {"rows": int(m.sum()), "share": float(m.mean())} for k, m in rules.items()}


def chebyshev_outlier(x):
    mean, sd = np.nanmean(x), np.nanstd(x)
    return np.abs(x - mean) > CHEBYSHEV_K * sd


def clean(raw):
    df = raw.copy()
    log = []

    def record(step, issue, affected, action):
        log.append({"step": step, "issue": issue, "rows_affected": int(affected), "action": action,
                    "rows_after": int(len(df))})

    dup = df.duplicated(KEYS)
    df = df[~dup]
    record("duplicates", "repeated (stock, day, second) key", dup.sum(), "dropped")

    empty = df[BOOK].isna().all(axis=1)
    empty_days = df.loc[empty, ["stock_id", "date_id"]].drop_duplicates()
    target_gone = int(df.loc[empty, "target"].isna().sum())
    df = df[~empty]
    record("empty_book", f"no book on {len(empty_days)} stock-days; {target_gone} of these rows also lack the target",
           empty.sum(), "dropped")

    no_target = df["target"].isna()
    df = df[~no_target]
    record("missing_target", "response not observed outside empty books", no_target.sum(), "dropped")

    pre = df["seconds_in_bucket"] < BOUNDARY
    structural = int(df.loc[pre, ["far_price", "near_price"]].isna().sum().sum())
    record("structural_missing", "far and near price absent before 15:55 by exchange rule", pre.sum(),
           f"kept as missing ({structural} cells), phase flag added")

    far_dev = np.log(df["far_price"] / df["reference_price"]).to_numpy()
    fences = fence_counts(far_dev)
    far_out = chebyshev_outlier(far_dev)
    df.loc[far_out, "far_price"] = np.nan
    record("far_price_outlier", "far price more than 10 sd from the reference price (log scale)",
           far_out.sum(), "set to missing")

    lo, hi = df["target"].quantile(list(WINSOR))
    clipped = (df["target"] < lo) | (df["target"] > hi)
    df["target_w"] = df["target"].clip(lo, hi)
    record("winsorise", f"target beyond [{lo:.2f}, {hi:.2f}] bps", clipped.sum(),
           "clipped copy target_w; raw target kept")

    df = df.drop(columns=["row_id", "time_id"])
    df["post"] = (df["seconds_in_bucket"] >= BOUNDARY).astype("int8")
    df = downcast(df)
    record("dtypes", "64-bit storage for 8-bit and 16-bit quantities", len(df), "downcast")

    meta = {"winsor_bounds": [float(lo), float(hi)], "far_price_fences": fences}
    return df.reset_index(drop=True), log, meta


def downcast(df):
    out = df.copy()
    for c in ["stock_id", "seconds_in_bucket"]:
        out[c] = out[c].astype("int16")
    out["date_id"] = out["date_id"].astype("int16")
    out["imbalance_buy_sell_flag"] = out["imbalance_buy_sell_flag"].astype("int8")
    for c in ["imbalance_size", "matched_size", "bid_size", "ask_size", "target", "target_w"]:
        out[c] = out[c].astype("float32")
    return out


def panel_gaps(df, n_stocks=200, n_days=481):
    present = df.groupby(["stock_id", "date_id"]).size().unstack(fill_value=0) > 0
    present = present.reindex(index=range(n_stocks), columns=range(n_days), fill_value=False)
    return present


def encoded_shape(df):
    numeric = ["seconds_in_bucket", "imbalance_size", "reference_price", "matched_size", "far_price",
               "near_price", "bid_price", "bid_size", "ask_price", "ask_size", "wap", "target",
               "date_id"]
    one_hot = {
        "stock_id": int(df["stock_id"].nunique()),
        "imbalance_buy_sell_flag": int(df["imbalance_buy_sell_flag"].nunique()),
        "phase": 2,
    }
    engineered = ["signed_imbalance", "spread_bps", "near_gap_bps", "far_gap_bps", "target_w"]
    cols = len(numeric) + sum(one_hot.values()) + len(engineered)
    return {"rows": int(len(df)), "columns": cols, "numeric": len(numeric), "one_hot": one_hot,
            "engineered": len(engineered), "cells": int(len(df)) * cols}

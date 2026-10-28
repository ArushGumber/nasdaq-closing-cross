import numpy as np
import pandas as pd

HORIZON_STEPS = 6


def add_features(df):
    out = df.copy()
    flag = out["imbalance_buy_sell_flag"].astype("float64")
    out["signed_imbalance"] = flag * out["imbalance_size"] / out["matched_size"]
    out["spread_bps"] = (out["ask_price"] - out["bid_price"]) / out["wap"] * 1e4
    out["half_spread_bps"] = out["spread_bps"] / 2
    out["near_gap_bps"] = (out["near_price"] / out["reference_price"] - 1) * 1e4
    out["far_gap_bps"] = (out["far_price"] / out["reference_price"] - 1) * 1e4
    return out


def stock_returns(df):
    ordered = df.sort_values(["stock_id", "date_id", "seconds_in_bucket"])
    future = ordered.groupby(["stock_id", "date_id"])["wap"].shift(-HORIZON_STEPS)
    ret = (future / ordered["wap"] - 1) * 1e4
    return ret.reindex(df.index)


def reconstruct_index(df):
    """Recover the hidden index from target = stock return - index return.

    If the target is defined as stated, the implied index return is identical for every stock
    at the same timestamp, and it is a fixed weighted sum of the stock returns.
    """
    ret = stock_returns(df)
    frame = pd.DataFrame({"date_id": df["date_id"], "sec": df["seconds_in_bucket"],
                          "stock_id": df["stock_id"], "ret": ret,
                          "implied": ret - df["target"].astype("float64")}).dropna()
    by_time = frame.groupby(["date_id", "sec"])["implied"]
    spread_within = by_time.std()

    wide = frame.pivot_table(index=["date_id", "sec"], columns="stock_id", values="ret")
    target_index = by_time.median().reindex(wide.index)
    complete = wide.notna().all(axis=1)
    x = wide[complete].to_numpy()
    y = target_index[complete].to_numpy()
    w, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ w
    r2 = 1 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)
    return {
        "timestamps": int(len(spread_within)),
        "median_within_time_sd_bps": float(spread_within.median()),
        "p99_within_time_sd_bps": float(spread_within.quantile(0.99)),
        "regression_timestamps": int(complete.sum()),
        "weights_sum": float(w.sum()),
        "weights_positive": int((w > 1e-4).sum()),
        "weights_max": float(w.max()),
        "r2": float(r2),
        "weights": w.tolist(),
    }

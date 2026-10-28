import pandas as pd

from .inference import t_summary

N_BINS = 10


def edge_frame(s):
    flagged = s[s["imbalance_buy_sell_flag"] != 0].copy()
    flagged["gross"] = flagged["imbalance_buy_sell_flag"] * flagged["target"].astype("float64")
    flagged["strength"] = flagged["signed_imbalance"].abs()
    flagged["bin"] = flagged.groupby("post")["strength"].transform(
        lambda v: pd.qcut(v.rank(method="first"), N_BINS, labels=False))
    return flagged


def edge_by_strength(flagged):
    rows = []
    for (post, b), g in flagged.groupby(["post", "bin"]):
        day = g.groupby("date_id")[["gross", "half_spread_bps"]].mean()
        rows.append({"post": int(post), "bin": int(b), "gross": float(day["gross"].mean()),
                     "gross_se": float(day["gross"].std() / len(day) ** 0.5),
                     "half_spread": float(day["half_spread_bps"].mean()),
                     "median_strength": float(g["strength"].median())})
    return rows


def breakeven(flagged):
    out = {}
    for post, name in [(0, "pre"), (1, "post")]:
        g = flagged[flagged["post"] == post]
        out[name] = {"gross": float(g["gross"].mean()), "median_half_spread": float(g["half_spread_bps"].median()),
                     "share_cheaper": float((g["half_spread_bps"] < g["gross"].mean()).mean())}
    return out


def strongest_post(flagged):
    top = flagged[(flagged["post"] == 1) & (flagged["bin"] == N_BINS - 1)]
    net = (top["gross"] - top["half_spread_bps"]).groupby(top["date_id"]).mean()
    res = t_summary(net, alternative="greater")
    res["gross"] = float(top["gross"].mean())
    res["half_spread"] = float(top["half_spread_bps"].mean())
    res["trades"] = int(len(top))
    res["threshold_strength"] = float(top["strength"].min())
    return res


def edge_by_second(df):
    flagged = df[df["imbalance_buy_sell_flag"] != 0]
    gross = flagged["imbalance_buy_sell_flag"] * flagged["target"].astype("float64")
    frame = flagged[["date_id", "seconds_in_bucket", "half_spread_bps"]].assign(gross=gross)
    day = frame.groupby(["seconds_in_bucket", "date_id"])[["gross", "half_spread_bps"]].mean()
    g = day.groupby("seconds_in_bucket")
    return {"seconds": [int(x) for x in g.groups], "gross": g["gross"].mean().tolist(),
            "gross_se": (g["gross"].std() / g["gross"].count() ** 0.5).tolist(),
            "half_spread": g["half_spread_bps"].mean().tolist()}


def final_minute(df, second=540):
    g = df[(df["seconds_in_bucket"] == second) & (df["imbalance_buy_sell_flag"] != 0)]
    gross = g["imbalance_buy_sell_flag"] * g["target"].astype("float64")
    net = gross - g["half_spread_bps"]
    daily = net.groupby(g["date_id"]).mean()
    half = g["date_id"].max() // 2 + 1
    out = t_summary(daily, alternative="greater")
    out.update({"gross": float(gross.mean()), "half_spread": float(g["half_spread_bps"].mean()),
                "hit_rate": float((net > 0).mean()), "trades": int(len(g)),
                "first_half": t_summary(daily[daily.index < half], alternative="greater"),
                "second_half": t_summary(daily[daily.index >= half], alternative="greater")})
    return out


def cost_analysis(df, s):
    flagged = edge_frame(s)
    return {"by_strength": edge_by_strength(flagged), "breakeven": breakeven(flagged),
            "strongest_post": strongest_post(flagged), "by_second": edge_by_second(df),
            "final_minute": final_minute(df)}

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn

from .config import TEST_SECONDS
from .inference import t_summary

TRAIN_DAYS = (0, 240)
VALID_DAYS = (241, 300)
TEST_DAYS = (301, 480)
SEEDS = (0, 1, 2, 3, 4)
MATTERS_GAIN = 0.01
N_FLIP = 10000

SNAPSHOT = ["signed_imbalance", "abs_ratio", "side_buy", "side_sell", "log_matched", "spread_bps", "near_gap",
            "far_gap", "near_missing", "far_missing", "ref_gap_bps", "depth_imbalance", "seconds", "post"]
PATH = ["ret_10", "ret_30", "ret_60", "d_imb_10", "d_imb_60"]
FEATURES = SNAPSHOT + PATH
C2ST_FEATURES = ["signed_imbalance", "abs_ratio", "side_buy", "side_sell", "log_matched", "spread_bps",
                 "ref_gap_bps", "depth_imbalance"] + PATH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_frame(df):
    f = df.sort_values(["stock_id", "date_id", "seconds_in_bucket"]).reset_index(drop=True)
    keys = [f["stock_id"], f["date_id"]]
    wap = f["wap"].astype("float64")
    imb = f["signed_imbalance"].astype("float64")
    out = pd.DataFrame({
        "stock_id": f["stock_id"], "date_id": f["date_id"], "sec": f["seconds_in_bucket"],
        "target": f["target"].astype("float64"), "target_w": f["target_w"].astype("float64"),
        "half_spread": f["half_spread_bps"].astype("float64"),
        "signed_imbalance": imb, "abs_ratio": imb.abs(),
        "side_buy": (f["imbalance_buy_sell_flag"] == 1).astype(float),
        "side_sell": (f["imbalance_buy_sell_flag"] == -1).astype(float),
        "log_matched": np.log1p(f["matched_size"].astype("float64")),
        "spread_bps": f["spread_bps"].astype("float64"),
        "near_gap": f["near_gap_bps"].fillna(0).clip(-500, 500),
        "far_gap": f["far_gap_bps"].fillna(0).clip(-500, 500),
        "near_missing": f["near_price"].isna().astype(float), "far_missing": f["far_price"].isna().astype(float),
        "ref_gap_bps": (f["reference_price"] / f["wap"] - 1) * 1e4,
        "depth_imbalance": (f["bid_size"] - f["ask_size"]) / (f["bid_size"] + f["ask_size"]).replace(0, np.nan),
        "seconds": f["seconds_in_bucket"].astype(float) / 540, "post": f["post"].astype(float),
    })
    for steps, name in [(1, "ret_10"), (3, "ret_30"), (6, "ret_60")]:
        out[name] = (wap / wap.groupby(keys).shift(steps) - 1) * 1e4
    mid = (f["bid_price"] + f["ask_price"]).astype("float64") / 2
    stock_fwd = (wap.groupby(keys).shift(-6) / wap - 1) * 1e4
    mid_fwd = (mid.groupby(keys).shift(-6) / mid - 1) * 1e4
    out["target_mid"] = mid_fwd - (stock_fwd - out["target"])
    for steps, name in [(1, "d_imb_10"), (6, "d_imb_60")]:
        out[name] = imb - imb.groupby(keys).shift(steps)
    out[FEATURES] = out[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e4, 1e4)
    return out


def split(frame, days):
    return frame[frame["date_id"].between(*days)]


def daily_rank_ic(day, pred, target):
    frame = pd.DataFrame({"day": day, "x": pred, "y": target})
    frame["x"] = frame.groupby("day")["x"].rank()
    frame["y"] = frame.groupby("day")["y"].rank()
    return frame.groupby("day").apply(lambda d: d["x"].corr(d["y"]), include_groups=False)


class Scaler:
    def __init__(self, x):
        self.mean = x.mean(axis=0)
        self.sd = x.std(axis=0) + 1e-6

    def __call__(self, x):
        return ((x - self.mean) / self.sd).astype("float32")


class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GRU(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.gru = nn.GRU(n_in, 64, batch_first=True)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        return self.head(self.gru(x)[0]).squeeze(-1)


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_torch(model, batches, valid_fn, epochs=20, patience=3):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state, waited = -np.inf, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb, mb in batches():
            opt.zero_grad()
            loss = (((model(xb) - yb) ** 2) * mb).sum() / mb.sum()
            loss.backward()
            opt.step()
        score = valid_fn(model)
        if score > best:
            best, best_state, waited = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    return model, float(best)


def predict_rows(model, x, batch=65536):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i + batch]).to(DEVICE)).cpu().numpy()
                               for i in range(0, len(x), batch)])


def run_mlp(train, valid, test, features, seed):
    seed_all(seed)
    scale = Scaler(train[features].to_numpy())
    xt = torch.from_numpy(scale(train[features].to_numpy())).to(DEVICE)
    yt = torch.from_numpy(train["target_w"].to_numpy().astype("float32")).to(DEVICE)
    ones = torch.ones_like(yt)
    xv = scale(valid[features].to_numpy())
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def batches():
        order = torch.randperm(len(yt), generator=gen).to(DEVICE)
        for i in range(0, len(order), 4096):
            idx = order[i:i + 4096]
            yield xt[idx], yt[idx], ones[idx]

    def valid_fn(model):
        return daily_rank_ic(valid["date_id"].to_numpy(), predict_rows(model, xv), valid["target"].to_numpy()).mean()

    model, best = fit_torch(MLP(len(features)).to(DEVICE), batches, valid_fn)
    return predict_rows(model, scale(test[features].to_numpy())), best


def run_lgbm(train, valid, test, seed):
    params = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "seed": seed, "verbose": -1, "num_threads": 16}
    dtrain = lgb.Dataset(train[FEATURES].to_numpy(), train["target_w"].to_numpy())
    dvalid = lgb.Dataset(valid[FEATURES].to_numpy(), valid["target_w"].to_numpy(), reference=dtrain)
    booster = lgb.train(params, dtrain, 2000, valid_sets=[dvalid],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    importance = dict(zip(FEATURES, booster.feature_importance("gain").tolist()))
    return booster.predict(test[FEATURES].to_numpy()), booster.best_iteration, importance


def sequences(frame, features, scale):
    ordered = frame.sort_values(["stock_id", "date_id", "sec"])
    groups = ordered.groupby(["stock_id", "date_id"]).ngroup().to_numpy()
    step = (ordered["sec"].to_numpy() // 10).astype(int)
    n = groups.max() + 1
    x = np.zeros((n, 55, len(features)), dtype="float32")
    y = np.zeros((n, 55), dtype="float32")
    mask = np.zeros((n, 55), dtype="float32")
    x[groups, step] = scale(ordered[features].to_numpy())
    y[groups, step] = ordered["target_w"].to_numpy()
    mask[groups, step] = 1
    return x, y, mask, groups, step, ordered.index.to_numpy()


def run_gru(train, valid, test, seed):
    seed_all(seed)
    scale = Scaler(train[SNAPSHOT].to_numpy())
    xt, yt, mt, *_ = sequences(train, SNAPSHOT, scale)
    xt, yt, mt = (torch.from_numpy(a).to(DEVICE) for a in (xt, yt, mt))
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def batches():
        order = torch.randperm(len(yt), generator=gen).to(DEVICE)
        for i in range(0, len(order), 256):
            idx = order[i:i + 256]
            yield xt[idx], yt[idx], mt[idx]

    def seq_predict(model, frame):
        x, _, _, groups, step, index = sequences(frame, SNAPSHOT, scale)
        out = predict_rows(model, x, batch=2048)
        return pd.Series(out[groups, step], index=index).reindex(frame.index).to_numpy()

    def valid_fn(model):
        return daily_rank_ic(valid["date_id"].to_numpy(), seq_predict(model, valid), valid["target"].to_numpy()).mean()

    model, best = fit_torch(GRU(len(SNAPSHOT)).to(DEVICE), batches, valid_fn)
    return pd.Series(seq_predict(model, test), index=test.index), best


def minute_rows(frame):
    return frame[frame["sec"].isin(TEST_SECONDS)]


def gain_test(day, pred, base, target, alternative="greater"):
    ic_model = daily_rank_ic(day, pred, target)
    ic_base = daily_rank_ic(day, base, target)
    res = t_summary(ic_model - ic_base, alternative=alternative)
    res["ic_model"] = float(ic_model.mean())
    res["ic_base"] = float(ic_base.mean())
    return res, ic_model, ic_base


def sign_flip(delta, seed=0):
    rng = np.random.default_rng(seed)
    delta = np.asarray(delta, dtype="float64")
    null = (rng.choice([-1.0, 1.0], size=(N_FLIP, len(delta))) * delta).mean(axis=1)
    return float((1 + np.sum(null >= delta.mean())) / (N_FLIP + 1))


def diebold_mariano(day, pred, bench, target):
    diff = pd.Series((target - bench) ** 2 - (target - pred) ** 2).groupby(day).mean()
    return t_summary(diff, alternative="greater")


def c2st(train, test, seed=0):
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "seed": seed, "verbose": -1,
              "num_threads": 16}
    rows = minute_rows(train)
    booster = lgb.train(params, lgb.Dataset(rows[C2ST_FEATURES].to_numpy(), rows["post"].to_numpy()), 300)
    held = minute_rows(test)
    correct = ((booster.predict(held[C2ST_FEATURES].to_numpy()) > 0.5) == (held["post"].to_numpy() == 1))
    n, p_hat = len(correct), correct.mean()
    z = (p_hat - 0.5) / np.sqrt(0.25 / n)
    daily = pd.Series(correct).groupby(held["date_id"].to_numpy()).mean()
    return {"n": int(n), "accuracy": float(p_hat), "z": float(z), "p": float(stats.norm.sf(z)),
            "day_level": t_summary(daily, alternative="greater", null=0.5)}


def selective_trading(test_rows, pred, column="target"):
    keep = test_rows[column].notna().to_numpy()
    test_rows, pred = test_rows[keep], pred[keep]
    trade = np.abs(pred) > test_rows["half_spread"].to_numpy()
    sub = test_rows[trade]
    net = np.sign(pred[trade]) * sub[column].to_numpy() - sub["half_spread"].to_numpy()
    daily = pd.Series(net).groupby(sub["date_id"].to_numpy()).mean()
    res = t_summary(daily, alternative="greater")
    res.update({"trades": int(trade.sum()), "share_traded": float(trade.mean()), "hit_rate": float((net > 0).mean()),
                "days_with_trades": int(daily.size)})
    return res


def mid_check(test_rows, pred, base, lgbm):
    keep = test_rows["target_mid"].notna().to_numpy()
    rows = test_rows[keep]
    day = rows["date_id"].to_numpy()
    on_mid, ic_mid_model, ic_mid_base = gain_test(day, pred[keep], base[keep], rows["target_mid"].to_numpy())
    on_wap, ic_wap_model, ic_wap_base = gain_test(day, pred[keep], base[keep], rows["target"].to_numpy())
    ic_mid_lgbm = daily_rank_ic(day, lgbm[keep], rows["target_mid"].to_numpy())
    ic_wap_lgbm = daily_rank_ic(day, lgbm[keep], rows["target"].to_numpy())
    share = on_mid["estimate"] / on_wap["estimate"]
    return {"gain_on_mid": on_mid, "gain_on_wap_same_rows": on_wap, "share_retained": float(share),
            "mostly_wap_mechanics": bool(share < 0.5),
            "selective_trading_mid": selective_trading(test_rows, pred, "target_mid"),
            "daily": {"wap": {"imbalance": ic_wap_base.tolist(), "mlp": ic_wap_model.tolist(),
                              "lgbm": ic_wap_lgbm.tolist()},
                      "mid": {"imbalance": ic_mid_base.tolist(), "mlp": ic_mid_model.tolist(),
                              "lgbm": ic_mid_lgbm.tolist()}}}


def run_learning(df):
    frame = build_frame(df)
    train, valid, test = split(frame, TRAIN_DAYS), split(frame, VALID_DAYS), split(frame, TEST_DAYS)
    test_rows = minute_rows(test)
    day, target = test_rows["date_id"].to_numpy(), test_rows["target"].to_numpy()
    base = test_rows["signed_imbalance"].to_numpy()

    mlp, lgbm, gru = [], [], []
    mlp_snap = []
    seeds_ic = {"mlp": [], "lgbm": [], "gru": []}
    importance = []
    for seed in SEEDS:
        p, _ = run_mlp(train, valid, test_rows, FEATURES, seed)
        mlp.append(p)
        mlp_snap.append(run_mlp(train, valid, test_rows, SNAPSHOT, seed)[0])
        q, _, imp = run_lgbm(train, valid, test_rows, seed)
        lgbm.append(q)
        importance.append(imp)
        gru.append(run_gru(train, valid, test, seed)[0].loc[test_rows.index].to_numpy())
        for name, arr in [("mlp", p), ("lgbm", q), ("gru", gru[-1])]:
            seeds_ic[name].append(float(daily_rank_ic(day, arr, target).mean()))
    mlp, lgbm, gru, mlp_snap = (np.mean(a, axis=0) for a in (mlp, lgbm, gru, mlp_snap))

    primary, ic_mlp, ic_base = gain_test(day, mlp, base, target)
    primary["matters_threshold"] = MATTERS_GAIN
    primary["verdict"] = ("matters" if primary["ci"][0] > MATTERS_GAIN else
                          "does not matter" if primary["ci"][1] < MATTERS_GAIN else "inconclusive")
    delta = (ic_mlp - ic_base).to_numpy()
    lgbm_vs_base, ic_lgbm, _ = gain_test(day, lgbm, base, target)
    mlp_vs_lgbm, _, _ = gain_test(day, mlp, lgbm, target, alternative="two-sided")
    gru_vs_snap, ic_gru, ic_snap = gain_test(day, gru, mlp_snap, target, alternative="two-sided")

    slope = np.polyfit(train["signed_imbalance"], train["target_w"], 1)
    linear = np.polyval(slope, base)
    mean_imp = {k: float(np.mean([imp[k] for imp in importance])) for k in FEATURES}
    total = sum(mean_imp.values())

    return {
        "split": {"train": TRAIN_DAYS, "valid": VALID_DAYS, "test": TEST_DAYS, "train_rows": int(len(train)),
                  "test_rows": int(len(test_rows)), "features": FEATURES},
        "primary": primary,
        "permutation_p": sign_flip(delta),
        "lgbm_vs_imbalance": lgbm_vs_base,
        "mlp_vs_lgbm": mlp_vs_lgbm,
        "gru_vs_snapshot_mlp": gru_vs_snap,
        "dm_vs_zero": diebold_mariano(day, mlp, np.zeros_like(mlp), target),
        "dm_vs_linear": diebold_mariano(day, mlp, linear, target),
        "seed_ic": seeds_ic,
        "seed_sd": {k: float(np.std(v, ddof=1)) for k, v in seeds_ic.items()},
        "importance_share": {k: v / total for k, v in sorted(mean_imp.items(), key=lambda kv: -kv[1])},
        "c2st": c2st(train, test),
        "selective_trading": selective_trading(test_rows, mlp),
        "mid_check": mid_check(test_rows, mlp, base, lgbm),
        "daily": {"day": sorted(set(day.tolist())), "imbalance": ic_base.tolist(), "mlp": ic_mlp.tolist(),
                  "lgbm": ic_lgbm.tolist(), "gru": ic_gru.tolist(), "mlp_snapshot": ic_snap.tolist()},
    }

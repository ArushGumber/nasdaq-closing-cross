import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy import stats

from .plotting import COL, FULL, GRID, INK, MUTED, POST, PRE, grid, save

MODEL_COLORS = {"imbalance": MUTED, "lgbm": PRE, "mlp": POST, "gru": INK}
MODEL_LABELS = {"imbalance": "imbalance alone", "lgbm": "LightGBM", "mlp": "MLP", "gru": "GRU"}


def cumulative_ic(daily):
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.9), sharey=True)
    for ax, key, title in [(axes[0], "wap", "official target (WAP)"), (axes[1], "mid", "mid-price target")]:
        for model in ["imbalance", "lgbm", "mlp"]:
            y = np.cumsum(daily[key][model])
            ax.plot(np.arange(1, len(y) + 1), y, color=MODEL_COLORS[model],
                    linewidth=1.6 if model == "imbalance" else 1.1, label=MODEL_LABELS[model])
        ax.text(0.04, 0.96, title, transform=ax.transAxes, va="top", fontsize=8)
        ax.set_xlabel("held-out test day")
        grid(ax)
    axes[0].set_ylabel("cumulative daily IC")
    axes[1].legend(loc="center right", fontsize=7, handlelength=1.2)
    fig.tight_layout(w_pad=0.6)
    return save(fig, "fig_cumulative_ic")


def index_weights(weights):
    w = np.sort(np.asarray(weights))[::-1]
    fig, ax = plt.subplots(figsize=(COL, 1.5))
    ax.bar(np.arange(len(w)), w * 100, width=1.0, color=POST, linewidth=0)
    ax.set_xlabel("stock, ranked by recovered index weight")
    ax.set_ylabel("weight (%)")
    ax.set_xlim(-1, len(w))
    grid(ax)
    return save(fig, "figS_index_weights")


def coverage(present):
    fig, ax = plt.subplots(figsize=(FULL, 1.9))
    ax.imshow(~present.to_numpy(), aspect="auto", cmap=ListedColormap(["#f1f1ef", POST]), interpolation="nearest")
    ax.set_xlabel("trading day")
    ax.set_ylabel("stock")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(True)
    return save(fig, "figS_coverage")


def forest(rows, labels, xlabel, name, threshold=None, highlight=None, height=None):
    fig, ax = plt.subplots(figsize=(COL, height or 0.22 * len(rows) + 0.5))
    y = np.arange(len(rows))[::-1]
    for yi, label, r in zip(y, labels, rows):
        color = POST if highlight is None or label == highlight else MUTED
        ax.plot(r["ci"], [yi, yi], color=color, linewidth=1.6)
        ax.plot(r["estimate"], yi, "o", color=color, markersize=3.5)
    ax.axvline(0, color=INK, linewidth=0.6)
    if threshold is not None:
        ax.axvline(threshold, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", color=GRID, linewidth=0.5)
    return save(fig, name)


def robustness_forest(groups):
    return forest(list(groups.values()), list(groups), "rise in IC after 15:55 (95% CI)", "figS_robustness", 0.02)


def placebo_forest(jumps):
    labels = [f"15:{50 + int(b) // 60}" for b in jumps]
    return forest(list(jumps.values()), labels, "jump in IC from the minute before (95% CI)", "figS_placebo",
                  highlight="15:55")


def power_curves(power):
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.6))
    for ax, key, unit in [(axes[0], "T1", "edge (bps)"), (axes[1], "T3", "change in IC")]:
        r = power[key]
        a = 0.05 / 2 if r["two_sided"] else 0.05
        se = r["sd"] / np.sqrt(r["n"])
        x = np.linspace(0, 4 * r["mde"], 200)
        ax.plot(x, stats.norm.sf(stats.norm.ppf(1 - a) - x / se), color=POST)
        ax.axhline(0.8, color=MUTED, linewidth=0.8, linestyle="--")
        ax.axvline(r["mde"], color=PRE, linewidth=0.8)
        ax.set_xlabel(unit)
        ax.set_ylim(0, 1.02)
        grid(ax)
    axes[0].set_ylabel("power")
    fig.tight_layout(w_pad=0.8)
    return save(fig, "figS_power")


def edge_by_strength(rows):
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    for post, color, label in [(0, PRE, "edge before 15:55"), (1, POST, "edge after 15:55")]:
        part = sorted((r for r in rows if r["post"] == post), key=lambda r: r["bin"])
        x = [r["bin"] + 1 for r in part]
        ax.plot(x, [r["gross"] for r in part], "o-", color=color, label=label)
        ax.plot(x, [r["half_spread"] for r in part], "--", color=color, linewidth=0.9)
    ax.plot([], [], "--", color=MUTED, label="half-spread (dashed)")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("strength decile of |signed imbalance|")
    ax.set_ylabel("bps per trade")
    ax.legend(loc="upper left", fontsize=7)
    grid(ax)
    return save(fig, "figS_edge_strength")


def importance(shares, top=12):
    items = list(shares.items())[:top][::-1]
    fig, ax = plt.subplots(figsize=(COL, 0.17 * top + 0.5))
    ax.barh([k.replace("_", " ") for k, _ in items], [v * 100 for _, v in items], color=POST, height=0.6)
    ax.set_xlabel("share of LightGBM split gain (%)")
    ax.grid(True, axis="x", color=GRID, linewidth=0.5)
    return save(fig, "figS_importance")


def seed_spread(seed_ic, base_ic):
    fig, ax = plt.subplots(figsize=(COL, 1.5))
    keys = ["mlp", "lgbm", "gru"]
    for i, key in enumerate(keys):
        vals = seed_ic[key]
        ax.plot(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals, "o", color=MODEL_COLORS[key],
                markersize=4)
    ax.axhline(base_ic, color=MUTED, linestyle="--", linewidth=0.9)
    ax.text(2.45, base_ic, "imbalance alone", va="bottom", ha="right", fontsize=7.5, color=MUTED)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([MODEL_LABELS[k] for k in keys])
    ax.set_ylabel("mean test IC per seed")
    ax.set_xlim(-0.5, 2.5)
    grid(ax)
    return save(fig, "figS_seeds")


def t_fit(target, fits):
    x = np.asarray(target, dtype="float64")
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    edges = np.linspace(-80, 80, 161)
    ax.hist(x, bins=edges, density=True, color=MUTED, alpha=0.6, linewidth=0)
    grid_x = np.linspace(-80, 80, 600)
    n, t = fits["normal"], fits["student_t"]
    ax.plot(grid_x, stats.norm.pdf(grid_x, n["mean"], n["sd"]), color=PRE, label="normal (MLE)")
    ax.plot(grid_x, stats.t.pdf(grid_x, t["df"], t["loc"], t["scale"]), color=POST,
            label=f"Student t, {t['df']:.1f} d.f. (MLE)")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 0.2)
    ax.set_xlabel("target (bps)")
    ax.set_ylabel("density (log)")
    ax.legend(loc="upper left", fontsize=7, handlelength=1.2)
    ax.set_ylim(1e-6, 2)
    return save(fig, "figS_tfit")


def flips(observed, expected):
    k = np.arange(len(observed))
    fig, ax = plt.subplots(figsize=(COL, 1.6))
    ax.bar(k - 0.2, np.asarray(observed) / 1000, width=0.4, color=POST, label="observed")
    ax.bar(k + 0.2, np.asarray(expected) / 1000, width=0.4, color=MUTED, label="Poisson, same mean")
    ax.set_xticks(k)
    ax.set_xticklabels([str(i) for i in k[:-1]] + [f"{k[-1]}+"])
    ax.set_xlabel("side changes in one stock-day")
    ax.set_ylabel("stock-days (thousands)")
    ax.legend(loc="upper right")
    grid(ax)
    return save(fig, "figS_flips")


def countmin(error_share):
    e = np.sort(np.asarray(error_share))
    fig, ax = plt.subplots(figsize=(COL, 1.5))
    ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=POST)
    ax.axvline(1, color=PRE, linewidth=0.9, linestyle="--")
    ax.text(0.98, 0.1, "guarantee\n(epsilon N)", ha="right", fontsize=7.5, color=PRE)
    ax.set_xlabel("overestimate as a share of the guarantee")
    ax.set_ylabel("share of stock-days")
    ax.set_xlim(0, 1.05)
    grid(ax)
    return save(fig, "figS_countmin")

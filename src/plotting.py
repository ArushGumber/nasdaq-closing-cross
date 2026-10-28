import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .config import BOUNDARY, FIGURES

COL = 3.29
FULL = 6.85
POST = "#2f6db5"
PRE = "#c8613f"
MUTED = "#9aa0a6"
INK = "#222222"
GRID = "#e6e6e6"


def setup():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Carlito", "Calibri", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Carlito", "mathtext.it": "Carlito:italic", "mathtext.bf": "Carlito:bold",
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.01, "pdf.fonttype": 42,
    })


def save(fig, name):
    (FIGURES / "png").mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.pdf"
    fig.savefig(path)
    fig.savefig(FIGURES / "png" / f"{name}.png", dpi=220)
    plt.close(fig)
    return path


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def clock_axis(ax):
    ticks = np.arange(0, 541, 120)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"15:{50 + t // 60}" if t < 600 else "16:00" for t in ticks])
    ax.set_xlim(-10, 550)


def boundary_line(ax, label=True):
    ax.axvline(BOUNDARY, color=MUTED, linewidth=0.8, linestyle="--", zorder=0)
    if label:
        ax.text(BOUNDARY + 6, 1.0, "15:55", transform=ax.get_xaxis_transform(), va="top", fontsize=8, color=MUTED)


def target_distribution(target):
    x = target[np.isfinite(target)]
    mu, sd = x.mean(), x.std()
    fig, (ax, qq) = plt.subplots(1, 2, figsize=(COL, 1.75), gridspec_kw={"width_ratios": [1.6, 1]})
    edges = np.linspace(-60, 60, 121)
    ax.hist(x, bins=edges, density=True, color=POST, alpha=0.85, linewidth=0)
    grid_x = np.linspace(-60, 60, 400)
    ax.plot(grid_x, stats.norm.pdf(grid_x, mu, sd), color=PRE, linewidth=1.1)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 0.3)
    ax.text(-60, 0.06, "normal fit", color=PRE, fontsize=8)
    ax.set_xlabel("target (bps)")
    ax.set_ylabel("density (log)")

    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(x, 20000, replace=False))
    theo = stats.norm.ppf((np.arange(1, len(sample) + 1) - 0.5) / len(sample), mu, sd)
    qq.plot(theo, sample, ".", color=POST, markersize=1.5, rasterized=True)
    lim = [-120, 120]
    qq.plot(lim, lim, color=PRE, linewidth=0.9)
    qq.set_xlim(-45, 45)
    qq.set_ylim(*lim)
    qq.set_xlabel("normal quantile")
    qq.set_ylabel("sample quantile")
    fig.tight_layout(w_pad=0.6)
    return save(fig, "fig_target_distribution")


def autocorrelation(acf):
    lags = np.array(sorted(acf))
    vals = np.array([acf[k] for k in lags])
    fig, ax = plt.subplots(figsize=(COL, 1.45))
    colors = [POST if 0 < k < 6 else MUTED for k in lags]
    ax.bar(lags * 10, vals, width=6, color=colors)
    overlap = np.clip(1 - lags / 6, 0, None)
    ax.plot(lags * 10, overlap, "_", color=INK, markersize=9, markeredgewidth=1.2, label="share of window overlap")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.62), handlelength=1)
    ax.axhline(0, color=INK, linewidth=0.6)
    ax.axvline(55, color=MUTED, linewidth=0.8, linestyle="--")
    ax.text(58, 1.0, "windows stop overlapping", fontsize=8, color=MUTED, va="top")
    ax.set_xticks(lags[::2] * 10)
    ax.set_xlabel("lag between snapshots (seconds)")
    ax.set_ylabel("autocorrelation")
    grid(ax)
    return save(fig, "fig_autocorrelation")


def auction_profile(profile):
    sec = profile.index.to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(COL, 3.0), sharex=True)
    series = [("abs_target", "|target|, bps"), ("median_spread_bps", "spread, bps"),
              ("median_abs_imbalance", "|imbalance ratio|")]
    for ax, (col, label) in zip(axes, series):
        y = profile[col].to_numpy()
        ax.plot(sec[sec < BOUNDARY], y[sec < BOUNDARY], color=PRE)
        ax.plot(sec[sec >= BOUNDARY], y[sec >= BOUNDARY], color=POST)
        ax.set_ylabel(label)
        boundary_line(ax, label=ax is axes[0])
        grid(ax)
    clock_axis(axes[-1])
    axes[-1].set_xlabel("time of the closing auction")
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.3)
    return save(fig, "fig_auction_profile")


def decile_response(deciles):
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    for phase, color, label in [(0, PRE, "before 15:55"), (1, POST, "after 15:55")]:
        d = deciles[deciles["post"] == phase].sort_values("bin")
        x = d["bin"].to_numpy() + 1
        ax.plot(x, d["mean"], "o-", color=color, label=label)
        ax.fill_between(x, d["mean"] - 1.96 * d["se"], d["mean"] + 1.96 * d["se"], color=color, alpha=0.18,
                        linewidth=0)
    ax.axhline(0, color=INK, linewidth=0.6)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("signed imbalance decile (sell heavy to buy heavy)")
    ax.set_ylabel("mean target (bps)")
    ax.legend(loc="upper left")
    grid(ax)
    return save(fig, "fig_decile_response")


def outlier_fences(fences):
    labels = {"iqr_1.5": "1.5 IQR", "iqr_3": "3 IQR", "mad_5": "5 MAD", "sigma_3": "3 sd",
              "chebyshev_10": "10 sd (Chebyshev)"}
    keys = list(labels)
    shares = [fences[k]["share"] * 100 for k in keys]
    fig, ax = plt.subplots(figsize=(COL, 1.35))
    colors = [POST if k == "chebyshev_10" else MUTED for k in keys]
    y = np.arange(len(keys))[::-1]
    ax.barh(y, shares, color=colors, height=0.6)
    for yi, s in zip(y, shares):
        ax.text(s + 0.6, yi, f"{s:.2f}%" if s < 1 else f"{s:.1f}%", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[k] for k in keys])
    ax.set_xlabel("far prices flagged as outliers (%)")
    ax.set_xlim(0, 38)
    return save(fig, "fig_outlier_fences")


def daily_ic(pre, post):
    pre, post = np.asarray(pre), np.asarray(post)
    delta = post - pre
    fig, ax = plt.subplots(figsize=(COL, 1.6))
    edges = np.linspace(-0.4, 0.4, 41)
    ax.hist(delta, bins=edges, color=POST, alpha=0.85, linewidth=0)
    ax.axvline(0, color=INK, linewidth=0.6)
    ax.axvline(delta.mean(), color=PRE, linewidth=1.1)
    ax.text(delta.mean(), 1.0, " mean", transform=ax.get_xaxis_transform(), va="top", fontsize=8, color=PRE)
    ax.set_xlabel("daily IC after 15:55 minus before")
    ax.set_ylabel("days")
    grid(ax)
    return save(fig, "fig_daily_ic")


def ic_profile(profile):
    sec = np.asarray(profile["seconds"])
    mean = np.asarray(profile["mean"])
    se = np.asarray(profile["se"])
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    for part, color in [(sec < BOUNDARY, PRE), (sec >= BOUNDARY, POST)]:
        ax.plot(sec[part], mean[part], "o-", color=color, markersize=2.2, linewidth=1)
        ax.fill_between(sec[part], mean[part] - 1.96 * se[part], mean[part] + 1.96 * se[part], color=color,
                        alpha=0.2, linewidth=0)
    ax.axhline(0, color=INK, linewidth=0.6)
    boundary_line(ax)
    ax.annotate("last minute:\nwindow holds\nthe 16:00 cross", xy=(540, mean[-1]), xytext=(330, 0.2), va="center",
                fontsize=7.5, color=INK, arrowprops={"arrowstyle": "->", "linewidth": 0.6, "color": INK})
    clock_axis(ax)
    ax.set_xlabel("time of the snapshot")
    ax.set_ylabel("IC across stocks")
    grid(ax)
    return save(fig, "fig_ic_profile")


def edge_vs_cost(by_second):
    sec = np.asarray(by_second["seconds"])
    gross = np.asarray(by_second["gross"])
    se = np.asarray(by_second["gross_se"])
    cost = np.asarray(by_second["half_spread"])
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    ax.plot(sec, cost, color=MUTED, linewidth=1.4, label="half-spread (cost)")
    for part, color in [(sec < BOUNDARY, PRE), (sec >= BOUNDARY, POST)]:
        ax.plot(sec[part], gross[part], color=color, linewidth=1.2)
        ax.fill_between(sec[part], gross[part] - 1.96 * se[part], gross[part] + 1.96 * se[part], color=color,
                        alpha=0.2, linewidth=0)
    ax.plot([], [], color=POST, label="edge per trade")
    ax.axhline(0, color=INK, linewidth=0.6)
    boundary_line(ax, label=False)
    clock_axis(ax)
    ax.set_xlabel("time of the snapshot")
    ax.set_ylabel("bps per trade")
    ax.legend(loc="upper left", ncol=2, handlelength=1.2)
    ax.set_ylim(-1.5, 6.2)
    grid(ax)
    return save(fig, "fig_edge_vs_cost")

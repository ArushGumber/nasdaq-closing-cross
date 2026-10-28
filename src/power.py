import numpy as np
from scipy import stats

from .config import ALPHA, MATTERS_DELTA_IC, MATTERS_HIT_RATE, MATTERS_VAR_RATIO

POWER = 0.80


def detectable(sd, n, two_sided):
    a = ALPHA / 2 if two_sided else ALPHA
    return float((stats.norm.ppf(1 - a) + stats.norm.ppf(POWER)) * sd / np.sqrt(n))


def days_needed(sd, effect, two_sided):
    a = ALPHA / 2 if two_sided else ALPHA
    return int(np.ceil(((stats.norm.ppf(1 - a) + stats.norm.ppf(POWER)) * sd / effect) ** 2))


def power_table(tests, sdm):
    t = tests["tests"]
    t1_daily = np.asarray(t["T1"]["daily"])
    t3_delta = np.asarray(t["T3"]["daily_post"]) - np.asarray(t["T3"]["daily_pre"])
    t4 = t["T4"]["day_level_log_ratio"]
    t5 = t["T5"]["day_level"]
    rows = {
        "T1": {"unit": "bps per trade", "sd": float(t1_daily.std(ddof=1)), "n": len(t1_daily), "two_sided": False,
               "threshold": sdm, "observed": t["T1"]["primary"]["estimate"]},
        "T3": {"unit": "IC", "sd": float(t3_delta.std(ddof=1)), "n": len(t3_delta), "two_sided": True,
               "threshold": MATTERS_DELTA_IC, "observed": t["T3"]["primary"]["estimate"]},
        "T4": {"unit": "log variance ratio", "sd": t4["se"] * np.sqrt(t4["n"]), "n": t4["n"], "two_sided": True,
               "threshold": float(-np.log(MATTERS_VAR_RATIO[0])), "observed": t4["estimate"]},
        "T5": {"unit": "hit rate", "sd": t5["se"] * np.sqrt(t5["n"]), "n": t5["n"], "two_sided": False,
               "threshold": MATTERS_HIT_RATE - 0.5, "observed": t5["estimate"] - 0.5},
    }
    for r in rows.values():
        r["mde"] = detectable(r["sd"], r["n"], r["two_sided"])
        r["days_for_threshold"] = days_needed(r["sd"], r["threshold"], r["two_sided"])
    return rows

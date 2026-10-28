import json

from src import bounds, costs, eda, fits, learning, plotting, plotting_extra, power, robustness, sketch
from src.clean import clean, encoded_shape, panel_gaps, snapshot
from src.config import PROCESSED, RESULTS
from src.features import add_features, reconstruct_index
from src.inference import holm_family, run_tests, test_sample
from src.load import load_raw
from src.numbers import write_numbers


def dump(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2)


def main():
    raw = load_raw()
    before = snapshot(raw)
    raw_cells = {"rows": int(raw.shape[0]), "columns": int(raw.shape[1]), "cells": int(raw.size)}

    df, log, meta = clean(raw)
    del raw
    after = snapshot(df)
    df = add_features(df)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED)

    present = panel_gaps(df)
    index = reconstruct_index(df)
    weights = index.pop("weights")
    dump("prep", {"before": before, "after": after, "raw_shape": raw_cells, "encoded_shape": encoded_shape(df),
                  "steps": log, "meta": meta, "coverage": eda.coverage(present), "index": index,
                  "index_weights": weights})

    s = test_sample(df)
    acf = eda.autocorrelation(df)
    profile = eda.by_second(df)
    deciles = eda.decile_profile(df)
    dump("eda", {"describe": eda.describe(df), "flag_shares": eda.flag_shares(df),
                 "normal_gof": eda.normal_gof(s["target"].to_numpy()),
                 "tail_shares": bounds.tail_shares(df["target"].to_numpy()), "autocorrelation": acf,
                 "by_second": profile.reset_index().to_dict(orient="list"),
                 "deciles": deciles.to_dict(orient="list")})

    tests = run_tests(df)
    t1 = tests["tests"]["T1"]
    bound = max(abs(b) for b in meta["winsor_bounds"])
    dump("bounds", bounds.interval_comparison(t1["daily"], bound))
    dump("tests", tests)

    s_delta = [b - a for a, b in zip(tests["tests"]["T3"]["daily_pre"], tests["tests"]["T3"]["daily_post"])]
    placebo = robustness.placebo_boundaries(df)
    cost = costs.cost_analysis(df, s)
    extra = {
        "robustness": robustness.subgroup_delta_ic(s, df),
        "placebo": placebo,
        "power": power.power_table(tests, tests["sdm_bps"]),
        "costs": cost,
        "permutation_t1": robustness.permutation_t1(s),
        "sign_flip_t3": robustness.sign_flip_t3(s_delta),
        "target_fits": fits.target_fits(s["target"].to_numpy()),
        "side_flips": fits.side_flips(df),
        "count_min": sketch.heavy_stock_days(df),
        "mid_price": robustness.mid_price_checks(learning.build_frame(df)),
    }
    dump("extensions", extra)

    learned = learning.run_learning(df)
    dump("learning", learned)
    pvals = {k: v["primary"]["p"] for k, v in tests["tests"].items()}
    pvals["T6"] = learned["primary"]["p"]
    dump("family", holm_family(pvals))

    plotting.setup()
    plotting.ic_profile(placebo["profile"])
    plotting.edge_vs_cost(cost["by_second"])
    plotting.target_distribution(df["target"].to_numpy())
    plotting.autocorrelation(acf)
    plotting.auction_profile(profile)
    plotting.decile_response(deciles)
    plotting.outlier_fences(meta["far_price_fences"])
    plotting.daily_ic(tests["tests"]["T3"]["daily_pre"], tests["tests"]["T3"]["daily_post"])
    plotting_extra.cumulative_ic(learned["mid_check"]["daily"])
    plotting_extra.index_weights(weights)
    plotting_extra.coverage(present)
    plotting_extra.robustness_forest(extra["robustness"]["groups"])
    plotting_extra.placebo_forest(placebo["jumps"])
    plotting_extra.power_curves(extra["power"])
    plotting_extra.edge_by_strength(cost["by_strength"])
    plotting_extra.importance(learned["importance_share"])
    plotting_extra.seed_spread(learned["seed_ic"], learned["primary"]["ic_base"])
    plotting_extra.t_fit(s["target"].to_numpy(), extra["target_fits"])
    plotting_extra.flips(extra["side_flips"]["observed"], extra["side_flips"]["expected"])
    plotting_extra.countmin(extra["count_min"]["error_share_of_bound"])

    write_numbers()


if __name__ == "__main__":
    main()

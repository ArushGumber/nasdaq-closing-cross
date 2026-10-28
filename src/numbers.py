import json
import math

from .config import MATTERS_CRAMER_V, MATTERS_DELTA_IC, MATTERS_HIT_RATE, MATTERS_VAR_RATIO, RESULTS


def load(name):
    with open(RESULTS / f"{name}.json") as f:
        return json.load(f)


def count(x):
    return f"{int(x):,}".replace(",", "{,}")


def num(x, digits=2):
    return f"{x:.{digits}f}"


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}\\%"


def pval(p):
    if p == 0 or p < 1e-300:
        return "$<10^{-300}$"
    if p >= 0.001:
        return f"${p:.3f}$"
    exponent = math.floor(math.log10(p))
    mantissa = round(p / 10**exponent, 1)
    if mantissa >= 10:
        mantissa, exponent = mantissa / 10, exponent + 1
    return f"${mantissa:.1f}\\times10^{{{exponent}}}$"


def ci(pair, digits=2):
    return f"{pair[0]:.{digits}f} to {pair[1]:.{digits}f}"


def macros():
    prep, eda, tests, bounds = load("prep"), load("eda"), load("tests"), load("bounds")
    ext = load("extensions")
    learn = load("learning")
    t = tests["tests"]
    t1, t2, t3, t4, t5 = (t[k] for k in ["T1", "T2", "T3", "T4", "T5"])
    steps = {s["step"]: s for s in prep["steps"]}
    enc = prep["encoded_shape"]
    acf = eda["autocorrelation"]
    by_second = eda["by_second"]
    sec = by_second["seconds_in_bucket"]
    flag_share = by_second["flag_share"]
    tails = eda["tail_shares"]["3"]
    m = {
        "RawRows": count(prep["raw_shape"]["rows"]),
        "RawCols": count(prep["raw_shape"]["columns"]),
        "RawCells": count(prep["raw_shape"]["cells"]),
        "CleanRows": count(enc["rows"]),
        "EncCols": count(enc["columns"]),
        "EncCells": count(enc["cells"]),
        "EncNumeric": count(enc["numeric"]),
        "EncEngineered": count(enc["engineered"]),
        "Stocks": count(enc["one_hot"]["stock_id"]),
        "Days": count(len(t1["daily"])),
        "MemBefore": num(prep["before"]["memory_mb"], 0),
        "MemAfter": num(prep["after"]["memory_mb"], 0),
        "MissingBefore": count(prep["before"]["missing_cells"]),
        "MissingAfter": count(prep["after"]["missing_cells"]),
        "EmptyRows": count(steps["empty_book"]["rows_affected"]),
        "StructuralRows": count(steps["structural_missing"]["rows_affected"]),
        "FarOutliers": count(steps["far_price_outlier"]["rows_affected"]),
        "WinsorRows": count(steps["winsorise"]["rows_affected"]),
        "WinsorLow": num(prep["meta"]["winsor_bounds"][0], 1),
        "WinsorHigh": num(prep["meta"]["winsor_bounds"][1], 1),
        "FenceIqr": pct(prep["meta"]["far_price_fences"]["iqr_1.5"]["share"]),
        "FenceIqrThree": pct(prep["meta"]["far_price_fences"]["iqr_3"]["share"]),
        "FenceMad": pct(prep["meta"]["far_price_fences"]["mad_5"]["share"]),
        "FenceSigma": pct(prep["meta"]["far_price_fences"]["sigma_3"]["share"], 2),
        "FenceCheb": pct(prep["meta"]["far_price_fences"]["chebyshev_10"]["share"], 3),
        "StockDaysPresent": count(prep["coverage"]["present"]),
        "StockDaysMissing": count(prep["coverage"]["missing"]),
        "StocksWithGaps": count(prep["coverage"]["stocks_with_gaps"]),
        "MaxMissingDays": count(prep["coverage"]["max_missing_days_one_stock"]),
        "IndexUnexplained": pval(1 - prep["index"]["r2"]),
        "IndexWeightSum": num(prep["index"]["weights_sum"], 4),
        "IndexWeightMax": num(prep["index"]["weights_max"], 3),
        "IndexResidSd": num(prep["index"]["median_within_time_sd_bps"], 3),
        "TargetSd": num(eda["describe"]["target"]["sd"], 1),
        "TargetKurt": num(eda["describe"]["target"]["kurtosis"], 1),
        "TargetSkew": num(eda["describe"]["target"]["skew"], 2),
        "SpreadMedian": num(eda["describe"]["spread_bps"]["median"], 2),
        "TailObserved": pct(tails["observed"], 2),
        "TailNormal": pct(tails["normal"], 2),
        "TailCheb": pct(tails["chebyshev_max"], 1),
        "TailRatio": num(tails["observed"] / tails["normal"], 1),
        "EmptyNoTarget": count(prep["before"]["missing_by_column"]["target"]),
        "StockDaysBefore": count(prep["before"]["stock_days"]),
        "StructuralCells": count(2 * steps["structural_missing"]["rows_affected"]),
        "FarPostMissing": count(prep["after"]["missing_by_column"]["far_price"]
                                - steps["structural_missing"]["rows_affected"]
                                - steps["far_price_outlier"]["rows_affected"]),
        "GofChi": count(round(eda["normal_gof"]["chi2"])),
        "GofDf": count(eda["normal_gof"]["df"]),
        "AcfOne": num(acf["1"], 2),
        "AcfFive": num(acf["5"], 2),
        "AcfSix": num(acf["6"], 2),
        "FlagSell": pct(eda["flag_shares"]["-1"], 0),
        "FlagBuy": pct(eda["flag_shares"]["1"], 0),
        "FlagNone": pct(eda["flag_shares"]["0"], 0),
        "BalancedPre": pct(1 - flag_share[sec.index(290)], 0),
        "BalancedPost": pct(1 - flag_share[sec.index(300)], 0),
        "Sdm": num(tests["sdm_bps"], 2),
        "TestRows": count(tests["n_test_rows"]),
        "OneEst": num(t1["primary"]["estimate"], 2),
        "OneCi": ci(t1["primary"]["ci"]),
        "OneT": num(t1["primary"]["t"], 1),
        "OneDf": count(t1["primary"]["df"]),
        "OneP": pval(t1["primary"]["p"]),
        "OneRawEst": num(t1["raw_target"]["estimate"], 2),
        "OneRawCi": ci(t1["raw_target"]["ci"]),
        "OneBootCi": ci(t1["bootstrap_ci"]),
        "OneNaiveZ": num(t1["naive_row_z"]["z"], 1),
        "OneNaiveN": count(t1["naive_row_z"]["n"]),
        "OneOverlapN": count(t1["naive_overlapping_z"]["n"]),
        "OneWelchT": num(t1["welch_buy_vs_sell"]["t"], 1),
        "OneWelchDiff": num(t1["welch_buy_vs_sell"]["diff"], 2),
        "OneVarRatio": num(t1["variance_ratio_buy_sell"], 2),
        "OnePositiveDays": count(t1["positive_days"]),
        "OneShareOfSdm": pct(t1["primary"]["ci"][1] / tests["sdm_bps"], 0),
        "TwoChi": num(t2["primary"]["chi2"], 1),
        "TwoDf": count(t2["primary"]["df"]),
        "TwoP": pval(t2["primary"]["p"]),
        "TwoN": count(t2["primary"]["n"]),
        "TwoV": num(t2["primary"]["cramers_v"], 3),
        "TwoVThreshold": num(MATTERS_CRAMER_V, 2),
        "TwoMinExpected": count(round(t2["primary"]["min_expected"])),
        "TwoUpBuy": pct(t2["up_share"]["buy"]),
        "TwoUpSell": pct(t2["up_share"]["sell"]),
        "TwoDiff": num(100 * t2["two_proportion"]["diff"], 1),
        "TwoDiffCi": ci([100 * c for c in t2["two_proportion"]["ci"]], 1),
        "TwoPropZ": num(t2["two_proportion"]["z"], 2),
        "TwoG": num(t2["g_test"]["g"], 1),
        "TwoDayT": num(t2["day_level"]["t"], 2),
        "TwoDayP": pval(t2["day_level"]["p"]),
        "TwoAllChi": count(round(t2["all_rows"]["chi2"])),
        "TwoAllN": count(t2["all_rows"]["n"]),
        "ThreeEst": num(t3["primary"]["estimate"], 3),
        "ThreeCi": ci(t3["primary"]["ci"], 3),
        "ThreeT": num(t3["primary"]["t"], 1),
        "ThreeDf": count(t3["primary"]["df"]),
        "ThreeP": pval(t3["primary"]["p"]),
        "ThreeThreshold": num(MATTERS_DELTA_IC, 2),
        "ThreePre": num(t3["ic_pre"]["estimate"], 3),
        "ThreePreCi": ci(t3["ic_pre"]["ci"], 3),
        "ThreePost": num(t3["ic_post"]["estimate"], 3),
        "ThreePostCi": ci(t3["ic_post"]["ci"], 3),
        "ThreeRatio": num(t3["ic_post"]["estimate"] / t3["ic_pre"]["estimate"], 1),
        "ThreeFisherZ": num(t3["fisher_z"]["z"], 1),
        "ThreeSignDays": count(t3["sign_test"]["positive_days"]),
        "ThreeSignP": pval(t3["sign_test"]["p"]),
        "ThreeWilcoxonP": pval(t3["wilcoxon"]["p"]),
        "ThreePooledPre": num(t3["pooled_correlation"]["pre"]["r"], 3),
        "ThreePooledPost": num(t3["pooled_correlation"]["post"]["r"], 3),
        "ThreePooledPreT": num(t3["pooled_correlation"]["pre"]["t"], 1),
        "ThreePooledPostT": num(t3["pooled_correlation"]["post"]["t"], 1),
        "FourF": num(t4["primary"]["f"], 3),
        "FourCi": ci(t4["primary"]["ci"], 3),
        "FourDf": count(t4["primary"]["df1"]),
        "FourP": pval(t4["primary"]["p"]),
        "FourSdPre": num(t4["sd_pre"], 1),
        "FourSdPost": num(t4["sd_post"], 1),
        "FourKurtPre": num(t4["kurtosis_pre"], 0),
        "FourKurtPost": num(t4["kurtosis_post"], 0),
        "FourBf": count(round(t4["brown_forsythe"]["w"])),
        "FourBfP": pval(t4["brown_forsythe"]["p"]),
        "FourDayRatio": num(t4["day_level_log_ratio"]["ratio"], 2),
        "FourDayRatioCi": ci(t4["day_level_log_ratio"]["ratio_ci"], 2),
        "FourDayT": num(t4["day_level_log_ratio"]["t"], 1),
        "FourWinsF": num(t4["winsorised_f"]["f"], 3),
        "FourLow": num(MATTERS_VAR_RATIO[0], 2),
        "FourHigh": num(MATTERS_VAR_RATIO[1], 2),
        "FourDrop": pct(1 - t4["primary"]["f"], 0),
        "FiveN": count(t5["primary"]["n"]),
        "FiveEst": pct(t5["primary"]["estimate"]),
        "FiveCi": ci([100 * c for c in t5["primary"]["ci"]], 1),
        "FiveZ": num(t5["primary"]["z"], 1),
        "FiveP": pval(t5["primary"]["p"]),
        "FiveGross": pct(t5["gross_hit_rate"]),
        "FiveNet": num(t5["mean_net_bps"], 2),
        "FiveThreshold": pct(MATTERS_HIT_RATE, 0),
        "FiveDayT": num(t5["day_level"]["t"], 1),
        "BoundT": num(bounds["intervals"]["t"]["half_width"], 3),
        "BoundCheb": num(bounds["intervals"]["chebyshev"]["half_width"], 3),
        "BoundHoeff": num(bounds["intervals"]["hoeffding"]["half_width"], 2),
        "BoundChebRatio": num(bounds["intervals"]["chebyshev"]["width_vs_t"], 1),
        "BoundHoeffRatio": count(round(bounds["intervals"]["hoeffding"]["width_vs_t"])),
        "BoundTCi": ci([bounds["intervals"]["t"]["low"], bounds["intervals"]["t"]["high"]], 2),
        "BoundChebCi": ci([bounds["intervals"]["chebyshev"]["low"], bounds["intervals"]["chebyshev"]["high"]], 2),
        "BoundHoeffCi": ci([bounds["intervals"]["hoeffding"]["low"], bounds["intervals"]["hoeffding"]["high"]], 1),
        "BoundRange": num(bounds["bound"], 1),
    }
    m.update(extension_macros(ext, tests))
    m.update(learning_macros(learn))
    family = load("family")
    for key, word in zip(["T1", "T2", "T3", "T4", "T5", "T6"], ["One", "Two", "Three", "Four", "Five", "Six"]):
        m[f"Holm{word}"] = pval(family[key]["p_holm"])
    return m


def extension_macros(ext, tests):
    jumps = ext["placebo"]["jumps"]
    profile = ext["placebo"]["profile"]
    ic_at = dict(zip(profile["seconds"], profile["mean"]))
    rob = ext["robustness"]["groups"]
    pw = ext["power"]
    fm = ext["costs"]["final_minute"]
    sp = ext["costs"]["strongest_post"]
    be = ext["costs"]["breakeven"]
    fits = ext["target_fits"]
    flips = ext["side_flips"]
    cm = ext["count_min"]
    m = {
        "JumpReal": num(jumps["300"]["estimate"], 3),
        "JumpRealCi": ci(jumps["300"]["ci"], 3),
        "JumpLast": num(jumps["540"]["estimate"], 3),
        "JumpLastCi": ci(jumps["540"]["ci"], 3),
        "JumpFour": num(jumps["480"]["estimate"], 3),
        "JumpTwo": num(jumps["120"]["estimate"], 3),
        "JumpSevenMin": num(min(v["estimate"] for v in jumps.values()), 3),
        "IcBeforeBoundary": num(ic_at[290], 3),
        "IcAtBoundary": num(ic_at[300], 3),
        "IcLast": num(ic_at[540], 3),
        "IcMin": num(min(profile["mean"]), 3),
        "SensWith": num(ext["placebo"]["sensitivity"]["with_last_minute"]["estimate"], 3),
        "SensWithCi": ci(ext["placebo"]["sensitivity"]["with_last_minute"]["ci"], 3),
        "SensWithout": num(ext["placebo"]["sensitivity"]["without_last_minute"]["estimate"], 3),
        "SensWithoutCi": ci(ext["placebo"]["sensitivity"]["without_last_minute"]["ci"], 3),
        "RobPositive": count(ext["robustness"]["positive"]),
        "RobTotal": count(ext["robustness"]["total"]),
        "PermOneP": pval(ext["permutation_t1"]["p"]),
        "PermOneMax": num(ext["permutation_t1"]["null_max"], 3),
        "PermOneN": count(ext["permutation_t1"]["permutations"]),
        "FlipThreeP": pval(ext["sign_flip_t3"]["p"]),
        "FlipThreeMax": num(ext["sign_flip_t3"]["null_max"], 3),
        "FlipThreeN": count(ext["sign_flip_t3"]["draws"]),
        "FinalGross": num(fm["gross"], 2),
        "FinalHalf": num(fm["half_spread"], 2),
        "FinalNet": num(fm["estimate"], 2),
        "FinalNetCi": ci(fm["ci"], 2),
        "FinalT": num(fm["t"], 1),
        "FinalHit": pct(fm["hit_rate"]),
        "FinalFirst": num(fm["first_half"]["estimate"], 2),
        "FinalSecond": num(fm["second_half"]["estimate"], 2),
        "FinalSecondCi": ci(fm["second_half"]["ci"], 2),
        "StrongGross": num(sp["gross"], 2),
        "StrongHalf": num(sp["half_spread"], 2),
        "StrongNet": num(sp["estimate"], 2),
        "StrongNetCi": ci(sp["ci"], 2),
        "BreakPre": num(be["pre"]["gross"], 2),
        "BreakPost": num(be["post"]["gross"], 2),
        "TFitDf": num(fits["student_t"]["df"], 1),
        "TFitScale": num(fits["student_t"]["scale"], 1),
        "TFitChi": count(round(fits["student_t"]["gof"]["chi2"])),
        "TFitGain": num(fits["loglik_gain_per_row"], 3),
        "NormChi": count(round(fits["normal"]["gof"]["chi2"])),
        "FlipMean": num(flips["mean"], 2),
        "FlipVar": num(flips["variance"], 2),
        "FlipDispersion": num(flips["dispersion"], 2),
        "FlipZero": pct(flips["zero_share"]),
        "FlipZeroPoisson": pct(flips["poisson_zero_share"]),
        "FlipChi": count(round(flips["chi2"])),
        "FlipDf": count(flips["df"]),
        "MidOneEst": num(ext["mid_price"]["target_mid"]["edge"]["estimate"], 2),
        "MidOneCi": ci(ext["mid_price"]["target_mid"]["edge"]["ci"], 2),
        "MidOneT": num(ext["mid_price"]["target_mid"]["edge"]["t"], 1),
        "WapOneEst": num(ext["mid_price"]["target"]["edge"]["estimate"], 2),
        "MidOneRetained": pct(ext["mid_price"]["edge_retained"], 0),
        "MidFiveHit": pct(ext["mid_price"]["target_mid"]["hit_rate"]),
        "WapFiveHit": pct(ext["mid_price"]["target"]["hit_rate"]),
        "CmItems": count(cm["items"]),
        "CmCounters": count(cm["counters"]),
        "CmWidth": count(cm["width"]),
        "CmDepth": count(cm["depth"]),
        "CmWithin": pct(cm["share_within_bound"], 0),
        "CmMaxShare": pct(cm["max_error"] / cm["bound"], 0),
        "CmTop": count(cm["top_overlap"]),
        "CmMemory": pct(cm["counters"] / cm["items"], 0),
    }
    for key, word, digits in [("T1", "One", 2), ("T3", "Three", 2), ("T4", "Four", 3), ("T5", "Five", 2)]:
        m[f"Mde{word}"] = num(pw[key]["mde"], 3)
        m[f"Days{word}"] = count(pw[key]["days_for_threshold"])
        m[f"Thr{word}"] = num(pw[key]["threshold"], digits)
    m["MdeOneRatio"] = count(round(pw["T1"]["threshold"] / pw["T1"]["mde"]))
    rows = []
    for name, r in rob.items():
        rows.append(f"{name} & {r['pre']:.3f} & {r['post']:.3f} & {r['estimate']:+.3f} ({r['ci'][0]:.3f} to {r['ci'][1]:.3f}) \\\\")
    m["RobRows"] = "\n".join(rows)
    jrows = []
    for b, r in jumps.items():
        b = int(b)
        label = f"15:{50 + (b - 60) // 60}:00 to 15:{50 + b // 60}:00"
        jrows.append(f"{label} & {r['estimate']:+.3f} & {r['ci'][0]:.3f} to {r['ci'][1]:.3f} \\\\")
    m["JumpRows"] = "\n".join(jrows)
    return m


def learning_macros(learn):
    pr = learn["primary"]
    mid = learn["mid_check"]
    sel = learn["selective_trading"]
    sel_mid = mid["selective_trading_mid"]
    c2 = learn["c2st"]
    imp = list(learn["importance_share"].items())
    m = {
        "SixEst": num(pr["estimate"], 3),
        "SixCi": ci(pr["ci"], 3),
        "SixT": num(pr["t"], 1),
        "SixDf": count(pr["df"]),
        "SixP": pval(pr["p"]),
        "SixPerm": pval(learn["permutation_p"]),
        "SixThreshold": num(pr["matters_threshold"], 2),
        "SixMlp": num(pr["ic_model"], 3),
        "SixBase": num(pr["ic_base"], 3),
        "SixRatio": num(pr["ic_model"] / pr["ic_base"], 1),
        "SixTestDays": count(pr["n"]),
        "SixTrainRows": count(learn["split"]["train_rows"]),
        "SixTestRows": count(learn["split"]["test_rows"]),
        "SixFeatures": count(len(learn["split"]["features"])),
        "LgbmIc": num(learn["lgbm_vs_imbalance"]["ic_model"], 3),
        "LgbmGain": num(learn["lgbm_vs_imbalance"]["estimate"], 3),
        "LgbmGainCi": ci(learn["lgbm_vs_imbalance"]["ci"], 3),
        "MlpLgbm": num(learn["mlp_vs_lgbm"]["estimate"], 3),
        "MlpLgbmCi": ci(learn["mlp_vs_lgbm"]["ci"], 3),
        "GruIc": num(learn["gru_vs_snapshot_mlp"]["ic_model"], 3),
        "SnapIc": num(learn["gru_vs_snapshot_mlp"]["ic_base"], 3),
        "GruGain": num(learn["gru_vs_snapshot_mlp"]["estimate"], 3),
        "GruGainCi": ci(learn["gru_vs_snapshot_mlp"]["ci"], 3),
        "DmZeroT": num(learn["dm_vs_zero"]["t"], 1),
        "DmLinT": num(learn["dm_vs_linear"]["t"], 1),
        "DmLinP": pval(learn["dm_vs_linear"]["p"]),
        "SeedSdMlp": num(learn["seed_sd"]["mlp"], 4),
        "SeedSdLgbm": num(learn["seed_sd"]["lgbm"], 4),
        "SeedSdGru": num(learn["seed_sd"]["gru"], 4),
        "SeedRatio": count(round(pr["estimate"] / max(learn["seed_sd"]["mlp"], 1e-9))),
        "MidGain": num(mid["gain_on_mid"]["estimate"], 3),
        "MidGainCi": ci(mid["gain_on_mid"]["ci"], 3),
        "MidMlp": num(mid["gain_on_mid"]["ic_model"], 3),
        "MidBase": num(mid["gain_on_mid"]["ic_base"], 3),
        "MidShare": pct(mid["share_retained"], 0),
        "SelNet": num(sel["estimate"], 2),
        "SelNetCi": ci(sel["ci"], 2),
        "SelT": num(sel["t"], 1),
        "SelHit": pct(sel["hit_rate"]),
        "SelShare": pct(sel["share_traded"]),
        "SelTrades": count(sel["trades"]),
        "SelMidNet": num(sel_mid["estimate"], 2),
        "SelMidNetCi": ci(sel_mid["ci"], 2),
        "SelMidHit": pct(sel_mid["hit_rate"]),
        "SelMidT": num(sel_mid["t"], 1),
        "SelMidP": pval(sel_mid["p"]),
        "CtAcc": pct(c2["accuracy"]),
        "CtZ": num(c2["z"], 0),
        "CtN": count(c2["n"]),
        "CtDayCi": ci([100 * c for c in c2["day_level"]["ci"]], 1),
        "ImpFirst": imp[0][0].replace("_", " "),
        "ImpFirstShare": pct(imp[0][1], 0),
        "ImpSecond": imp[1][0].replace("_", " "),
        "ImpThird": imp[2][0].replace("_", " "),
        "ImpImbShare": pct(learn["importance_share"]["signed_imbalance"], 0),
    }
    return m


def write_numbers(path=RESULTS / "numbers.tex"):
    lines = [f"\\newcommand{{\\n{k}}}{{{v}}}" for k, v in macros().items()]
    path.write_text("\n".join(lines) + "\n")
    return path

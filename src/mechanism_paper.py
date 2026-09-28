"""Numbers, table and figure for the merged letter's mechanism section (paper/PLAN_mechanism.md: T2, L1, L2).

Inputs: output/mechanism/results.json and formation_beta_panel.csv (from src/crsp_mechanism.py, CRSP-derived,
portfolio level only) and the letter's own real-time pipeline (src/realtime.py).
Run: python src/mechanism_paper.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy.stats import norm  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime import (load_factors, load_monthly, monthly_panel, realtime_managed, CAP, GEN, PAPER_IMG,  # noqa: E402
                      num)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MECH = os.path.join(ROOT, "output", "mechanism")
KAPPA_PREMIUM = 4.0
SPLIT = pd.Timestamp("2016-12-31")


def clark_west(f, a, b, w=None, lags=6):
    f, a, b = (np.asarray(z, float) for z in (f, a, b))
    w = np.ones(len(f)) if w is None else np.asarray(w, float)
    adj = w * ((f - b) ** 2 - ((f - a) ** 2 - (a - b) ** 2))
    t = float(sm.OLS(adj, np.ones(len(adj))).fit(cov_type="HAC", cov_kwds={"maxlags": lags}).tvalues[0])
    return float(norm.sf(t))


def r2_os(f, a, b, w=None):
    f, a, b = (np.asarray(z, float) for z in (f, a, b))
    w = np.ones(len(f)) if w is None else np.asarray(w, float)
    return float(1 - np.sum(w * (f - a) ** 2) / np.sum(w * (f - b) ** 2))


def main():
    res = json.load(open(os.path.join(MECH, "results.json")))
    panel = pd.read_csv(os.path.join(MECH, "formation_beta_panel.csv"), index_col=0, parse_dates=True)
    daily, monthly = load_factors(), load_monthly()
    mom = monthly_panel(daily["Mom"], monthly["Mom"])
    mkt = monthly_panel(daily["Mkt-RF"], monthly["Mkt-RF"])
    rt = realtime_managed(mom, CAP)
    out = {}

    # L1: where the covariance channel comes from
    bp_hold = panel["beta_pred"].shift(1).reindex(rt.index)           # predicted at the end of m-1, held in m
    d = rt.assign(bp=bp_hold).dropna(subset=["bp", "w", "f"])
    wm, fm = d["w"].mean(), d["f"].mean()
    contrib = (d["w"] - wm) * (d["f"] - fm)
    neg = d["bp"] < 0
    cov_total = contrib.sum() / len(d)
    out["L1"] = {"months": len(d), "first": str(d.index.min().date()), "last": str(d.index.max().date()),
                 "share_months_neg": float(neg.mean()), "share_cov_neg": float(contrib[neg].sum() / len(d) / cov_total),
                 "cov_ann_pct": float(cov_total * 12 * 100),
                 "mean_f_neg_ann": float(d.loc[neg, "f"].mean() * 1200), "mean_f_pos_ann": float(d.loc[~neg, "f"].mean() * 1200),
                 "mean_w_neg": float(d.loc[neg, "w"].mean()), "mean_w_pos": float(d.loc[~neg, "w"].mean())}

    # L2: parameter-free mean forecast
    comp = (panel["beta_pred"] * mkt["sig"].reindex(panel.index)).dropna()
    f_all = monthly["Mom"]
    idx = comp.index[(comp.index >= pd.Timestamp("1973-06-30"))]
    mu_bar = f_all.expanding().mean().reindex(idx)
    cbar = comp.expanding().mean().reindex(idx)
    mu_hat = mu_bar + KAPPA_PREMIUM * (comp.reindex(idx) - cbar)
    f_next = f_all.shift(-1).reindex(idx)
    wts = 1.0 / mom["sig"].reindex(idx)
    e = pd.DataFrame({"f": f_next, "a": mu_hat, "b": mu_bar, "w": wts}).dropna()
    out["L2"] = {}
    for per, x in (("full", e), ("pre", e[e.index < SPLIT]), ("post", e[e.index >= SPLIT])):
        out["L2"][per] = {"months": len(x), "r2_unw": r2_os(x.f, x.a, x.b), "r2_w": r2_os(x.f, x.a, x.b, x.w),
                          "cw_p_unw": clark_west(x.f, x.a, x.b), "cw_p_w": clark_west(x.f, x.a, x.b, x.w)}

    # figure: realized against predicted beta, by decile of the prediction (French's momentum, full CRSP sample)
    p = panel.dropna(subset=["beta_pred", "french_mom_next", "R_M_next"]).copy()
    p["bin"] = pd.qcut(p["beta_pred"], 10, labels=False)
    pts = []
    for _, g in p.groupby("bin"):
        r = sm.OLS(g["french_mom_next"], sm.add_constant(g["R_M_next"])).fit()
        pts.append((g["beta_pred"].mean(), r.params.iloc[1], r.bse.iloc[1]))
    pts = np.array(pts)
    t2f = res["full"]["T2_french"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.errorbar(pts[:, 0], pts[:, 1], yerr=1.96 * pts[:, 2], fmt="o", capsize=3, label="realized, by decile of the prediction")
    g = np.linspace(pts[:, 0].min() - 0.1, pts[:, 0].max() + 0.1, 50)
    ax.plot(g, g, "k--", lw=0.9, label="45-degree line (prediction exact)")
    ax.plot(g, t2f["d"] + t2f["c"] * g, color="C1", lw=1.2, label=f"fitted: slope {t2f['c']:.2f}, level {t2f['d']:+.2f}")
    ax.set_xlabel("predicted beta (Proposition 1, no fitted parameters)")
    ax.set_ylabel("realized market beta next month")
    ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(PAPER_IMG, "fig_formation_beta.png"), dpi=200)
    plt.close(fig)
    out["figure_bins"] = pts.tolist()

    json.dump(out, open(os.path.join(MECH, "paper_numbers.json"), "w"), indent=1)
    write_tex(res, out)
    print(json.dumps(out, indent=1, default=float))


def write_tex(res, out):
    m = []
    add = lambda k, v: m.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    g = res["replication_gate"]
    add("RepCorr", f"{g['corr_vw_vs_french']:.4f}")
    add("RepMonths", f"{g['months']:,}")
    add("CrspEnd", pd.Timestamp(g["crsp_last_month"]).strftime("%B %Y"))
    rows = []
    labels = {"T2_ew": "Equal-weighted", "T2_french": "French's momentum"}
    for s, sname in (("full", "1929 to 2025"), ("from_1963", "1963 to 2025")):
        for k in ("T2_ew", "T2_french"):
            t = res[s][k]
            rows.append(f"{sname} & {labels[k]} & {num(t['c'])} ({num(t['c_se'])}) & {num(t['d'])} ({num(t['d_se'])}) & "
                        f"{t['wald_c1_d0_p']:.3f} & {num(t['r2_gain_over_market'])} & {num(t['share_of_gm'])} & {t['months']} \\\\")
    open(os.path.join(GEN, "table_mechanism.tex"), "w").write("\n".join(rows) + "\n")
    t = res["full"]["T2_french"]
    add("MechCFull", num(t["c"])); add("MechCSeFull", num(t["c_se"])); add("MechDFull", num(t["d"]))
    add("MechShareFull", f"{100 * t['share_of_gm']:.0f}"); add("MechRtwoFull", f"{100 * t['r2_gain_over_market']:.0f}")
    t = res["from_1963"]["T2_french"]
    add("MechCPost", num(t["c"])); add("MechCSePost", num(t["c_se"])); add("MechDPost", num(t["d"]))
    add("MechWaldPost", f"{t['wald_c1_d0_p']:.3f}")
    t = res["from_1963"]["T2_ew"]
    add("MechCEwPost", num(t["c"])); add("MechCSeEwPost", num(t["c_se"]))
    t1 = res["full"]["T1"]
    add("MechTOneSlope", num(t1["slope"])); add("MechTOneCorr", num(t1["corr"]))
    l1 = out["L1"]
    add("LOneShareMonths", f"{100 * l1['share_months_neg']:.0f}"); add("LOneShareCov", f"{100 * l1['share_cov_neg']:.0f}")
    add("LOneMeanFNeg", num(l1["mean_f_neg_ann"], 1)); add("LOneMeanFPos", num(l1["mean_f_pos_ann"], 1))
    add("LOneMeanWNeg", num(l1["mean_w_neg"])); add("LOneMeanWPos", num(l1["mean_w_pos"]))
    l2 = out["L2"]
    for per, P in (("pre", "Pre"), ("post", "Post"), ("full", "Full")):
        add(f"LTwoRUnw{P}", num(100 * l2[per]["r2_unw"], 2)); add(f"LTwoRW{P}", num(100 * l2[per]["r2_w"], 2))
        add(f"LTwoCwUnw{P}", f"{l2[per]['cw_p_unw']:.3f}"); add(f"LTwoCwW{P}", f"{l2[per]['cw_p_w']:.3f}")
    open(os.path.join(GEN, "mechanism_numbers.tex"), "w").write("\n".join(m) + "\n")


if __name__ == "__main__":
    main()

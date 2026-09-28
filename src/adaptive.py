"""
adaptive.py - adaptive learning rates for volatility management (pre-registered in paper/PLAN_adaptive.md).

The current rule (realtime.py) has three estimators with implicit learning rates:
  variance signal   prior month's realized variance (equal weight on ~21 days)
  scale constant    expanding window since 1963 (step size 1/t, so it stops learning)
  combination       expanding mean and covariance (step size 1/t)
This module replaces each with estimators whose learning rate is set by a half-life, runs the
fixed 24-variant grid for all six factors, and applies the three pre-registered selection rules.

Signals (forecast of month m's variance, made at the end of month m-1 from daily data only)
  S0  realized variance of month m-1 (current)
  S1  exponentially weighted daily variance, half-life 5 trading days
  S2  half-life 21 trading days
  S3  half-life 63 trading days
  S4  HAR (Corsi 2009) in logs: next-month variance on last day / last week / last month, real-time OLS
  S5  realized variance from the last 13 complete weekly returns (the weekly-averaging idea)
Scale constant c = sd_w(f) / sd_w(f / S) with weights
  B0  equal (expanding, current);  B1, B2, B3  exponential, half-life 36, 60, 120 months
Combination weights (mean, covariance of original and managed): C0 expanding, C1/C2 half-life 60/120 months

Run: python src/adaptive.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realtime as RT  # noqa: E402
from realtime import (BURN_IN, CAP, KAPPA, MIN_COMBO, POST_START, SEED, lw_test, sharpe, holm,  # noqa: E402
                      max_dd_equal_vol, num, pfmt, macro, load_monthly, monthly_panel, realtime_managed)
from vol_managed_factors import OUT, FIG, FACTORS, load_factors  # noqa: E402

SIGNALS = ["S0", "S1", "S2", "S3", "S4", "S5"]
SIGNAL_LABEL = {"S0": "Prior-month RV", "S1": "EWMA, 5-day half-life", "S2": "EWMA, 21-day half-life",
                "S3": "EWMA, 63-day half-life", "S4": "HAR", "S5": "13-week weekly RV"}
BASELINES = {"B0": None, "B1": 36, "B2": 60, "B3": 120}
BASE_LABEL = {"B0": "expanding", "B1": "3-yr half-life", "B2": "5-yr half-life", "B3": "10-yr half-life"}
COMBOS = {"C0": None, "C1": 60, "C2": 120}
EWMA_HL = {"S1": 5, "S2": 21, "S3": 63}
SELECT_WINDOW = 120              # trailing months for real-time selection (rule 2)
SPA_BOOT = 4999
SPA_BLOCK = 12
GEN = RT.GEN


# ---------------------------------------------------------------- signals

def signals(daily: pd.Series, months: pd.DatetimeIndex, rv: pd.Series) -> pd.DataFrame:
    """Each column: variance forecast made at the end of month m (used for month m+1)."""
    r = daily.dropna()
    r2 = r ** 2
    out = pd.DataFrame(index=months, dtype=float)
    out["S0"] = rv.reindex(months)

    def at_month_end(series: pd.Series) -> pd.Series:
        s = series.groupby(series.index.to_period("M")).last()
        s.index = s.index.to_timestamp("M")
        return s.reindex(months)

    for key, hl in EWMA_HL.items():
        ew = r2.ewm(halflife=hl, adjust=False).mean()
        out[key] = at_month_end(ew) * 21

    # HAR components at each month end (daily, weekly, monthly), real-time log regression
    comp_d = at_month_end(r2) * 21
    comp_w = at_month_end(r2.rolling(5).mean()) * 21
    comp_m = at_month_end(r2.rolling(21).mean()) * 21
    X = np.log(np.column_stack([comp_d, comp_w, comp_m]).clip(min=1e-10))
    y_next = np.log(rv.reindex(months).shift(-1).values.clip(min=1e-10))    # target: next month's RV
    har = np.full(len(months), np.nan)
    for t in range(61, len(months)):
        # pairs (X_s, log RV_{s+1}) with s + 1 <= t are known at the end of month t: s = 0..t-1
        Xs, ys = X[:t], y_next[:t]
        keep = np.isfinite(ys) & np.isfinite(Xs).all(axis=1)
        A = np.column_stack([np.ones(keep.sum()), Xs[keep]])
        beta, *_ = np.linalg.lstsq(A, ys[keep], rcond=None)
        har[t] = np.exp(beta[0] + X[t] @ beta[1:])
    out["S4"] = har

    # weekly returns (calendar weeks ending Friday), compounded from daily; complete weeks only
    wk = (1 + r).groupby(r.index.to_period("W-FRI")).prod() - 1
    wk_end = wk.index.end_time.normalize()
    s5 = np.full(len(months), np.nan)
    for i, m in enumerate(months):
        done = wk.values[wk_end <= m]
        if len(done) >= 13:
            s5[i] = np.sum(done[-13:] ** 2) * 21 / 65
    out["S5"] = s5
    return out


# ---------------------------------------------------------------- construction

def _wsd(x: np.ndarray, w: np.ndarray) -> float:
    mu = np.sum(w * x) / np.sum(w)
    return float(np.sqrt(np.sum(w * (x - mu) ** 2) / np.sum(w)))


def managed(f: np.ndarray, sig: np.ndarray, index: pd.DatetimeIndex, halflife: float | None,
            cap: float = CAP) -> pd.DataFrame:
    """Exposure w_m = min(c_{m-1} / S_{m-1}, cap); c from data through m-1, weighted by half-life."""
    s_lag = np.r_[np.nan, sig[:-1]]                 # forecast made at end of m-1, used in month m
    n = len(f)
    w = np.full(n, np.nan)
    for t in range(BURN_IN - 1, n):
        past = np.arange(1, t)
        ok = np.isfinite(s_lag[past]) & (s_lag[past] > 0)
        p = past[ok]
        if len(p) < 24 or not np.isfinite(s_lag[t]) or s_lag[t] <= 0:
            continue
        wts = np.ones(len(p)) if halflife is None else 0.5 ** ((t - 1 - p) / halflife)
        if halflife is None:
            c = np.std(f[p], ddof=1) / np.std(f[p] / s_lag[p], ddof=1)
        else:
            c = _wsd(f[p], wts) / _wsd(f[p] / s_lag[p], wts)
        w[t] = min(c / s_lag[t], cap)
    df = pd.DataFrame({"f": f, "w": w}, index=index)
    df["turn"] = df["w"].diff().abs()
    df = df.iloc[BURN_IN:].copy()
    df["gross"] = df["w"] * df["f"]
    df["net"] = df["gross"] - KAPPA * df["turn"]
    return df


def combination(rt: pd.DataFrame, halflife: float | None, cap: float = CAP) -> pd.Series:
    """Real-time mean-variance mix of original and net managed factor; weights with a half-life."""
    f, net, w = rt["f"].values, rt["net"].values, rt["w"].values
    n = len(f)
    e = np.full(n, np.nan)
    for t in range(MIN_COMBO - 1, n):
        X = np.column_stack([f[:t], net[:t]])
        ok = np.isfinite(X).all(axis=1)
        X = X[ok]
        if len(X) < MIN_COMBO - 1 or not np.isfinite(w[t]):
            continue
        age = (np.arange(t)[ok])
        wts = np.ones(len(X)) if halflife is None else 0.5 ** ((t - 1 - age) / halflife)
        wts = wts / wts.sum()
        mu = wts @ X
        D = X - mu
        cov = (D * wts[:, None]).T @ D
        ab = np.linalg.solve(cov, mu)
        port = X @ ab
        s = _wsd(X[:, 0], wts) / _wsd(port, wts)
        e[t] = np.clip(s * (ab[0] + ab[1] * w[t]), -cap, cap)
    turn = np.abs(np.diff(e, prepend=np.nan))
    combo = e * f - KAPPA * turn
    combo[:MIN_COMBO] = np.nan
    return pd.Series(combo, index=rt.index)


def realtime_selection(variants: dict, window: int = SELECT_WINDOW) -> pd.DataFrame:
    """Each month hold the variant with the best trailing-window net Sharpe; pay costs on actual exposure changes."""
    keys = list(variants)
    idx = variants[keys[0]].index
    nets = np.column_stack([variants[k]["net"].values for k in keys])
    ws = np.column_stack([variants[k]["w"].values for k in keys])
    f = variants[keys[0]]["f"].values
    n = len(idx)
    chosen = np.full(n, -1)
    held_w = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    for t in range(window, n):
        past = nets[t - window: t]
        sd = past.std(axis=0, ddof=1)
        sr = np.where(sd > 0, past.mean(axis=0) / sd, -np.inf)
        sr[~np.isfinite(sr)] = -np.inf
        k = int(np.argmax(sr))
        chosen[t] = k
        held_w[t] = ws[t, k]
        prev = held_w[t - 1] if np.isfinite(held_w[t - 1]) else ws[t - 1, k]
        ret[t] = ws[t, k] * f[t] - KAPPA * abs(ws[t, k] - prev)
    return pd.DataFrame({"net": ret, "w": held_w, "chosen": [keys[c] if c >= 0 else None for c in chosen], "f": f},
                        index=idx)


# ---------------------------------------------------------------- Hansen (2005) SPA

def spa_test(bench: np.ndarray, models: np.ndarray, rng: np.random.Generator) -> dict:
    """Hansen's consistent SPA p-value that no model beats the benchmark in Sharpe ratio.

    Loss differential d_kt = r_kt / sd(r_k) - r_0t / sd(r_0), so E[d_k] is the monthly Sharpe
    difference. Stationary bootstrap (Politis-Romano), mean block length SPA_BLOCK.
    """
    keep = np.isfinite(bench) & np.isfinite(models).all(axis=1)
    b, M = bench[keep], models[keep]
    T, K = M.shape
    d = M / M.std(axis=0, ddof=1) - (b / b.std(ddof=1))[:, None]
    dbar = d.mean(axis=0)
    # stationary bootstrap indices
    p = 1.0 / SPA_BLOCK
    idx = np.empty((SPA_BOOT, T), dtype=int)
    idx[:, 0] = rng.integers(0, T, SPA_BOOT)
    newblock = rng.random((SPA_BOOT, T)) < p
    jumps = rng.integers(0, T, (SPA_BOOT, T))
    for t in range(1, T):
        idx[:, t] = np.where(newblock[:, t], jumps[:, t], (idx[:, t - 1] + 1) % T)
    dstar = d[idx].mean(axis=1)                                   # (B, K)
    omega = np.sqrt(T) * dstar.std(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, np.inf)
    stat = max(float(np.max(np.sqrt(T) * dbar / omega)), 0.0)
    thresh = -np.sqrt(2 * np.log(np.log(T)))
    g = np.where(np.sqrt(T) * dbar / omega >= thresh, dbar, 0.0)
    z = np.sqrt(T) * (dstar - g) / omega
    tstar = np.maximum(z.max(axis=1), 0.0)
    return {"stat": stat, "p": float(np.mean(tstar >= stat)), "best": int(np.argmax(dbar)), "T": T}


# ---------------------------------------------------------------- main

def main() -> None:
    os.makedirs(GEN, exist_ok=True)
    daily, monthly = load_factors(), load_monthly()
    grid_rows, sel_rows, stress_rows, combo_rows, spa_rows = [], [], [], [], []
    store = {}
    for fac in FACTORS:
        panel = monthly_panel(daily[fac], monthly[fac])
        sig = signals(daily[fac], panel.index, panel["rv"])
        f = panel["ret"].values
        variants = {}
        for s in SIGNALS:
            for b, hl in BASELINES.items():
                variants[f"{s}-{b}"] = managed(f, sig[s].values, panel.index, hl)
        cur = variants["S0-B0"]
        # sanity: S0-B0 must reproduce the canonical rule and S2-B0 the adopted main specification exactly
        ref = realtime_managed(panel, CAP, signal="rv")
        assert np.allclose(cur["w"].values, ref["w"].values, equal_nan=True), "S0-B0 differs from realtime.py"
        ref2 = realtime_managed(panel, CAP)
        assert np.allclose(variants["S2-B0"]["w"].values, ref2["w"].values, equal_nan=True), "S2-B0 differs"
        pre_m, post_m = cur.index < POST_START, cur.index >= POST_START
        for key, v in variants.items():
            grid_rows.append({"factor": fac, "variant": key, "signal": key[:2], "baseline": key[3:],
                              "sr_full": sharpe(v["net"]), "sr_pre": sharpe(v["net"][pre_m]),
                              "sr_post": sharpe(v["net"][post_m]), "turn": v["turn"].mean(),
                              "avg_w_post": v["w"][post_m].mean()})
        store[fac] = (variants, cur, sig)

        # rule 1: select on 1973-2016 only, evaluate after
        pre_sr = {k: sharpe(v["net"][pre_m]) for k, v in variants.items()}
        best = max(pre_sr, key=pre_sr.get)
        bv = variants[best]
        t_cur = lw_test(bv["net"][post_m].values, cur["net"][post_m].values)
        t_org = lw_test(bv["net"][post_m].values, bv["f"][post_m].values)
        # rule 2: real-time selection
        rs = realtime_selection(variants)
        valid = rs["net"].notna()
        t2_full = lw_test(rs["net"][valid].values, cur["net"][valid].values)
        post_v = valid & (rs.index >= POST_START)
        t2_post = lw_test(rs["net"][post_v].values, cur["net"][post_v].values)
        switches = int((rs["chosen"][valid] != rs["chosen"][valid].shift()).sum() - 1)
        store[fac] = store[fac] + (rs, best)
        sel_rows.append({"factor": fac, "rule1_variant": best, "rule1_sr_pre": pre_sr[best],
                         "cur_sr_pre": pre_sr["S0-B0"], "rule1_sr_post": sharpe(bv["net"][post_m]),
                         "cur_sr_post": sharpe(cur["net"][post_m]), "orig_sr_post": sharpe(cur["f"][post_m]),
                         "rule1_p_vs_cur": t_cur["p"], "rule1_p_vs_orig": t_org["p"],
                         "rule2_sr": sharpe(rs["net"][valid]), "cur_sr_same": sharpe(cur["net"][valid]),
                         "rule2_p_vs_cur": t2_full["p"], "rule2_sr_post": sharpe(rs["net"][post_v]),
                         "rule2_p_post": t2_post["p"], "rule2_start": rs.index[valid][0], "rule2_switches": switches,
                         "rule2_most_common": rs["chosen"][valid].value_counts().index[0]})
        # rule 3: SPA over the 24 variants against the current version
        keys = [k for k in variants if k != "S0-B0"]
        M_full = np.column_stack([variants[k]["net"].values for k in keys])
        for per, mask in (("full", np.ones(len(cur), bool)), ("post", post_m)):
            res = spa_test(cur["net"].values[mask], M_full[mask], np.random.default_rng(SEED))
            spa_rows.append({"factor": fac, "period": per, "p": res["p"], "best": keys[res["best"]]})
        # combination learning rates on the current managed factor and on the rule-1 variant
        for base_key, base_v in (("S0-B0", cur), (best, bv)):
            for ck, hl in COMBOS.items():
                cb = combination(base_v, hl)
                combo_rows.append({"factor": fac, "managed": base_key, "combo": ck, "sr_full": sharpe(cb),
                                   "sr_post": sharpe(cb[cb.index >= POST_START]),
                                   "sr_orig_same": sharpe(np.where(np.isnan(cb.values), np.nan, cur["f"].values))})
        # stress windows
        for win, (a, z) in {"2009": ("2009-03-01", "2009-05-31"), "2020": ("2020-02-01", "2020-04-30")}.items():
            for k in dict.fromkeys(("S0-B0", "S1-B0", "S2-B0", "S5-B0", best)):
                v = variants[k]
                m = (v.index >= a) & (v.index <= z)
                stress_rows.append({"factor": fac, "window": win, "variant": k,
                                    "exposures": ", ".join(f"{x:.2f}" for x in v["w"][m]),
                                    "factor_ret": ", ".join(f"{x * 100:.1f}" for x in v["f"][m]),
                                    "net_sum": float(v["net"][m].sum() * 100)})

    grid = pd.DataFrame(grid_rows)
    sel = pd.DataFrame(sel_rows).set_index("factor")
    for col in ("rule1_p_vs_cur", "rule2_p_vs_cur", "rule2_p_post"):
        sel[col + "_holm"] = pd.Series(holm(sel[col].to_dict()))
    spa = pd.DataFrame(spa_rows)
    combos = pd.DataFrame(combo_rows)
    stress = pd.DataFrame(stress_rows)
    grid.round(4).to_csv(os.path.join(OUT, "adaptive_grid.csv"), index=False)
    sel.round(4).to_csv(os.path.join(OUT, "adaptive_selection.csv"))
    spa.round(4).to_csv(os.path.join(OUT, "adaptive_spa.csv"), index=False)
    combos.round(4).to_csv(os.path.join(OUT, "adaptive_combination.csv"), index=False)
    stress.to_csv(os.path.join(OUT, "adaptive_stress.csv"), index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    for fac in FACTORS:
        g = grid[grid.factor == fac]
        print(f"\n{fac}: net Sharpe by signal (rows) x scale constant (cols)")
        for per in ("sr_pre", "sr_post"):
            print(per)
            print(g.pivot(index="signal", columns="baseline", values=per).to_string())
    print("\nSelection rules:\n", sel.drop(columns=["rule2_start"]).to_string())
    print("\nSPA:\n", spa.to_string(index=False))
    print("\nCombination learning rates:\n", combos.to_string(index=False))
    print("\nStress windows:\n", stress[stress.factor.isin(["Mom", "Mkt-RF"])].to_string(index=False))
    make_figures(grid, store)
    write_paper_outputs(grid, sel, spa, combos, store)


def write_paper_outputs(grid, sel, spa, combos, store) -> None:
    """Table 5 rows and every in-text number of the learning-rate section."""
    L = []
    for fac in FACTORS:
        r = sel.loc[fac]
        sp = spa[spa.factor == fac].set_index("period")
        L.append(f"{RT.NAMES[fac]} & {r.rule1_variant} & {num(r.orig_sr_post)} & {num(r.cur_sr_post)} & "
                 f"{num(r.rule1_sr_post)} & {pfmt(r.rule1_p_vs_cur)} & {num(r.cur_sr_same)} & {num(r.rule2_sr)} & "
                 f"{pfmt(r.rule2_p_vs_cur)} & {pfmt(sp.loc['full', 'p'])} & {pfmt(sp.loc['post', 'p'])} \\\\")
    open(os.path.join(GEN, "table_adaptive.tex"), "w").write("\n".join(L) + "\n")

    g = grid.set_index(["factor", "variant"])
    mo = sel.loc["Mom"]
    m = macro("AdRuleOneMom", mo.rule1_variant)
    m += macro("AdMomPreSel", num(mo.rule1_sr_pre)) + macro("AdMomPreCur", num(mo.cur_sr_pre))
    m += macro("AdMomPostSel", num(mo.rule1_sr_post)) + macro("AdMomPostCur", num(mo.cur_sr_post))
    m += macro("AdMomPostSelP", pfmt(mo.rule1_p_vs_cur))
    m += macro("AdMomRuleTwo", num(mo.rule2_sr)) + macro("AdMomRuleTwoCur", num(mo.cur_sr_same))
    m += macro("AdMomRuleTwoP", pfmt(mo.rule2_p_vs_cur)) + macro("AdMomSwitches", str(int(mo.rule2_switches)))
    m += macro("AdMomRuleTwoStart", f"{mo.rule2_start:%B %Y}")
    sm = spa[spa.factor == "Mom"].set_index("period")
    m += macro("AdMomSpaFull", pfmt(sm.loc["full", "p"])) + macro("AdMomSpaPost", pfmt(sm.loc["post", "p"]))
    m += macro("AdMomSpaPostBest", sm.loc["post", "best"])
    # block-length sensitivity of the post-2016 SPA p-value for momentum
    variants, cur = store["Mom"][0], store["Mom"][1]
    keys = [k for k in variants if k != "S0-B0"]
    M = np.column_stack([variants[k]["net"].values for k in keys])
    post = cur.index >= POST_START
    global SPA_BLOCK
    saved, ps = SPA_BLOCK, []
    for blk in (6, 12, 24):
        SPA_BLOCK = blk
        ps.append(spa_test(cur["net"].values[post], M[post], np.random.default_rng(SEED))["p"])
    SPA_BLOCK = saved
    m += macro("AdMomSpaPostMin", pfmt(min(ps))) + macro("AdMomSpaPostMax", pfmt(max(ps)))
    m += macro("AdMomPostSFive", num(g.loc[("Mom", "S5-B0"), "sr_post"]))
    m += macro("AdMomPreSFive", num(g.loc[("Mom", "S5-B0"), "sr_pre"]))
    m += macro("AdMomPreBOne", num(g.loc[("Mom", "S0-B1"), "sr_pre"])) + macro("AdMomPostBOne", num(g.loc[("Mom", "S0-B1"), "sr_post"]))
    n_s1_worse = sum(g.loc[(f, "S1-B0"), "sr_pre"] < g.loc[(f, "S0-B0"), "sr_pre"] for f in FACTORS)
    m += macro("AdSOneWorsePre", str(int(n_s1_worse)))
    c = combos[(combos.factor == "Mom") & (combos.managed == "S0-B0")].set_index("combo")
    m += macro("AdMomComboCZero", num(c.loc["C0", "sr_full"])) + macro("AdMomComboCOne", num(c.loc["C1", "sr_full"]))
    m += macro("AdMomComboCTwo", num(c.loc["C2", "sr_full"]))
    n_c_worse = sum(
        (combos[(combos.factor == f) & (combos.managed == "S0-B0")].set_index("combo").loc[["C1", "C2"], "sr_full"]
         < combos[(combos.factor == f) & (combos.managed == "S0-B0")].set_index("combo").loc["C0", "sr_full"]).all()
        for f in FACTORS)
    m += macro("AdComboWorseCount", str(int(n_c_worse)))
    mk = store["Mkt-RF"][0]
    feb = pd.Timestamp("2020-02-29")
    win = lambda v: v[(v.index >= "2020-02-01") & (v.index <= "2020-04-30")]
    m += macro("AdMktFebCur", num(mk["S0-B0"].loc[feb, "w"])) + macro("AdMktFebFast", num(mk["S1-B0"].loc[feb, "w"]))
    m += macro("AdMktCovidCur", num(win(mk["S0-B0"])["net"].sum() * 100, 1))
    m += macro("AdMktCovidFast", num(win(mk["S1-B0"])["net"].sum() * 100, 1))
    apr = [mk[f"{sg}-B0"].loc[pd.Timestamp("2020-04-30"), "w"] for sg in SIGNALS]
    m += macro("AdMktAprMax", num(max(apr)))
    hs = spa[spa.factor == "HML"].set_index("period")
    m += macro("AdHmlSpaPost", pfmt(hs.loc["post", "p"]))
    hv = sel.loc["HML"]
    m += macro("AdHmlRuleOne", hv.rule1_variant) + macro("AdHmlPostSel", num(hv.rule1_sr_post))
    m += macro("AdHmlPostCur", num(hv.cur_sr_post)) + macro("AdHmlPostSelP", pfmt(hv.rule1_p_vs_cur))
    open(os.path.join(GEN, "adaptive_numbers.tex"), "w").write(m)


def make_figures(grid, store) -> None:
    # Figure: Sharpe difference vs current version across the grid, pre and post, all factors
    fig, axes = plt.subplots(2, 6, figsize=(15, 6.2), sharex=True, sharey=True)
    vmax = 0.4
    for j, fac in enumerate(FACTORS):
        g = grid[grid.factor == fac]
        for i, (per, lab) in enumerate((("sr_pre", "1973 to 2016"), ("sr_post", "2017 to 2026"))):
            piv = g.pivot(index="signal", columns="baseline", values=per)
            diff = piv - piv.loc["S0", "B0"]
            ax = axes[i, j]
            im = ax.imshow(diff.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
            for (a, b), val in np.ndenumerate(diff.values):
                ax.text(b, a, f"{val:+.2f}".replace("0.", "."), ha="center", va="center", fontsize=7.5)
            ax.set_xticks(range(4))
            ax.set_xticklabels(["B0", "B1", "B2", "B3"], fontsize=7)
            ax.set_yticks(range(6))
            ax.set_yticklabels(SIGNALS, fontsize=7)
            if i == 0:
                ax.set_title(RT.NAMES[fac], fontsize=9)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
    for ax in axes.ravel():
        ax.add_patch(plt.Rectangle((-0.5, 1.5), 1, 1, fill=False, ec="black", lw=1.6))   # adopted S2-B0
    fig.colorbar(im, ax=axes, shrink=0.8, label="net Sharpe minus canonical rule (S0-B0)")
    fig.savefig(os.path.join(FIG, "fig_adaptive_grid.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(RT.PAPER_IMG, "fig_adaptive_grid.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure: exposure paths in the two stress windows the idea is about (momentum and market)
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.2))
    for i, fac in enumerate(("Mom", "Mkt-RF")):
        variants = store[fac][0]
        for j, (a, z) in enumerate((("2008-06-01", "2010-06-30"), ("2019-10-01", "2021-06-30"))):
            ax = axes[i, j]
            for k, col in (("S0-B0", "#8a8a8e"), ("S1-B0", "#c0392b"), ("S2-B0", "#1f4e79"), ("S5-B0", "#2e8b57")):
                v = variants[k]
                m = (v.index >= a) & (v.index <= z)
                ax.plot(v.index[m], v["w"][m], lw=1.4, color=col, label=f"{k[:2]}: {SIGNAL_LABEL[k[:2]]}")
            ax2 = ax.twinx()
            v = variants["S0-B0"]
            m = (v.index >= a) & (v.index <= z)
            ax2.bar(v.index[m], v["f"][m] * 100, width=20, alpha=0.25, color="#555")
            ax2.set_ylabel("factor return, %", fontsize=7)
            ax.set_ylim(0, 1.6)
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.tick_params(axis="x", labelsize=7)
            ax.set_title(f"{RT.NAMES[fac]}, {a[:4]} to {z[:4]}", fontsize=9)
            ax.set_ylabel("exposure", fontsize=8)
            ax.grid(alpha=0.3)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIG, "fig_adaptive_stress.png"), dpi=200)
    fig.savefig(os.path.join(RT.PAPER_IMG, "fig_adaptive_stress.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

"""
realtime.py - does volatility management survive when positions use only past data?

The original study (vol_managed_factors.py) follows Moreira and Muir (2017): the timing
signal 1/RV[m-1] is known in advance, but the scale constant is fitted on the full sample,
exposure is uncapped, and the spanning alpha is an in-sample statistic. Cederburg,
O'Doherty, Wang and Yan (2020) show that these choices matter. This module rebuilds the
Fama-French five factors and momentum so that every position uses only information
available at the end of the prior month, and writes every number the letter reports.

Construction (month m, all inputs dated m-1 or earlier)
  f_m      monthly factor return, French monthly file (compounded)
  RV_m     sum over days d in month m of (r_d - mean_m(r))^2, French daily file
           (Moreira and Muir's definition)
  c_{m-1}  sd(f_1..f_{m-1}) / sd(f_s / RV_{s-1}, s <= m-1)      expanding window
  w_m      min(c_{m-1} / RV_{m-1}, L)                           capped exposure
  net_m    w_m f_m - kappa |w_m - w_{m-1}|                      kappa = 14 bp
The first 120 months are a burn-in; positions start in July 1973.

Real-time combination (Cederburg et al. 2020), from month 60 of the real-time sample:
  (a, b)   = Sigma^{-1} mu of (f, net) over the expanding window
  s        = sd(f) / sd(a f + b net) over the same window
  e_m      = clip(s (a + b w_m), -L, L)                         effective exposure
  combo_m  = e_m f_m - kappa |e_m - e_{m-1}|

Inference
  Sharpe differences: studentized circular block bootstrap of Ledoit and Wolf (2008),
  HAC (Bartlett) standard error on the data, block-sum standard error in each resample,
  symmetric studentized 95% intervals. Holm (1979) adjustment within each family of six.
  Spanning regression net_m = alpha + beta f_m + e_m with Newey-West (6 lags) t-stats.

Outputs
  output/realtime_summary.csv, output/realtime_subperiods.csv, output/realtime_terciles.csv,
  output/realtime_robustness.csv, output/figures/*.png, paper/generated/*.tex

Run: python src/realtime.py
"""
from __future__ import annotations

import io
import os
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vol_managed_factors import BASE, DATA, OUT, FIG, FACTORS, load_factors  # noqa: E402

GEN = os.path.join(BASE, "paper", "generated")
PAPER_IMG = os.path.join(BASE, "paper", "images")

FF5_M_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOM_M_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"

KAPPA = 0.0014          # 14 bp per unit change in exposure (Moreira-Muir Table IV, highest level)
BURN_IN = 120           # months before the first real-time position
MIN_COMBO = 60          # months of real-time history before the combination trades
CAP = 1.5               # baseline exposure cap (Moreira-Muir Table IV also caps at 1.5)
BLOCK = 12              # bootstrap block length, months
N_BOOT = 4999
SEED = 20260928
POST_START = pd.Timestamp("2017-01-01")   # Moreira-Muir data end 2015, Cederburg et al. Dec 2016
ALT_SPLIT = pd.Timestamp("2016-01-01")
SIGNAL_HALFLIFE = 21     # trading days; pre-registered variant S2 (paper/PLAN_adaptive.md), adopted
MAIN_SIGNAL = "sig"      # adaptive EWMA signal; "rv" = canonical prior-month realized variance
NAMES = {"Mkt-RF": "Market", "SMB": "Size (SMB)", "HML": "Value (HML)",
         "RMW": "Profitability (RMW)", "CMA": "Investment (CMA)", "Mom": "Momentum"}


# ---------------------------------------------------------------- data

def _fetch_monthly(url: str) -> pd.DataFrame:
    raw = urllib.request.urlopen(url, timeout=60).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    lines = zf.read(zf.namelist()[0]).decode("latin-1").splitlines()
    header, rows = None, []
    for line in lines:
        cells = [c.strip() for c in line.split(",")]
        if header is None and cells[0] == "" and len(cells) > 1 and any(cells[1:]):
            header = [c for c in cells[1:] if c]
            continue
        if header is not None and cells[0].isdigit() and len(cells[0]) == 6:
            rows.append(cells[: len(header) + 1])
        elif header is not None and rows and cells[0] == "":
            break                                   # end of the monthly block
    df = pd.DataFrame(rows, columns=["Date"] + header)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index("Date").astype(float)
    df = df.mask(df <= -99.99) / 100.0
    return df


def load_monthly() -> pd.DataFrame:
    cache = os.path.join(DATA, "ff_factors_monthly.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    ff5 = _fetch_monthly(FF5_M_URL)
    mom = _fetch_monthly(MOM_M_URL)
    mom.columns = ["Mom"]
    df = ff5.join(mom, how="inner").dropna()
    os.makedirs(DATA, exist_ok=True)
    df.to_csv(cache)
    return df


def ewma_signal(daily: pd.Series, halflife: float = SIGNAL_HALFLIFE) -> pd.Series:
    """Exponentially weighted daily variance, sampled on the last trading day of each month, scaled to a month.

    sigma2_d = lam * sigma2_{d-1} + (1 - lam) * r_d^2 with lam = 2^(-1/halflife): every daily
    observation is used, with a learning rate set by the half-life (pre-registered choice S2).
    """
    r2 = daily.dropna() ** 2
    ew = r2.ewm(halflife=halflife, adjust=False).mean()
    s = ew.groupby(ew.index.to_period("M")).last() * 21
    s.index = s.index.to_timestamp("M")
    return s


def monthly_panel(daily: pd.Series, monthly: pd.Series) -> pd.DataFrame:
    """Monthly return from the monthly file; two variance signals known at each month end:
    rv  (canonical) realized variance of the month about its mean (Moreira and Muir)
    sig (adaptive)  exponentially weighted daily variance with a 21-day half-life"""
    g = daily.groupby(daily.index.to_period("M"))
    rv = g.apply(lambda x: float(np.sum((x.values - x.values.mean()) ** 2)))
    rv.index = rv.index.to_timestamp("M")
    return pd.DataFrame({"ret": monthly, "rv": rv, "sig": ewma_signal(daily)}).dropna()


# ---------------------------------------------------------------- construction

def realtime_managed(m: pd.DataFrame, cap: float | None, power: float = 1.0, burn_in: int = BURN_IN,
                     signal: str = MAIN_SIGNAL) -> pd.DataFrame:
    """Exposure w_m from data through m-1; power 1 = inverse variance, 0.5 = inverse volatility.

    signal "sig" is the adaptive EWMA signal (main specification); "rv" is the canonical rule."""
    f = m["ret"].values
    sig = m[signal].shift(1).values ** power
    n = len(f)
    w = np.full(n, np.nan)
    for t in range(burn_in - 1, n):            # one extra month so the first trade has a prior weight
        past = slice(1, t)
        c = np.std(f[past], ddof=1) / np.std(f[past] / sig[past], ddof=1)
        w[t] = c / sig[t]
        if cap is not None:
            w[t] = min(w[t], cap)
    df = pd.DataFrame({"f": f, "w": w}, index=m.index)
    df["turn"] = df["w"].diff().abs()
    df = df.iloc[burn_in:].copy()
    df["gross"] = df["w"] * df["f"]
    df["net"] = df["gross"] - KAPPA * df["turn"]
    return df


def realtime_combination(rt: pd.DataFrame, cap: float, kappa: float = KAPPA) -> tuple[pd.Series, pd.Series]:
    """Cederburg et al. (2020) real-time mean-variance mix of original and managed factor.

    Weights are estimated on the managed factor net of kappa; the mix pays kappa on its own exposure changes.
    """
    f, w = rt["f"].values, rt["w"].values
    net = (rt["gross"] - kappa * rt["turn"]).values
    n = len(f)
    e = np.full(n, np.nan)
    for t in range(MIN_COMBO - 1, n):
        X = np.column_stack([f[:t], net[:t]])
        mu, cov = X.mean(axis=0), np.cov(X, rowvar=False)
        ab = np.linalg.solve(cov, mu)
        s = np.std(f[:t], ddof=1) / np.std(X @ ab, ddof=1)
        e[t] = np.clip(s * (ab[0] + ab[1] * w[t]), -cap, cap)
    turn = np.abs(np.diff(e, prepend=np.nan))
    combo = e * f - kappa * turn
    combo[:MIN_COMBO] = np.nan                  # first trading month is MIN_COMBO (needs a prior exposure)
    return pd.Series(combo, index=rt.index, name="combo"), pd.Series(e, index=rt.index, name="e")


# ---------------------------------------------------------------- statistics

def sharpe(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(x.mean() / x.std(ddof=1) * np.sqrt(12))


def _sr_diff(v: np.ndarray) -> np.ndarray:
    """Monthly Sharpe difference from v = (mean1, mean2, second moment1, second moment2)."""
    return v[..., 0] / np.sqrt(v[..., 2] - v[..., 0] ** 2) - v[..., 1] / np.sqrt(v[..., 3] - v[..., 1] ** 2)


def _grad(v: np.ndarray) -> np.ndarray:
    a1, a2, b1, b2 = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
    d1, d2 = (b1 - a1 ** 2) ** 1.5, (b2 - a2 ** 2) ** 1.5
    return np.stack([b1 / d1, -b2 / d2, -a1 / (2 * d1), a2 / (2 * d2)], axis=-1)


def _hac(y: np.ndarray, lags: int) -> np.ndarray:
    u = y - y.mean(axis=0)
    T = len(u)
    S = u.T @ u / T
    for k in range(1, lags + 1):
        G = u[k:].T @ u[:-k] / T
        S += (1 - k / (lags + 1)) * (G + G.T)
    return S


def lw_test(r1: np.ndarray, r2: np.ndarray, rng=None, block: int = BLOCK) -> dict:
    """Ledoit-Wolf (2008) studentized circular block bootstrap for SR(r1) - SR(r2).

    Each call draws from its own generator seeded with SEED, so the same test on the same
    data always returns the same p-value wherever it is reported.
    """
    rng = np.random.default_rng(SEED)
    keep = ~(np.isnan(r1) | np.isnan(r2))
    r1, r2 = r1[keep], r2[keep]
    T = len(r1)
    y = np.column_stack([r1, r2, r1 ** 2, r2 ** 2])
    v = y.mean(axis=0)
    d = _sr_diff(v)
    g = _grad(v)
    se = np.sqrt(g @ _hac(y, block) @ g / T)
    b = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=(N_BOOT, b))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % T      # (B, b, l)
    ys = y[idx]                                                           # (B, b, l, 4)
    n = b * block
    vs = ys.reshape(N_BOOT, n, 4).mean(axis=1)
    ds = _sr_diff(vs)
    zeta = (ys.sum(axis=2) - block * vs[:, None, :]) / np.sqrt(block)    # (B, b, 4)
    psi = np.einsum("kbi,kbj->kij", zeta, zeta) / b
    gs = _grad(vs)
    ses = np.sqrt(np.einsum("ki,kij,kj->k", gs, psi, gs) / n)
    z = np.abs(ds - d) / ses
    p = (np.sum(z >= abs(d) / se) + 1) / (N_BOOT + 1)
    q = np.quantile(z, 0.95)
    ann = np.sqrt(12)
    return {"diff": d * ann, "se": se * ann, "p": p, "lo": (d - q * se) * ann, "hi": (d + q * se) * ann,
            "draws": (ds - d) * ann, "draw_se": ses * ann, "T": T}


def change_test(a: dict, b: dict) -> float:
    """p-value for equal Sharpe gains in two disjoint periods, from independent studentized draws."""
    stat = abs(a["diff"] - b["diff"]) / np.sqrt(a["se"] ** 2 + b["se"] ** 2)
    zs = np.abs(a["draws"] - b["draws"]) / np.sqrt(a["draw_se"] ** 2 + b["draw_se"] ** 2)
    return float((np.sum(zs >= stat) + 1) / (len(zs) + 1))


def spanning(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    keep = ~(np.isnan(y) | np.isnan(x))
    res = sm.OLS(y[keep], sm.add_constant(x[keep])).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return float(res.params[0] * 12), float(res.tvalues[0])


def holm(p: dict) -> dict:
    items = sorted(p.items(), key=lambda kv: kv[1])
    out, run = {}, 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (len(items) - i) * v))
        out[k] = run
    return out


def max_dd_equal_vol(x: np.ndarray, ref: np.ndarray) -> float:
    """Largest peak-to-trough loss of summed returns, x scaled to ref's volatility (percentage points)."""
    x, ref = np.asarray(x, dtype=float), np.asarray(ref, dtype=float)
    keep = ~np.isnan(x)
    xs = x[keep] * np.std(ref[keep], ddof=1) / np.std(x[keep], ddof=1)
    c = np.cumsum(xs)
    return float((c - np.maximum.accumulate(c)).min() * 100)


def breakeven_bps(rt: pd.DataFrame) -> float:
    target = sharpe(rt["f"])
    if sharpe(rt["gross"]) <= target:
        return 0.0
    lo, hi = 0.0, 0.10
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sharpe(rt["gross"] - mid * rt["turn"]) > target else (lo, mid)
    return lo * 1e4


# ---------------------------------------------------------------- LaTeX helpers

def num(x: float, d: int = 2) -> str:
    s = f"{abs(x):.{d}f}"
    return f"\\ensuremath{{-}}{s}" if round(x, d) < 0 else s


def pfmt(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def macro(name: str, val: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{val}}}\n"


# ---------------------------------------------------------------- main

def main() -> None:
    rng = np.random.default_rng(SEED)
    for d in (OUT, FIG, GEN, PAPER_IMG):
        os.makedirs(d, exist_ok=True)
    daily, monthly = load_factors(), load_monthly()
    panels = {fac: monthly_panel(daily[fac], monthly[fac]) for fac in FACTORS}
    first = panels["Mkt-RF"].index[BURN_IN]
    last = panels["Mkt-RF"].index[-1]
    print(f"Monthly {monthly.index.min():%Y-%m} to {monthly.index.max():%Y-%m}; "
          f"real-time positions {first:%Y-%m} to {last:%Y-%m}")

    base, base_canon, rows, sub, terc, tests = {}, {}, [], [], [], {}
    for fac in FACTORS:
        rt = realtime_managed(panels[fac], CAP)
        rt["combo"], rt["e"] = realtime_combination(rt, CAP)
        base[fac] = rt
        f, net, cb = rt["f"].values, rt["net"].values, rt["combo"].values
        tm, tc = lw_test(net, f, rng), lw_test(cb, f, rng)
        a, t = spanning(net, f)
        can = realtime_managed(panels[fac], CAP, signal="rv")
        base_canon[fac] = can
        tcan = lw_test(net, can["net"].values)
        rows.append({"factor": fac, "months": len(rt), "sr_orig": sharpe(f), "sr_net": sharpe(net),
                     "sr_canon": sharpe(can["net"]), "p_vs_canon": tcan["p"], "turn_canon": can["turn"].mean(),
                     "diff": tm["diff"], "lo": tm["lo"], "hi": tm["hi"], "p": tm["p"],
                     "sr_orig_cw": sharpe(np.where(np.isnan(cb), np.nan, f)), "sr_combo": sharpe(cb),
                     "diff_c": tc["diff"], "p_c": tc["p"], "alpha": a, "alpha_t": t,
                     "dd_orig": max_dd_equal_vol(f, f), "dd_net": max_dd_equal_vol(net, f),
                     "avg_w": rt["w"].mean(), "share_cap": float((rt["w"] >= CAP - 1e-12).mean()),
                     "avg_turn": rt["turn"].mean(), "breakeven": breakeven_bps(rt),
                     "combo_start": rt["combo"].first_valid_index(),
                     "combo_clip": float((rt["e"].dropna().abs() >= CAP - 1e-12).mean())})
        for per, mask in (("pre", rt.index < POST_START), ("post", rt.index >= POST_START)):
            r = rt[mask]
            tm_s, tc_s = lw_test(r["net"].values, r["f"].values, rng), lw_test(r["combo"].values, r["f"].values, rng)
            tests[(fac, per)] = tm_s
            a_s, t_s = spanning(r["net"].values, r["f"].values)
            rc = base_canon[fac][mask]
            tcs = lw_test(r["net"].values, rc["net"].values)
            sub.append({"factor": fac, "period": per, "months": int(mask.sum()), "sr_orig": sharpe(r["f"]),
                        "sr_canon": sharpe(rc["net"]), "p_vs_canon": tcs["p"],
                        "cov_canon": float(np.cov(rc["w"], rc["f"], bias=True)[0, 1] * 12),
                        "cost_canon": float((KAPPA * rc["turn"]).mean() * 12),
                        "sr_net": sharpe(r["net"]), "diff": tm_s["diff"], "lo": tm_s["lo"], "hi": tm_s["hi"],
                        "p": tm_s["p"], "sr_combo": sharpe(r["combo"]), "diff_c": tc_s["diff"],
                        "lo_c": tc_s["lo"], "hi_c": tc_s["hi"], "p_c": tc_s["p"], "alpha": a_s, "alpha_t": t_s,
                        "cov_wf": float(np.cov(r["w"], r["f"], bias=True)[0, 1] * 12),
                        "avg_w": float(r["w"].mean()), "mean_f": float(r["f"].mean() * 12),
                        "ew_ef": float(r["w"].mean() * r["f"].mean() * 12),
                        "cost": float((KAPPA * r["turn"]).mean() * 12), "mean_net": float(r["net"].mean() * 12),
                        "low_exposure_months": int((r["w"] < 0.2).sum()),
                        "M": (1 + (np.cov(r["w"], r["f"], bias=True)[0, 1] - (KAPPA * r["turn"]).mean())
                              / (r["w"].mean() * r["f"].mean())) if r["f"].mean() > 0 else np.nan,
                        "R": float(r["w"].mean() * r["f"].std(ddof=1) / r["net"].std(ddof=1)),
                        "p_unadj": tm_s["p"]})
            q = pd.qcut(r["w"].rank(method="first"), 3, labels=["turbulent", "middle", "calm"])
            for lab, val in (r.groupby(q, observed=True)["f"].mean() * 12).items():
                terc.append({"factor": fac, "period": per, "tercile": lab, "mean": val})

    summary = pd.DataFrame(rows).set_index("factor")
    summary["p_holm"] = pd.Series(holm(summary["p"].to_dict()))
    summary["p_c_holm"] = pd.Series(holm(summary["p_c"].to_dict()))
    subs = pd.DataFrame(sub).set_index(["factor", "period"])
    for per in ("pre", "post"):
        for col in ("p", "p_c"):
            h = holm({k: subs.loc[(k, per), col] for k in FACTORS})
            for k in FACTORS:
                subs.loc[(k, per), col + "_holm"] = h[k]
    for k in FACTORS:
        subs.loc[(k, "post"), "p_change"] = change_test(tests[(k, "pre")], tests[(k, "post")])
    terc = (pd.DataFrame(terc).pivot_table(index=["factor", "period"], columns="tercile", values="mean",
                                           observed=True)[["turbulent", "middle", "calm"]])

    # ------------------------------------------------ robustness: momentum (full sample), value (after the split)
    rob = []

    def rob_row(label, rt_m, rt_v, block=BLOCK, split=POST_START):
        tm = lw_test(rt_m["net"].values, rt_m["f"].values, rng, block)
        pv = rt_v[rt_v.index >= split]
        tv = lw_test(pv["net"].values, pv["f"].values, rng, block)
        av, avt = spanning(pv["net"].values, pv["f"].values)
        rob.append({"spec": label, "mom_orig": sharpe(rt_m["f"]), "mom_sr": sharpe(rt_m["net"]), "mom_p": tm["p"],
                    "hml_alpha": av, "hml_t": avt, "hml_p": tv["p"]})

    rob_row("Baseline", base["Mom"], base["HML"])
    rob_row("Canonical signal (prior-month RV)", base_canon["Mom"], base_canon["HML"])
    rob_row("6-month bootstrap blocks", base["Mom"], base["HML"], block=6)
    rob_row("24-month bootstrap blocks", base["Mom"], base["HML"], block=24)
    rob_row("Split at January 2016", base["Mom"], base["HML"], split=ALT_SPLIT)
    rob_row("Cap 2", realtime_managed(panels["Mom"], 2.0), realtime_managed(panels["HML"], 2.0))
    rob_row("No cap", realtime_managed(panels["Mom"], None), realtime_managed(panels["HML"], None))
    rob_row("Inverse volatility", realtime_managed(panels["Mom"], CAP, 0.5), realtime_managed(panels["HML"], CAP, 0.5))
    # The burn-in only sets the first trading month: c uses every month since 1963 whatever its length,
    # so exposures in any given month do not depend on it. What can matter is the start date.
    mm, hh = realtime_managed(panels["Mom"], CAP, burn_in=180), realtime_managed(panels["HML"], CAP, burn_in=180)
    rob_row(f"Start in {mm.index[0]:%B %Y}", mm, hh)
    rob = pd.DataFrame(rob)

    # combination variants for momentum: which friction removes its significance?
    combo_var = {}
    rt_m = base["Mom"]
    for label, cap, kappa in (("uncapped", 1e9, KAPPA), ("free", 1e9, 0.0)):
        c, _ = realtime_combination(rt_m, cap, kappa)
        t = lw_test(c.values, rt_m["f"].values, rng)
        combo_var[label] = (sharpe(c), t["p"])

    summary.drop(columns=["combo_start"]).round(4).to_csv(os.path.join(OUT, "realtime_summary.csv"))
    subs.round(4).to_csv(os.path.join(OUT, "realtime_subperiods.csv"))
    terc.round(4).to_csv(os.path.join(OUT, "realtime_terciles.csv"))
    rob.round(4).to_csv(os.path.join(OUT, "realtime_robustness.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print(summary.drop(columns=["combo_start"]).to_string())
    print(subs.to_string())
    print(terc.to_string())
    print(rob.to_string(index=False))

    write_tables(summary, subs, rob)
    write_macros(summary, subs, terc, rob, base, first, last, combo_var)
    make_figures(subs, terc, base)
    print("\nWrote output/realtime_*.csv, output/figures/*.png, paper/generated/*.tex")


# ---------------------------------------------------------------- tables

def _b(flag: bool):
    return (lambda s: f"\\textbf{{{s}}}") if flag else (lambda s: s)


def write_tables(summary, subs, rob) -> None:
    L = []
    for k in FACTORS:
        r = summary.loc[k]
        b, bc = _b(r.p_holm < 0.05), _b(r.p_c_holm < 0.05)
        L.append(f"{NAMES[k]} & {num(r.sr_orig)} & {num(r.sr_canon)} & {b(num(r.sr_net))} & {b(pfmt(r.p_holm))} & "
                 f"{num(r.sr_orig_cw)} & {bc(num(r.sr_combo))} & {bc(pfmt(r.p_c_holm))} & "
                 f"{num(r.dd_orig, 0)} & {num(r.dd_net, 0)} \\\\")
    open(os.path.join(GEN, "table_full.tex"), "w").write("\n".join(L) + "\n")

    L = []
    for k in FACTORS:
        cells = []
        for per in ("pre", "post"):
            r = subs.loc[(k, per)]
            b = _b(r.p_holm < 0.05)
            cells += [num(r.sr_orig), num(r.sr_canon), b(num(r.sr_net)), b(pfmt(r.p_holm)),
                      f"{num(r.alpha * 100, 1)} ({num(r.alpha_t)})"]
        L.append(f"{NAMES[k]} & " + " & ".join(cells) + " \\\\")
    open(os.path.join(GEN, "table_periods.tex"), "w").write("\n".join(L) + "\n")

    L = []
    for k in FACTORS:
        for per, lab in (("pre", "1973 to 2016"), ("post", "2017 to 2026")):
            r = subs.loc[(k, per)]
            name = NAMES[k] if per == "pre" else ""
            mch = "n/a" if np.isnan(r.M) else num(r.M)
            L.append(f"{name} & {lab} & {num(r.avg_w)} & {num(r.mean_f * 100, 1)} & {num(r.ew_ef * 100, 1)} & "
                     f"{num(r.cov_wf * 100, 1)} & {num(r.cost * 100, 1)} & {num(r.mean_net * 100, 1)} & "
                     f"{mch} & {num(r.R)} \\\\")
        if k != FACTORS[-1]:
            L.append("\\addlinespace[2pt]")
    open(os.path.join(GEN, "table_decomp.tex"), "w").write("\n".join(L) + "\n")

    L = []
    for _, r in rob.iterrows():
        L.append(f"{r.spec} & {num(r.mom_orig)} & {num(r.mom_sr)} & {pfmt(r.mom_p)} & "
                 f"{num(r.hml_alpha * 100, 1)} & {num(r.hml_t)} & {pfmt(r.hml_p)} \\\\")
    open(os.path.join(GEN, "table_robust.tex"), "w").write("\n".join(L) + "\n")


def write_macros(summary, subs, terc, rob, base, first, last, combo_var) -> None:
    S, P = summary, subs
    m = macro("RTstart", f"{first:%B %Y}") + macro("RTend", f"{last:%B %Y}")
    m += macro("RTmonths", str(int(S.loc["Mom", "months"])))
    m += macro("PreMonths", str(int(P.loc[("Mom", "pre"), "months"])))
    m += macro("PostMonths", str(int(P.loc[("Mom", "post"), "months"])))
    m += macro("ComboStart", f"{S.loc['Mom', 'combo_start']:%B %Y}")
    mo = S.loc["Mom"]
    m += macro("MomOrig", num(mo.sr_orig)) + macro("MomNet", num(mo.sr_net)) + macro("MomPHolm", pfmt(mo.p_holm))
    m += macro("MomDiffLo", num(mo.lo)) + macro("MomDiffHi", num(mo.hi))
    m += macro("MomComboOrig", num(mo.sr_orig_cw)) + macro("MomCombo", num(mo.sr_combo))
    m += macro("MomComboPHolm", pfmt(mo.p_c_holm))
    m += macro("MomDDOrig", num(-mo.dd_orig, 0)) + macro("MomDDNet", num(-mo.dd_net, 0))
    m += macro("MomBreakeven", f"{mo.breakeven:.0f}") + macro("MomBreakevenX", f"{mo.breakeven / (KAPPA * 1e4):.0f}")
    m += macro("MomAvgW", num(mo.avg_w)) + macro("MomShareCap", f"{mo.share_cap * 100:.0f}")
    m += macro("MomCanon", num(mo.sr_canon)) + macro("MomPVsCanon", pfmt(mo.p_vs_canon))
    m += macro("MomTurn", num(mo.avg_turn)) + macro("MomTurnCanon", num(mo.turn_canon))
    ratios = S["avg_turn"] / S["turn_canon"]
    m += macro("TurnRatioMin", f"{ratios.min() * 100:.0f}") + macro("TurnRatioMax", f"{ratios.max() * 100:.0f}")
    rm = S.loc["RMW"]
    m += macro("RmwAlpha", num(rm.alpha * 100, 1)) + macro("RmwAlphaT", num(rm.alpha_t, 1))
    m += macro("RmwOrig", num(rm.sr_orig)) + macro("RmwNet", num(rm.sr_net)) + macro("RmwP", pfmt(rm.p))
    m += macro("RmwComboOrig", num(rm.sr_orig_cw)) + macro("RmwCombo", num(rm.sr_combo))
    pre, post = P.loc[("Mom", "pre")], P.loc[("Mom", "post")]
    m += macro("MomPreOrig", num(pre.sr_orig)) + macro("MomPreNet", num(pre.sr_net))
    m += macro("MomPostOrig", num(post.sr_orig)) + macro("MomPostNet", num(post.sr_net))
    m += macro("MomPostCanon", num(post.sr_canon)) + macro("MomPreCanon", num(pre.sr_canon))
    m += macro("MomPostPVsCanon", pfmt(post.p_vs_canon))
    m += macro("MomPostCovCanon", num(post.cov_canon * 100, 1)) + macro("MomPostCostCanon", num(post.cost_canon * 100, 1))
    m += macro("MomPostP", pfmt(post.p)) + macro("MomPostLo", num(post.lo)) + macro("MomPostHi", num(post.hi))
    m += macro("MomPreGain", num(pre["diff"])) + macro("MomPostGain", num(post["diff"]))
    m += macro("MomChangeP", pfmt(post.p_change))
    m += macro("MomPreAlpha", num(pre.alpha * 100, 1)) + macro("MomPostAlpha", num(post.alpha * 100, 1))
    m += macro("MomPreAvgW", num(pre.avg_w)) + macro("MomPostAvgW", num(post.avg_w))
    m += macro("MomPreCovPct", num(pre.cov_wf * 100, 1)) + macro("MomPostCovPct", num(post.cov_wf * 100, 1))
    m += macro("MomPostLowMonths", str(int(post.low_exposure_months)))
    m += macro("MomComboClip", f"{S.loc['Mom', 'combo_clip'] * 100:.0f}")
    m += macro("MomComboP", pfmt(mo.p_c))
    m += macro("MomComboUncapSR", num(combo_var["uncapped"][0])) + macro("MomComboUncapP", pfmt(combo_var["uncapped"][1]))
    m += macro("MomComboFreeSR", num(combo_var["free"][0])) + macro("MomComboFreeP", pfmt(combo_var["free"][1]))
    m += macro("MomPrePUnadj", pfmt(pre.p_unadj)) + macro("MomPrePHolm", pfmt(pre.p_holm))
    m += macro("MomPreM", num(pre.M)) + macro("MomPreR", num(pre.R))
    rp = P.loc[("RMW", "pre")]
    m += macro("RmwPreM", num(rp.M)) + macro("RmwPreR", num(rp.R))
    m += macro("MomPostM", num(post.M)) + macro("MomPostR", num(post.R))
    m += macro("MomPreCostPct", num(pre.cost * 100, 1)) + macro("MomPostCostPct", num(post.cost * 100, 1))
    m += macro("MomPreMeanF", num(pre.mean_f * 100, 1)) + macro("MomPostMeanF", num(post.mean_f * 100, 1))
    m += macro("MomP", pfmt(mo.p))
    hv = P.loc[("HML", "post")]
    m += macro("HmlPostAlpha", num(hv.alpha * 100, 1)) + macro("HmlPostAlphaT", num(hv.alpha_t))
    m += macro("HmlPostOrig", num(hv.sr_orig)) + macro("HmlPostNet", num(hv.sr_net))
    m += macro("HmlPostP", pfmt(hv.p)) + macro("HmlPostPHolm", pfmt(hv.p_holm))
    m += macro("HmlPostCovPct", num(hv.cov_wf * 100, 1)) + macro("HmlPreCovPct", num(P.loc[("HML", "pre"), "cov_wf"] * 100, 1))
    cv = P.loc[("CMA", "post")]
    m += macro("CmaPostAlpha", num(cv.alpha * 100, 1)) + macro("CmaPostAlphaT", num(cv.alpha_t))
    m += macro("CmaPostP", pfmt(cv.p)) + macro("CmaPostPHolm", pfmt(cv.p_holm))
    for fac, tag in (("Mom", "Mom"), ("HML", "Hml"), ("CMA", "Cma")):
        for per, ptag in (("pre", "Pre"), ("post", "Post")):
            for t in ("turbulent", "middle", "calm"):
                m += macro(f"{tag}{ptag}{t.capitalize()}", num(terc.loc[(fac, per), t] * 100, 1))
            m += macro(f"{tag}{ptag}Cov", num(P.loc[(fac, per), "cov_wf"] * 100, 2))
    hmlp = base["HML"][base["HML"].index >= POST_START]
    top = hmlp.sort_values("f", ascending=False).head(8)
    m += macro("HmlTopMin", num(top["w"].min())) + macro("HmlTopMax", num(top["w"].max()))
    m += macro("HmlTopInTwentyOneTwo", str(int(((top.index.year >= 2021) & (top.index.year <= 2022)).sum())))
    r16 = rob[rob.spec == "Split at January 2016"].iloc[0]
    m += macro("HmlSplitSixteenAlpha", num(r16.hml_alpha * 100, 1))
    m += macro("HmlSplitSixteenT", num(r16.hml_t)) + macro("HmlSplitSixteenP", pfmt(r16.hml_p))
    m += macro("MomRobMaxP", pfmt(rob["mom_p"].max()))
    m += macro("MomRobMinSR", num(rob["mom_sr"].min())) + macro("MomRobMaxSR", num(rob["mom_sr"].max()))
    open(os.path.join(GEN, "numbers.tex"), "w").write(m)


# ---------------------------------------------------------------- figures

def make_design_figure(base) -> None:
    """Study design: (a) sample windows, (b) what is known when each monthly position is set."""
    blue, red, grey = "#1f4e79", "#c0392b", "#8a8a8e"
    rt = base["Mom"]
    start, combo0, end = rt.index[0], rt["combo"].first_valid_index(), rt.index[-1]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 4.9), gridspec_kw={"height_ratios": [1.25, 1]})
    yr = lambda d: d.year + (d.month - 1) / 12
    rows = [("Moreira and Muir (2017) sample", 1926.6, 2016.0, grey),
            ("Cederburg et al. (2020) sample", 1926.6, 2017.0, grey),
            ("Burn-in: scale constant only", 1963.5, yr(start), "#bdbdbd"),
            ("Real-time managed factor", yr(start), yr(end) + 1 / 12, blue),
            ("Real-time combination", yr(combo0), yr(end) + 1 / 12, red)]
    for i, (lab, a, b, col) in enumerate(rows):
        ax.barh(i, b - a, left=a, height=0.55, color=col)
        ax.text(a - 1, i, lab, ha="right", va="center", fontsize=8.5)
    ax.axvline(2017, color="#555", lw=0.9, ls=":")
    ax.text(2017.3, -0.9, "split: January 2017", fontsize=8.5, va="center")
    ax.set_xlim(1900, 2029)
    ax.set_ylim(-1.3, len(rows) - 0.4)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xticks([1930, 1950, 1963, 1973, 1990, 2017, 2026])
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    ax.set_title("(a) Samples", fontsize=10, loc="left")

    bx.set_xlim(0, 10)
    bx.set_ylim(0, 3)
    bx.axis("off")
    bx.set_title("(b) Information used for the position held in month m", fontsize=10, loc="left")
    boxes = [(0.2, "Up to the end of month $m-1$\ndaily $r_d$ for $S$, monthly $f$\nexpanding window for $c$", grey),
             (3.55, "End of month $m-1$\nvariance signal $S_{m-1}$ observed\n$c_{m-1}$, $w_m$, $e_m$ set", blue),
             (6.9, "Month $m$\nearn $w_m f_m$\npay $\\kappa\\,|w_m-w_{m-1}|$", red)]
    for x, txt, col in boxes:
        bx.add_patch(plt.Rectangle((x, 0.35), 2.9, 2.3, fill=False, ec=col, lw=1.4))
        bx.text(x + 1.45, 1.5, txt, ha="center", va="center", fontsize=8.5, linespacing=1.6)
    for x0, x1 in ((3.1, 3.55), (6.45, 6.9)):
        bx.annotate("", xy=(x1, 1.5), xytext=(x0, 1.5), arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
    fig.tight_layout()
    _save(fig, "fig_design.png")


def make_figures(subs, terc, base) -> None:
    grey, blue, red = "#8a8a8e", "#1f4e79", "#c0392b"
    labels = [NAMES[k] for k in FACTORS]
    make_design_figure(base)

    # Figure 1: Sharpe gains with 95% studentized intervals, by period
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    y = np.arange(len(FACTORS))
    for ax, (dcol, lo, hi, title) in zip(axes, [("diff", "lo", "hi", "(a) Managed factor, net, minus original"),
                                                ("diff_c", "lo_c", "hi_c", "(b) Real-time combination minus original")]):
        for per, off, col, lab in (("pre", -0.15, blue, "1973 to 2016"), ("post", 0.15, red, "2017 to 2026")):
            d = np.array([subs.loc[(k, per), dcol] for k in FACTORS])
            lo_v = np.array([subs.loc[(k, per), lo] for k in FACTORS])
            hi_v = np.array([subs.loc[(k, per), hi] for k in FACTORS])
            ax.errorbar(d, y + off, xerr=[d - lo_v, hi_v - d], fmt="o", color=col, ms=4, capsize=2, lw=1, label=lab)
        ax.axvline(0, color="#555", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("difference in annualized Sharpe ratio")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "fig_sharpe_gains.png")

    # Figure 2: mean return by real-time exposure tercile
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    x = np.arange(len(FACTORS))
    shades = {"turbulent": red, "middle": "#bdbdbd", "calm": blue}
    for ax, (per, title) in zip(axes, (("pre", "(a) 1973 to 2016"), ("post", "(b) 2017 to 2026"))):
        for i, t in enumerate(("turbulent", "middle", "calm")):
            vals = [terc.loc[(k, per), t] * 100 for k in FACTORS]
            ax.bar(x + (i - 1) * 0.27, vals, 0.27, color=shades[t], label=f"{t} months")
        ax.axhline(0, color="#555", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Mkt", "SMB", "HML", "RMW", "CMA", "Mom"])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("mean factor return, % per year")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_terciles.png")

    # Figure 3: exposure and returns after 2016, momentum and value
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True)
    for ax, fac, tag in zip(axes, ("Mom", "HML"), ("(a)", "(b)")):
        r = base[fac][base[fac].index >= POST_START]
        ax.bar(r.index, r["f"] * 100, width=20, color=np.where(r["f"] > 0, "#9ecae1", "#fcbba1"))
        ax.set_ylabel("return, %")
        ax2 = ax.twinx()
        ax2.plot(r.index, r["w"], color=blue if fac == "Mom" else red, lw=1.4)
        ax2.set_ylim(0, 1.6)
        ax2.set_ylabel("exposure")
        ax.set_title(f"{tag} {NAMES[fac]}: monthly return (bars) and real-time exposure (line)", fontsize=10)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_exposure.png")

    # Figure 4: momentum cumulative returns at 10% volatility (display scaling only)
    s = base["Mom"]

    def at10(v):
        v = np.asarray(v, dtype=float)
        return v * 0.10 / (np.nanstd(v, ddof=1) * np.sqrt(12))

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(s.index, np.cumsum(at10(s["f"])), color=grey, lw=1.4, label="Momentum (original)")
    ax.plot(s.index, np.cumsum(at10(s["net"])), color=blue, lw=1.6, label="Real-time managed, cap 1.5, net of costs")
    c = s["combo"].dropna()
    ax.plot(c.index, np.cumsum(at10(c)), color=red, lw=1.2, ls="--", label="Real-time combination, net of costs")
    ax.axvline(POST_START, color="#555", lw=0.8, ls=":")
    ax.set_ylabel("cumulative return, each scaled to 10% vol")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_momentum.png")


def _save(fig, name: str) -> None:
    fig.savefig(os.path.join(FIG, name), dpi=200)
    fig.savefig(os.path.join(PAPER_IMG, name), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

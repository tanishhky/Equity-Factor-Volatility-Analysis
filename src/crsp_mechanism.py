"""Parameter-free CRSP test of momentum's formation beta (paper/PLAN_mechanism.md).

Stock-level CRSP data are licensed: they are cached under data/crsp/ (gitignored) and never committed.
Only portfolio-level series and test statistics are written to output/.

Run: python src/crsp_mechanism.py inspect | pull | run
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import gaussian_kde

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "crsp")
OUT = os.path.join(ROOT, "output", "mechanism")
BETA_WIN, BETA_MIN, FORM = 60, 36, 11


# ---------------------------------------------------------------- WRDS access

def wrds_user() -> str:
    for line in open(os.path.expanduser("~/.pgpass")):
        f = line.strip().split(":")
        if f[0].startswith("wrds-pgdata"):
            return f[3]
    raise RuntimeError("no WRDS entry in ~/.pgpass")


def connect():
    import wrds
    return wrds.Connection(wrds_username=wrds_user())


def inspect():
    db = connect()
    for q in ("select max(date) as last from crsp.msf", "select max(mthcaldt) as last from crsp.msf_v2"):
        try:
            print(q, "->", db.raw_sql(q).iloc[0, 0])
        except Exception as e:                      # noqa: BLE001
            print(q, "-> unavailable:", str(e).splitlines()[0][:120])
    for lib, tab in (("crsp", "msf_v2"), ("crsp", "msf")):
        try:
            print(tab, list(db.describe_table(lib, tab)["name"]))
        except Exception as e:                      # noqa: BLE001
            print(tab, "describe failed:", str(e).splitlines()[0][:120])
    db.close()


# ---------------------------------------------------------------- panel construction (pure; tested on simulated data)

def month_end(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s) + pd.offsets.MonthEnd(0)


def rolling_beta(R: np.ndarray, x: np.ndarray, win: int = BETA_WIN, min_obs: int = BETA_MIN):
    """OLS beta and its squared standard error of each column of R (T x N stock excess returns, NaN = missing)
    on x (T market excess returns), over the window of `win` months ending at each row (inclusive)."""
    T, N = R.shape
    m = np.isfinite(R)
    Y = np.where(m, R, 0.0)
    X = np.where(m, x[:, None], 0.0)
    def rs(A):
        c = np.cumsum(np.vstack([np.zeros((1, A.shape[1])), A]), axis=0)
        return c[win:] - c[:-win]
    n, sx, sy = rs(m.astype(float)), rs(X), rs(Y)
    sxx, sxy, syy = rs(X * X), rs(X * Y), rs(Y * Y)
    with np.errstate(invalid="ignore", divide="ignore"):
        Sxx = sxx - sx * sx / n
        Sxy = sxy - sx * sy / n
        Syy = syy - sy * sy / n
        b = Sxy / Sxx
        s2 = (Syy - b * Sxy) / (n - 2)
        se2 = s2 / Sxx
    ok = (n >= min_obs) & (Sxx > 0)
    beta, var = np.full((T, N), np.nan), np.full((T, N), np.nan)
    beta[win - 1:][ok] = b[ok]
    var[win - 1:][ok] = se2[ok]
    return beta, var


def quantile_density(x: np.ndarray, c: float) -> float:
    """Density of x at the point c by the quantile-spacing (sparsity) estimator, 2h / (Q(p + h) - Q(p - h)) with
    p = F(c) and the Hall-Sheather bandwidth (amendment A2); less smoothing bias at a quantile than a kernel."""
    from scipy.stats import norm
    n = len(x)
    p = float(np.mean(x <= c))
    z = norm.ppf(p)
    h = n ** (-1 / 3) * norm.ppf(0.975) ** (2 / 3) * (1.5 * norm.pdf(z) ** 2 / (2 * z * z + 1)) ** (1 / 3)
    h = min(h, p - 1e-3, 1 - p - 1e-3)
    q1, q2 = np.quantile(x, [p - h, p + h])
    return 2 * h / (q2 - q1)


def build_panel(ret: pd.DataFrame, me: pd.DataFrame, nyse: pd.DataFrame, mkt: pd.DataFrame) -> pd.DataFrame:
    """ret, me, nyse: wide (month-end x permno) monthly total return, market equity, NYSE flag.
    mkt: French monthly Mkt-RF and RF (decimal). Returns one row per formation month m with the portfolio returns
    in m+1, the directly measured formation beta, the predicted beta and its inputs."""
    idx = ret.index
    mk = mkt.reindex(idx)
    rf, rm = mk["RF"].values, mk["Mkt-RF"].values
    R = ret.values
    logr = np.log1p(R)
    # formation return over months m-11..m-1 (all 11 required)
    c = np.cumsum(np.vstack([np.zeros((1, R.shape[1])), np.nan_to_num(logr)]), axis=0)
    cnt = np.cumsum(np.vstack([np.zeros((1, R.shape[1])), np.isfinite(logr)]), axis=0)
    T = len(idx)
    form = np.full(R.shape, np.nan)
    for t in range(FORM + 1, T):
        s, e = t - FORM, t                      # rows t-11 .. t-1
        ok = (cnt[e] - cnt[s]) == FORM
        form[t, ok] = (c[e] - c[s])[ok]
    # R_F is the coefficient on each stock's beta in its compounded formation return (amendments A1, A2): to first
    # order, log(1 + rf + b rm + u) is linear in b with coefficient rm (1 - rm - rf); the risk-free part is common to
    # every stock and cancels in the sort
    RF_mkt = pd.Series(rm * (1 - rm - rf)).rolling(FORM).sum().shift(1).values
    # betas from months m-71..m-12: rolling window ending at m-12
    b_end, v_end = rolling_beta(R - rf[:, None], rm)
    beta = np.full(R.shape, np.nan)
    bvar = np.full(R.shape, np.nan)
    beta[12:], bvar[12:] = b_end[:-12], v_end[:-12]
    rows = []
    MEv, NY = me.values, nyse.values.astype(bool)
    for t in range(T - 1):
        r_i, size, nx = form[t], MEv[t], NY[t]
        valid = np.isfinite(r_i) & np.isfinite(size) & (size > 0)
        if valid.sum() < 50 or not np.isfinite(RF_mkt[t]):
            continue
        ny = valid & nx
        if ny.sum() < 20:
            continue
        s_bp = np.median(size[ny])
        lo_bp, hi_bp = np.percentile(r_i[ny], [30, 70])
        nxt = R[t + 1]
        row = {"date": idx[t], "RF": RF_mkt[t], "R_M_next": rm[t + 1], "rf_next": rf[t + 1]}
        legs, bform, pred = {}, [], []
        for g, gmask in (("S", valid & (size <= s_bp)), ("B", valid & (size > s_bp))):
            hi, lo = gmask & (r_i >= hi_bp), gmask & (r_i <= lo_bp)
            for leg, lm in (("H", hi), ("L", lo)):
                hm = lm & np.isfinite(nxt)
                w = size[hm]
                legs[g + leg] = (np.average(nxt[hm], weights=w) if w.sum() > 0 else np.nan, np.mean(nxt[hm]) if hm.any() else np.nan)
            bm = gmask & np.isfinite(beta[t])
            if bm.sum() < 30:
                bform.append(np.nan); pred.append(np.nan); continue
            bh, bl = bm & (r_i >= hi_bp), bm & (r_i <= lo_bp)
            bform.append(np.mean(beta[t][bh]) - np.mean(beta[t][bl]))
            sb2 = max(np.var(beta[t][bm], ddof=1) - np.mean(bvar[t][bm]), 0.0)
            rg = r_i[gmask]
            q_hi, q_lo = (rg >= hi_bp).mean(), (rg <= lo_bp).mean()
            pred.append(sb2 * RF_mkt[t] * (quantile_density(rg, hi_bp) / q_hi + quantile_density(rg, lo_bp) / q_lo))
            row[f"sigma_b2_{g}"] = sb2
        row["mom_vw"] = 0.5 * (legs["SH"][0] + legs["BH"][0]) - 0.5 * (legs["SL"][0] + legs["BL"][0])
        row["mom_ew"] = 0.5 * (legs["SH"][1] + legs["BH"][1]) - 0.5 * (legs["SL"][1] + legs["BL"][1])
        row["beta_form"] = np.nanmean(bform) if np.isfinite(bform).any() else np.nan
        row["beta_pred"] = np.mean(pred) if np.isfinite(pred).all() else np.nan
        row["n_stocks"] = int(valid.sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------- tests (paper/PLAN_mechanism.md)

def t1(panel: pd.DataFrame) -> dict:
    d = panel[["beta_form", "beta_pred"]].dropna()
    res = sm.OLS(d["beta_form"], sm.add_constant(d["beta_pred"])).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    return {"slope": float(res.params.iloc[1]), "slope_se": float(res.bse.iloc[1]), "const": float(res.params.iloc[0]),
            "r2": float(res.rsquared), "corr": float(d.corr().iloc[0, 1]), "months": len(d)}


def t2(panel: pd.DataFrame, col: str) -> dict:
    d = panel[[col, "beta_pred", "R_M_next", "RF"]].dropna()
    y = d[col]
    X = pd.DataFrame({"const": 1.0, "c": d["beta_pred"] * d["R_M_next"], "d": d["R_M_next"]})
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    wald = res.wald_test("c = 1, d = 0", use_f=False, scalar=True)
    base = sm.OLS(y, sm.add_constant(d["R_M_next"])).fit()
    gm = sm.OLS(y, pd.DataFrame({"const": 1.0, "rm": d["R_M_next"], "rfrm": d["RF"] * d["R_M_next"]})).fit()
    r2_gain, r2_gm = res.rsquared - base.rsquared, gm.rsquared - base.rsquared
    return {"c": float(res.params["c"]), "c_se": float(res.bse["c"]), "c_t_vs_0": float(res.tvalues["c"]),
            "c_p_one_sided": float(1 - __import__("scipy.stats").stats.norm.cdf(res.tvalues["c"])),
            "d": float(res.params["d"]), "d_se": float(res.bse["d"]),
            "wald_c1_d0_stat": float(wald.statistic), "wald_c1_d0_p": float(wald.pvalue),
            "r2_gain_over_market": float(r2_gain), "r2_gain_grundy_martin_in_sample": float(r2_gm),
            "share_of_gm": float(r2_gain / r2_gm) if r2_gm > 0 else np.nan, "months": len(d)}


def outcome(t: dict) -> str:
    lo, hi = t["c"] - 1.96 * t["c_se"], t["c"] + 1.96 * t["c_se"]
    if lo <= 1 <= hi and t["wald_c1_d0_p"] >= 0.05:
        return "magnitude confirmed"
    if t["c_p_one_sided"] < 0.05:
        return "sign and shape confirmed, magnitude off by the factor c"
    return "mechanism fails as a monthly prediction"


if __name__ == "__main__":
    {"inspect": inspect}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()

"""
vol_managed_factors.py - Moreira & Muir (2017) volatility-managed factor portfolios.

For each Fama-French factor (Mkt-RF, SMB, HML, RMW, CMA) plus Momentum (MOM), build
the volatility-managed version:

    f_managed[m] = (c / RV[m-1]) * f[m]

where RV[m-1] is the realized variance of the factor over the prior month (sum of
squared daily returns), and c is a single constant chosen so the managed series matches
the original's full-sample volatility. The timing signal 1/RV[m-1] uses only past data,
so the leverage decision is out-of-sample; c is a scale-only normalization for
comparability (it does not affect the Sharpe ratio or the spanning-regression t-stat).

The key Moreira-Muir result: regressing the managed factor on the original factor,
    f_managed[m] = alpha + beta * f[m] + e[m],
yields a positive, significant alpha for several factors - volatility timing earns
return the original factor does not span.

Outputs:
  data/ff_factors_daily.csv          cached Ken French daily factors
  output/vol_managed_metrics.csv     per-factor metrics (original vs managed) + spanning alpha
  output/figures/cum_mkt.png         cumulative market factor: managed vs original
  output/figures/sharpe_bars.png     Sharpe original vs managed across factors

Run: python src/vol_managed_factors.py
"""
from __future__ import annotations

import io
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output")
FIG = os.path.join(OUT, "figures")

FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
COST_PER_TURN = 0.0014  # 14 bps per unit change in leverage (Moreira-Muir cost robustness)


def _fetch_zip_csv(url: str) -> pd.DataFrame:
    """Download a Ken French daily CSV zip and return the daily block as a DataFrame."""
    raw = urllib.request.urlopen(url, timeout=30).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    name = zf.namelist()[0]
    text = zf.read(name).decode("latin-1").splitlines()

    rows = []
    for line in text:
        parts = [p.strip() for p in line.split(",")]
        # daily data rows start with an 8-digit date (YYYYMMDD)
        if parts and parts[0].isdigit() and len(parts[0]) == 8:
            rows.append(parts)
    # find header columns: the line whose first cell is empty and which precedes data
    header = None
    for line in text:
        cells = [c.strip() for c in line.split(",")]
        if cells[0] == "" and len(cells) > 1 and any(c for c in cells[1:]):
            header = [c for c in cells[1:] if c]
            break
    cols = ["Date"] + header
    df = pd.DataFrame(rows)
    df = df.iloc[:, : len(cols)]
    df.columns = cols
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.set_index("Date").astype(float) / 100.0  # percent -> decimal
    return df


def load_factors() -> pd.DataFrame:
    os.makedirs(DATA, exist_ok=True)
    cache = os.path.join(DATA, "ff_factors_daily.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    ff5 = _fetch_zip_csv(FF5_URL)
    mom = _fetch_zip_csv(MOM_URL)
    mom.columns = ["Mom"]
    df = ff5.join(mom, how="inner").dropna()
    df.to_csv(cache)
    return df


def to_monthly(daily: pd.Series) -> pd.DataFrame:
    """Monthly factor return (sum of daily) and realized variance (sum of squared daily)."""
    g = daily.groupby(daily.index.to_period("M"))
    out = pd.DataFrame({"ret": g.sum(), "rv": g.apply(lambda x: np.sum(x.values ** 2))})
    out.index = out.index.to_timestamp("M")
    return out


def perf(returns: np.ndarray) -> dict:
    mu = returns.mean() * 12
    vol = returns.std(ddof=1) * np.sqrt(12)
    sharpe = mu / vol if vol > 0 else np.nan
    cum = np.cumsum(returns)              # long-short factors: cumulative sum
    dd = (cum - np.maximum.accumulate(cum)).min()
    return {"ann_ret": mu, "ann_vol": vol, "sharpe": sharpe, "max_dd": dd}


def analyze_factor(daily: pd.Series) -> dict:
    m = to_monthly(daily)
    rv_lag = m["rv"].shift(1)
    valid = rv_lag.notna() & (rv_lag > 0)
    f = m["ret"][valid].values
    w = (1.0 / rv_lag[valid]).values

    raw = w * f
    c = f.std(ddof=1) / raw.std(ddof=1)   # vol-match (scale only)
    managed = c * raw
    w_scaled = c * w

    orig_p = perf(f)
    man_p = perf(managed)

    # spanning regression with Newey-West (HAC) standard errors
    X = sm.add_constant(f)
    res = sm.OLS(managed, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    alpha_ann = res.params[0] * 12
    alpha_t = res.tvalues[0]
    r2 = res.rsquared

    # transaction-cost drag on the leverage changes
    turnover = np.abs(np.diff(w_scaled, prepend=w_scaled[0]))
    net = managed - COST_PER_TURN * turnover
    net_p = perf(net)

    return {
        "n_months": len(f),
        "orig_sharpe": orig_p["sharpe"], "man_sharpe": man_p["sharpe"], "net_sharpe": net_p["sharpe"],
        "orig_ret": orig_p["ann_ret"], "man_ret": man_p["ann_ret"],
        "orig_dd": orig_p["max_dd"], "man_dd": man_p["max_dd"],
        "alpha_ann": alpha_ann, "alpha_t": alpha_t, "r2": r2,
        "avg_turnover": turnover.mean(),
        "_series": (m.index[valid], np.cumsum(f), np.cumsum(managed)),
    }


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    df = load_factors()
    print(f"Ken French daily factors: {df.index.min().date()} to {df.index.max().date()} "
          f"({len(df)} days)")

    rows = {}
    series = {}
    for fac in FACTORS:
        r = analyze_factor(df[fac])
        series[fac] = r.pop("_series")
        rows[fac] = r

    table = pd.DataFrame(rows).T
    show = table[["orig_sharpe", "man_sharpe", "net_sharpe", "alpha_ann", "alpha_t", "r2", "avg_turnover"]].copy()
    show.columns = ["Sharpe(orig)", "Sharpe(managed)", "Sharpe(net)", "alpha_ann", "alpha_t", "R2", "turnover"]
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print("\nVolatility-managed factors (Moreira-Muir), monthly:\n")
    print(show.to_string())

    os.makedirs(OUT, exist_ok=True)
    table.round(4).to_csv(os.path.join(OUT, "vol_managed_metrics.csv"))

    # figure 1: cumulative market factor managed vs original
    idx, cum_o, cum_m = series["Mkt-RF"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(idx, cum_o, color="#8a8a8e", lw=1.4, label="Market factor (buy-and-hold)")
    ax.plot(idx, cum_m, color="#1f4e79", lw=1.6, label="Volatility-managed market factor")
    ax.set_title("Volatility-Managed Market Factor vs Buy-and-Hold (cumulative excess return)")
    ax.set_ylabel("cumulative sum of monthly excess returns")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cum_mkt.png"), dpi=200)
    plt.close(fig)

    # figure 2: Sharpe bars original vs managed
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(FACTORS))
    ax.bar(x - 0.2, [rows[f]["orig_sharpe"] for f in FACTORS], 0.4, label="original", color="#8a8a8e")
    ax.bar(x + 0.2, [rows[f]["man_sharpe"] for f in FACTORS], 0.4, label="vol-managed", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(FACTORS)
    ax.set_ylabel("annualized Sharpe ratio")
    ax.set_title("Sharpe Ratio: Original vs Volatility-Managed Factors")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sharpe_bars.png"), dpi=200)
    plt.close(fig)
    print(f"\nWrote output/vol_managed_metrics.csv and 2 figures to output/figures/")


if __name__ == "__main__":
    main()

"""
realtime.py - does volatility management survive when everything is estimated in real time?

The original study (vol_managed_factors.py) follows Moreira and Muir (2017): the timing
signal 1/RV[m-1] is known in advance, but the scale constant c is fitted on the full
sample, the weights are uncapped, and the spanning alpha is an in-sample statistic.
Cederburg, O'Doherty, Wang and Yan (2020) show that these choices matter: most managed
strategies lose to their unmanaged factor once an investor has to form the combination
in real time. This module asks the same question for the Fama-French five factors and
momentum, with every estimate using only data available at the time.

Four checks, each using only information up to month m-1 to set the position in month m:

1. Real-time scaling. c[m-1] = sd(f[1..m-1]) / sd(f[s] / RV[s-1], s <= m-1), an
   expanding-window version of the full-sample constant.
2. Leverage caps. Exposure w[m] = c[m-1] / RV[m-1] is capped at 1.5x and 2x (and left
   uncapped for reference). Barroso and Santa-Clara (2015) cap momentum's scaling too.
3. Real-time combination (Cederburg et al. 2020). The investor holds the mean-variance
   mix of the original and managed factor, with weights estimated on an expanding
   window, rescaled to the original factor's real-time volatility. If volatility timing
   adds value an investor could have captured, this mix beats the original factor.
4. Post-sample. Moreira and Muir's data end in 2015 and Cederburg et al.'s in December
   2016, so 2017 onward is out of sample for both.

Sharpe differences are tested with a circular block bootstrap (12-month blocks), and
the six factors' p-values are adjusted with Holm's step-down procedure.

Outputs:
  output/realtime_summary.csv     one row per factor and strategy variant
  output/realtime_subperiods.csv  spanning alpha and real-time Sharpe by subperiod
  output/realtime_terciles.csv    mean factor return by real-time exposure tercile (mechanism)
  output/figures/realtime_mom.png cumulative real-time managed vs original momentum

Run: python src/realtime.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vol_managed_factors import FACTORS, COST_PER_TURN, OUT, FIG, load_factors, to_monthly  # noqa: E402

BURN_IN = 120          # months of history before the first real-time position (10 years)
CAPS = [None, 2.0, 1.5]
POST_START = "2017-01-01"   # Moreira-Muir (2017) data end in 2015, Cederburg et al. (2020) in Dec 2016
N_BOOT = 5000
BLOCK = 12
SEED = 20260928


# ---------------------------------------------------------------- construction

def realtime_positions(m: pd.DataFrame, cap: float | None, power: float = 1.0) -> pd.DataFrame:
    """Real-time managed exposure for each month, using data through m-1 only.

    power=1 scales by inverse variance (Moreira-Muir); power=0.5 by inverse volatility.
    """
    f = m["ret"].values
    rv_lag = m["rv"].shift(1).values ** power
    n = len(f)
    w = np.full(n, np.nan)
    for t in range(BURN_IN, n):
        past = slice(1, t)                     # months with a defined lagged RV, before t
        raw_past = f[past] / rv_lag[past]
        c = np.std(f[past], ddof=1) / np.std(raw_past, ddof=1)
        w[t] = c / rv_lag[t]
        if cap is not None:
            w[t] = min(w[t], cap)
    out = pd.DataFrame({"f": f, "w": w}, index=m.index)
    out["managed"] = out["w"] * out["f"]
    return out.iloc[BURN_IN:]


def realtime_combination(f: np.ndarray, fm: np.ndarray, min_obs: int = 60) -> np.ndarray:
    """Cederburg et al. (2020) real-time mix of original f and managed fm.

    Mean-variance weights on (f, fm) from an expanding window, then rescaled so the
    mix targets the original factor's expanding-window volatility.
    """
    n = len(f)
    out = np.full(n, np.nan)
    for t in range(min_obs, n):
        X = np.column_stack([f[:t], fm[:t]])
        mu = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        try:
            w = np.linalg.solve(cov, mu)
        except np.linalg.LinAlgError:
            continue
        port_past = X @ w
        sd_p = np.std(port_past, ddof=1)
        if sd_p <= 0:
            continue
        scale = np.std(f[:t], ddof=1) / sd_p
        out[t] = scale * (w[0] * f[t] + w[1] * fm[t])
    return out


# ---------------------------------------------------------------- statistics

def sharpe(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    return x.mean() / x.std(ddof=1) * np.sqrt(12)


def boot_sharpe_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Sharpe(a) - Sharpe(b) and a two-sided circular-block-bootstrap p-value."""
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    n = len(a)
    diff = sharpe(a) - sharpe(b)
    n_blocks = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(N_BOOT, -1)[:, :n] % n
    A, B = a[idx], b[idx]
    d = (A.mean(1) / A.std(1, ddof=1) - B.mean(1) / B.std(1, ddof=1)) * np.sqrt(12)
    # centre the bootstrap distribution on zero to test H0: no difference
    p = np.mean(np.abs(d - d.mean()) >= abs(diff))
    return diff, p


def max_dd(x: np.ndarray) -> float:
    c = np.nancumsum(x)
    return float((c - np.maximum.accumulate(c)).min())


def spanning_alpha(fm: np.ndarray, f: np.ndarray) -> tuple[float, float]:
    keep = ~(np.isnan(fm) | np.isnan(f))
    res = sm.OLS(fm[keep], sm.add_constant(f[keep])).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return res.params[0] * 12, res.tvalues[0]


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    k = len(items)
    adj, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (k - i) * p))
        adj[name] = running
    return adj


def breakeven_cost(managed: np.ndarray, orig: np.ndarray, w: np.ndarray) -> float:
    """Cost per unit of leverage turnover (in bps) at which the managed Sharpe falls to the original's."""
    turn = np.abs(np.diff(w, prepend=w[0]))
    target = sharpe(orig)
    lo, hi = 0.0, 0.05
    if sharpe(managed) <= target:
        return 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sharpe(managed - mid * turn) > target:
            lo = mid
        else:
            hi = mid
    return lo * 1e4


# ---------------------------------------------------------------- main

def main() -> None:
    rng = np.random.default_rng(SEED)
    df = load_factors()
    print(f"Ken French daily factors: {df.index.min().date()} to {df.index.max().date()}")

    rows, sub_rows, terc_rows, series = [], [], [], {}
    p_managed, p_combo = {}, {}
    for fac in FACTORS:
        m = to_monthly(df[fac])
        for cap in CAPS:
            rt = realtime_positions(m, cap)
            f, fm, w = rt["f"].values, rt["managed"].values, rt["w"].values
            turn = np.abs(np.diff(w, prepend=w[0]))
            net = fm - COST_PER_TURN * turn
            combo = realtime_combination(f, fm)
            d_man, p_man = boot_sharpe_diff(net, f, rng)
            d_com, p_com = boot_sharpe_diff(combo, f, rng)
            a, t = spanning_alpha(fm, f)
            label = "uncapped" if cap is None else f"cap {cap:g}x"
            rows.append({
                "factor": fac, "variant": label,
                "start": rt.index[0].strftime("%Y-%m"), "months": len(rt),
                "sharpe_orig": sharpe(f), "sharpe_managed": sharpe(fm), "sharpe_managed_net": sharpe(net),
                "sharpe_combo": sharpe(combo),
                "sharpe_orig_combo_window": sharpe(np.where(np.isnan(combo), np.nan, f)),
                "maxdd_orig": max_dd(f), "maxdd_managed_net": max_dd(net),
                "diff_net_vs_orig": d_man, "p_net_vs_orig": p_man,
                "diff_combo_vs_orig": d_com, "p_combo_vs_orig": p_com,
                "alpha_ann": a, "alpha_t": t,
                "avg_leverage": np.nanmean(w), "max_leverage": np.nanmax(w),
                "share_capped": np.mean(w >= cap - 1e-12) if cap else 0.0,
                "avg_turnover": turn.mean(),
                "breakeven_bps": breakeven_cost(fm, f, w),
            })
            if cap == 1.5:
                p_managed[fac], p_combo[fac] = p_man, p_com
                series[fac] = rt.assign(net=net, combo=combo)
                for name, mask in [("1973-2016", rt.index < POST_START), ("2017 onward", rt.index >= POST_START)]:
                    a_s, t_s = spanning_alpha(fm[mask], f[mask])
                    _, p_s = boot_sharpe_diff(net[mask], f[mask], rng)
                    _, pc_s = boot_sharpe_diff(combo[mask], f[mask], rng)
                    sub_rows.append({
                        "factor": fac, "period": name, "months": int(mask.sum()),
                        "sharpe_orig": sharpe(f[mask]), "sharpe_managed_net": sharpe(net[mask]),
                        "sharpe_combo": sharpe(combo[mask]), "alpha_ann": a_s, "alpha_t": t_s,
                        "p_net_vs_orig": p_s, "p_combo_vs_orig": pc_s,
                    })
                    # mechanism: where does the factor earn its return, calm or turbulent months?
                    r = rt[mask]
                    terc = pd.qcut(r["w"], 3, labels=["turbulent (low exposure)", "middle", "calm (high exposure)"])
                    for lab, val in (r.groupby(terc, observed=True)["f"].mean() * 12).items():
                        terc_rows.append({"factor": fac, "period": name, "tercile": lab, "mean_ann": val})

        # robustness: inverse-volatility scaling, 1.5x cap
        rt = realtime_positions(m, 1.5, power=0.5)
        f, fm, w = rt["f"].values, rt["managed"].values, rt["w"].values
        net = fm - COST_PER_TURN * np.abs(np.diff(w, prepend=w[0]))
        d, p = boot_sharpe_diff(net, f, rng)
        rows.append({"factor": fac, "variant": "inv-vol, cap 1.5x", "start": rt.index[0].strftime("%Y-%m"),
                     "months": len(rt), "sharpe_orig": sharpe(f), "sharpe_managed": sharpe(fm),
                     "sharpe_managed_net": sharpe(net), "diff_net_vs_orig": d, "p_net_vs_orig": p,
                     "maxdd_orig": max_dd(f), "maxdd_managed_net": max_dd(net),
                     "avg_leverage": np.nanmean(w), "breakeven_bps": breakeven_cost(fm, f, w)})

    table = pd.DataFrame(rows)
    adj_m, adj_c = holm(p_managed), holm(p_combo)
    table["p_net_vs_orig_holm"] = [adj_m[f] if v == "cap 1.5x" else np.nan for f, v in zip(table.factor, table.variant)]
    table["p_combo_vs_orig_holm"] = [adj_c[f] if v == "cap 1.5x" else np.nan for f, v in zip(table.factor, table.variant)]
    subs = pd.DataFrame(sub_rows)

    os.makedirs(FIG, exist_ok=True)
    table.round(4).to_csv(os.path.join(OUT, "realtime_summary.csv"), index=False)
    subs.round(4).to_csv(os.path.join(OUT, "realtime_subperiods.csv"), index=False)
    terc = pd.DataFrame(terc_rows).pivot_table(index=["factor", "period"], columns="tercile", values="mean_ann")
    terc.round(4).to_csv(os.path.join(OUT, "realtime_terciles.csv"))

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    cols = ["factor", "variant", "sharpe_orig", "sharpe_managed_net", "sharpe_orig_combo_window", "sharpe_combo", "p_net_vs_orig",
            "p_combo_vs_orig", "alpha_t", "maxdd_orig", "maxdd_managed_net", "avg_leverage", "breakeven_bps"]
    print("\nReal-time volatility management (positions use data through m-1 only):\n")
    print(table[cols].to_string(index=False))
    print("\nHolm-adjusted p-values (cap 1.5x): net vs original", {k: round(v, 3) for k, v in adj_m.items()})
    print("Holm-adjusted p-values (cap 1.5x): combination vs original", {k: round(v, 3) for k, v in adj_c.items()})
    print("\nSubperiods (cap 1.5x):\n")
    print(subs.to_string(index=False))
    print("\nAnnualized factor return by real-time exposure tercile (cap 1.5x):\n")
    print(terc.to_string())

    s = series["Mom"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(s.index, np.nancumsum(s["f"]), color="#8a8a8e", lw=1.4, label="Momentum (original)")
    ax.plot(s.index, np.nancumsum(s["net"]), color="#1f4e79", lw=1.6, label="Real-time managed, 1.5x cap, net of costs")
    ax.plot(s.index, np.nancumsum(s["combo"]), color="#c0392b", lw=1.2, ls="--", label="Real-time combination")
    ax.axvline(pd.Timestamp(POST_START), color="#555", lw=0.8, ls=":")
    ax.text(pd.Timestamp(POST_START), ax.get_ylim()[1] * 0.95, " published samples end", fontsize=8, va="top")
    ax.set_ylabel("cumulative sum of monthly returns")
    ax.set_title("Momentum: real-time volatility management")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "realtime_mom.png"), dpi=200)
    plt.close(fig)
    print("\nWrote output/realtime_summary.csv, output/realtime_subperiods.csv, output/figures/realtime_mom.png")


if __name__ == "__main__":
    main()

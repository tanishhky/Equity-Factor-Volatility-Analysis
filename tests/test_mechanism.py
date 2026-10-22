"""The CRSP pipeline recovers a known formation beta on a simulated market and never uses future data."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import crsp_mechanism as C  # noqa: E402


def simulated_market(T=360, N=2000, seed=1, tails=None, vol_beta=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1950-01-31", periods=T, freq="ME")
    b = 1 + 0.4 * rng.standard_normal(N)
    rm = rng.normal(0.006, 0.045, T)
    rf = np.full(T, 0.003)
    sd = 0.10 + vol_beta * (b - 1)                       # idiosyncratic volatility rising with beta (follow-up F1)
    sd = np.clip(sd, 0.03, None)[None, :]
    u = sd * (rng.standard_normal((T, N)) if tails is None else rng.standard_t(tails, (T, N)) * np.sqrt((tails - 2) / tails))
    R = rf[:, None] + b[None, :] * rm[:, None] + u
    me = pd.DataFrame(np.exp(rng.normal(5, 1.5, N))[None, :] * np.ones((T, 1)), index=idx)
    nyse = pd.DataFrame((rng.random(N) < 0.4)[None, :] * np.ones((T, 1), bool), index=idx)
    mkt = pd.DataFrame({"Mkt-RF": rm, "RF": rf}, index=idx)
    return pd.DataFrame(R, index=idx), me, nyse, mkt, b


def _cs(seeds, tails=None):
    out = []
    for sd in seeds:
        ret, me, nyse, mkt, _ = simulated_market(seed=sd, tails=tails)
        panel = C.build_panel(ret, me, nyse, mkt)
        out.append((C.t2(panel, "mom_ew"), C.t1(panel)))
    return out


def test_pipeline_recovers_formation_beta():
    # one market can miss by chance; across markets the slope must centre on 1 and the joint test hold its size
    for tails in (None, 4.0):
        out = _cs(range(200, 208), tails)
        c = np.array([o[0]["c"] for o in out])
        assert abs(c.mean() - 1) < 0.07, (tails, c)
        assert sum(o[0]["wald_c1_d0_p"] < 0.05 for o in out) <= 2, (tails, [o[0]["wald_c1_d0_p"] for o in out])
        assert np.mean([o[1]["slope"] for o in out]) - 1 < 0.08


def test_pipeline_uses_no_future_data():
    ret, me, nyse, mkt, _ = simulated_market(T=200, N=600, seed=2)
    a = C.build_panel(ret, me, nyse, mkt)
    ret2, mkt2 = ret.copy(), mkt.copy()
    ret2.iloc[150:] = ret2.iloc[150:] * -3 + 0.2
    mkt2.iloc[150:, 0] = -0.3
    b = C.build_panel(ret2, me, nyse, mkt2)
    cols = ["beta_form", "beta_pred", "RF", "mom_ew", "mom_vw"]
    last = ret.index[148]                    # holding month 149 < 150
    pd.testing.assert_frame_equal(a.loc[:last, cols], b.loc[:last, cols])


def test_volatility_drag_gives_momentum_a_negative_level():
    """When idiosyncratic volatility rises with beta, compounding pushes high-beta stocks into the loser leg whatever
    the market did, so the registered prediction leaves a negative level d (the sign found on CRSP). The first-order
    correction registered as F1 overshoots by about 25% here and failed validation (paper/PLAN_mechanism.md)."""
    d_reg = []
    for sd in range(300, 308):
        ret, me, nyse, mkt, _ = simulated_market(seed=sd, vol_beta=0.08)
        d_reg.append(C.t2(C.build_panel(ret, me, nyse, mkt), "mom_ew")["d"])
    assert np.mean(d_reg) < -0.03, d_reg

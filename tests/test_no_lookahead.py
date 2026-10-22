"""Every real-time position must be invariant to data dated after it is set.

For a cut-off month t, replace every return and realized variance from t onward with noise and
check that exposures up to and including month t are identical: the position held in
month t is set at the end of t-1, so it may not depend on month t's own return either, for the managed factor and for the real-time combination.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from realtime import BURN_IN, MIN_COMBO, realtime_combination, realtime_managed  # noqa: E402


def _panel(n=420, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1960-01-31", periods=n, freq="ME")
    rv = np.exp(rng.normal(-6.5, 0.6, n))
    ret = rng.normal(0.005, 1, n) * np.sqrt(rv * 21)
    sig = rv * np.exp(rng.normal(0, 0.2, n))
    return pd.DataFrame({"ret": ret, "rv": rv, "sig": sig}, index=idx)


def _future_shock(m, t, seed=1):
    rng = np.random.default_rng(seed)
    out = m.copy()
    k = len(m) - t
    out.iloc[t:, 0] = rng.normal(0, 0.2, k)              # returns from month t onward, including t itself
    out.iloc[t:, 1] = np.exp(rng.normal(-3, 1.0, k))     # RV from month t onward (enters w only from t+1)
    out.iloc[t:, 2] = np.exp(rng.normal(-3, 1.0, k))     # adaptive signal from month t onward
    return out


def test_managed_exposure_ignores_future():
    m = _panel()
    for signal in ("sig", "rv"):
      for cap in (None, 1.5):
        for t in (BURN_IN + 5, BURN_IN + 150, len(m) - 10):
            a = realtime_managed(m, cap, signal=signal)
            b = realtime_managed(_future_shock(m, t), cap, signal=signal)
            upto = a.index[a.index <= m.index[t]]
            assert np.allclose(a.loc[upto, "w"], b.loc[upto, "w"]), (signal, cap, t)


def test_combination_exposure_ignores_future():
    m = _panel()
    t = BURN_IN + MIN_COMBO + 100
    a, b = realtime_managed(m, 1.5), realtime_managed(_future_shock(m, t), 1.5)
    _, ea = realtime_combination(a, 1.5)
    _, eb = realtime_combination(b, 1.5)
    upto = ea.index[ea.index <= m.index[t]]
    assert np.allclose(ea.loc[upto].dropna(), eb.loc[upto].dropna())


def test_shock_actually_changes_later_positions():
    m = _panel()
    t = BURN_IN + 150
    a, b = realtime_managed(m, None), realtime_managed(_future_shock(m, t), None)
    later = a.index[a.index > m.index[t]]
    assert not np.allclose(a.loc[later, "w"], b.loc[later, "w"])


# ---------------------------------------------------------------- adaptive estimators (src/adaptive.py)

def _daily(n_days=6000, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1990-01-01", periods=n_days)
    vol = np.exp(np.cumsum(rng.normal(0, 0.03, n_days)) * 0.3) * 0.01
    return pd.Series(rng.normal(0, 1, n_days) * vol, index=idx)


def test_adaptive_signals_ignore_future_days():
    from adaptive import signals
    from realtime import monthly_panel, ewma_signal
    d = _daily()
    monthly = (1 + d).groupby(d.index.to_period("M")).prod() - 1
    monthly.index = monthly.index.to_timestamp("M")
    panel = monthly_panel(d, monthly)
    cut = panel.index[150]
    shocked = d.copy()
    rng = np.random.default_rng(9)
    shocked[shocked.index > cut] = rng.normal(0, 0.05, int((shocked.index > cut).sum()))
    s_a = signals(d, panel.index, panel["rv"])
    rv_b = shocked.groupby(shocked.index.to_period("M")).apply(lambda x: float(np.sum((x - x.mean()) ** 2)))
    rv_b.index = rv_b.index.to_timestamp("M")
    s_b = signals(shocked, panel.index, rv_b.reindex(panel.index))
    upto = panel.index <= cut
    for col in s_a.columns:
        a, b = s_a.loc[upto, col].values, s_b.loc[upto, col].values
        ok = np.isfinite(a)
        assert np.allclose(a[ok], b[ok]), col
    assert np.allclose(ewma_signal(d).loc[:cut].values, ewma_signal(shocked).loc[:cut].values)


def test_adaptive_managed_and_combination_ignore_future():
    from adaptive import managed, combination
    m = _panel()
    t = BURN_IN + MIN_COMBO + 90
    shocked = _future_shock(m, t)
    for hl in (None, 36, 120):
        a = managed(m["ret"].values, m["sig"].values, m.index, hl)
        b = managed(shocked["ret"].values, shocked["sig"].values, m.index, hl)
        upto = a.index <= m.index[t]
        assert np.allclose(a.loc[upto, "w"], b.loc[upto, "w"], equal_nan=True), hl
        for chl in (None, 60):
            ca, cb = combination(a, chl), combination(b, chl)
            u = upto & ca.notna().values
            # combination return in month t uses f_t; compare exposures implicitly through months before t
            assert np.allclose(ca[u & (ca.index < m.index[t])], cb[u & (cb.index < m.index[t])]), (hl, chl)

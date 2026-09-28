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
    return pd.DataFrame({"ret": ret, "rv": rv}, index=idx)


def _future_shock(m, t, seed=1):
    rng = np.random.default_rng(seed)
    out = m.copy()
    k = len(m) - t
    out.iloc[t:, 0] = rng.normal(0, 0.2, k)              # returns from month t onward, including t itself
    out.iloc[t:, 1] = np.exp(rng.normal(-3, 1.0, k))     # RV from month t onward (enters w only from t+1)
    return out


def test_managed_exposure_ignores_future():
    m = _panel()
    for cap in (None, 1.5):
        for t in (BURN_IN + 5, BURN_IN + 150, len(m) - 10):
            a = realtime_managed(m, cap)
            b = realtime_managed(_future_shock(m, t), cap)
            upto = a.index[a.index <= m.index[t]]
            assert np.allclose(a.loc[upto, "w"], b.loc[upto, "w"]), (cap, t)


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

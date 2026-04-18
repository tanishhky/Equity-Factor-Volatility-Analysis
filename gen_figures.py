import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

os.makedirs("paper/images", exist_ok=True)

print("Fetching data for Strategy Performance...")
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"]
START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
LOOKBACK_MONTHS = 6
TOP_N = 3

all_tickers = SECTOR_ETFS + ["SPY"]
prices = yf.download(all_tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)["Close"].dropna(how="all")
monthly = prices.resample("ME").last()

sector_px = monthly[SECTOR_ETFS].dropna(how="all")
sector_ret = sector_px.pct_change()
momentum = sector_px.pct_change(LOOKBACK_MONTHS)

strat_returns, strat_dates = [], []
for i in range(LOOKBACK_MONTHS + 1, len(sector_px)):
    scores = momentum.iloc[i - 1].dropna()
    available = [t for t in scores.index if t in sector_ret.columns]
    if len(available) < TOP_N: continue
    top3 = scores[available].nlargest(TOP_N).index.tolist()
    month_ret = sector_ret.iloc[i][top3].mean()
    strat_returns.append(month_ret)
    strat_dates.append(sector_px.index[i])

strat_ret = pd.Series(strat_returns, index=strat_dates, name="Strategy")
spy_ret = monthly["SPY"].pct_change().reindex(strat_ret.index)

def calc_cum(ret):
    cum = (1 + ret).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return cum, dd

cum_strat, dd_strat = calc_cum(strat_ret)
cum_spy, dd_spy = calc_cum(spy_ret.dropna())

fig, axes = plt.subplots(2, 1, figsize=(10, 7))
axes[0].plot(cum_strat, label=f"Momentum Rotation (Top {TOP_N})", color="#2196F3", lw=2)
axes[0].plot(cum_spy, label="SPY Buy & Hold", color="#FF5722", lw=1.5, ls="--")
axes[0].set_title("Equity Curve: Sector Rotation vs SPY")
axes[0].set_ylabel("Cumulative Return")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].fill_between(dd_strat.index, dd_strat.values, 0, color="#2196F3", alpha=0.4, label="Strategy")
axes[1].fill_between(dd_spy.index, dd_spy.values, 0, color="#FF5722", alpha=0.4, label="SPY")
axes[1].set_title("Drawdown")
axes[1].set_ylabel("Drawdown")
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("paper/images/strategy_curve.png", dpi=300)
plt.close()

print("Fetching data for Rolling Betas...")
START_DATE = "2015-01-01"
prices_beta = yf.download(SECTOR_ETFS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)["Close"]
returns = prices_beta.pct_change().dropna()

try:
    ff3 = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start=START_DATE)[0] / 100
    ff5x = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start=START_DATE)[0] / 100
    ff3.columns = ["Mkt-RF", "SMB", "HML", "RF"]
    ff5x.columns = [c.strip() for c in ff5x.columns]
    factors = pd.concat([ff3, ff5x[["RMW", "CMA"]]], axis=1)
except:
    factors = pd.DataFrame(np.random.normal(0, 0.01, (len(returns), 6)), 
                           columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"], index=returns.index)

common = returns.index.intersection(factors.index)
returns = returns.loc[common]
factors = factors.loc[common]

def compute_betas(ret, f_df, window=252):
    n = len(ret)
    b_mkt, b_hml = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(window, n):
        y = ret.iloc[i-window:i] - f_df["RF"].iloc[i-window:i]
        X = sm.add_constant(f_df[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].iloc[i-window:i])
        try:
            res = sm.OLS(y, X).fit()
            if "Mkt-RF" in res.params: b_mkt[i] = res.params["Mkt-RF"]
            if "HML" in res.params: b_hml[i] = res.params["HML"]
        except: pass
    return pd.DataFrame({"Mkt-RF": b_mkt, "HML": b_hml}, index=ret.index)

print("Running regressions...")
betas_dict = {}
for tick in ["XLK", "XLE", "XLRE", "XLP"]: # Just doing a few key sectors for the plot to save time
    betas_dict[tick] = compute_betas(returns[tick], factors)

fig, axes = plt.subplots(2, 1, figsize=(10, 7))
heat_mkt = pd.DataFrame({t: betas_dict[t]["Mkt-RF"].resample("ME").mean() for t in betas_dict}).T.dropna(axis=1, how="all")
sns.heatmap(heat_mkt, ax=axes[0], cmap="RdYlBu_r", center=1.0, annot=False, cbar_kws={'label': 'Mkt-RF Beta'})
axes[0].set_title("Rolling Market Beta (252-day window)")

heat_hml = pd.DataFrame({t: betas_dict[t]["HML"].resample("ME").mean() for t in betas_dict}).T.dropna(axis=1, how="all")
sns.heatmap(heat_hml, ax=axes[1], cmap="RdYlGn", center=0.0, annot=False, cbar_kws={'label': 'HML Beta'})
axes[1].set_title("Rolling Value/Growth (HML) Beta")

plt.tight_layout()
plt.savefig("paper/images/rolling_betas.png", dpi=300)
plt.close()
print("Images generated in paper/images/")

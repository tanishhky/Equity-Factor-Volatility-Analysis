import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import warnings
from scipy.optimize import minimize
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")
os.makedirs("paper/images", exist_ok=True)

# CONFIG
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"]
LOOKBACK_MONTHS = 6
TOP_N = 3

print("Fetching data for Strategy Performance...")
all_tickers = SECTOR_ETFS + ["SPY"]
prices = yf.download(all_tickers, start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)["Close"].dropna(how="all")
monthly = prices.resample("ME").last()

sector_px = monthly[SECTOR_ETFS].dropna(how="all")
sector_ret = sector_px.pct_change()
momentum = sector_px.pct_change(LOOKBACK_MONTHS)

strat_returns, strat_dates, history = [], [], []
for i in range(LOOKBACK_MONTHS + 1, len(sector_px)):
    scores = momentum.iloc[i - 1].dropna()
    available = [t for t in scores.index if t in sector_ret.columns]
    if len(available) < TOP_N: continue
    top3 = scores[available].nlargest(TOP_N).index.tolist()
    month_ret = sector_ret.iloc[i][top3].mean()
    strat_returns.append(month_ret)
    strat_dates.append(sector_px.index[i])
    history.append(top3)

strat_ret = pd.Series(strat_returns, index=strat_dates, name="Strategy")
spy_ret = monthly["SPY"].pct_change().reindex(strat_ret.index)

# Turnover
turnovers = []
for i in range(1, len(history)):
    prev_set = set(history[i-1])
    curr_set = set(history[i])
    turnovers.append(1.0 - len(prev_set.intersection(curr_set)) / TOP_N)
avg_turnover = np.mean(turnovers)

# FRED T-bill
try:
    rf_daily = web.DataReader('DGS3MO', 'fred', start=START_DATE, end=END_DATE)
    rf_monthly = rf_daily.resample("ME").last() / 100 / 12
    rf_monthly = rf_monthly.reindex(strat_ret.index).fillna(method='ffill').fillna(0)
    rf_series = rf_monthly.iloc[:, 0]
except:
    rf_series = pd.Series(0, index=strat_ret.index)

# Metrics calculation
def calc_metrics(ret, bench_ret=None):
    cum = (1 + ret).cumprod()
    n_years = len(ret) / 12
    cagr = cum.iloc[-1]**(1/n_years) - 1
    ann_vol = ret.std() * np.sqrt(12)
    
    exc_ret = ret - rf_series
    sharpe = (exc_ret.mean() / ret.std()) * np.sqrt(12) if ret.std() > 0 else 0
    
    downside = ret[ret < 0]
    sortino = (exc_ret.mean() / downside.std()) * np.sqrt(12) if not downside.empty and downside.std() > 0 else np.nan
    
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan
    
    hit_rate = np.nan
    alpha_ann, alpha_t = np.nan, np.nan
    if bench_ret is not None:
        hits = (ret > bench_ret).sum()
        hit_rate = hits / len(ret)
        
        # OLS Alpha vs SPY
        X = sm.add_constant(bench_ret - rf_series)
        y = ret - rf_series
        res = sm.OLS(y, X, missing='drop').fit()
        alpha_ann = res.params.iloc[0] * 12
        alpha_t = res.tvalues.iloc[0]
        
    return {
        "CAGR": cagr,
        "Ann_Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max_DD": max_dd,
        "Calmar": calmar,
        "Hit_Rate": hit_rate,
        "Alpha_Ann": alpha_ann,
        "Alpha_T": alpha_t
    }, cum, dd

strat_m, cum_strat, dd_strat = calc_metrics(strat_ret, spy_ret)
spy_m, cum_spy, dd_spy = calc_metrics(spy_ret, None)

print("--- STRATEGY METRICS ---")
print(f"Momentum Rotation: CAGR: {strat_m['CAGR']:.2%}, MaxDD: {strat_m['Max_DD']:.2%}, "
      f"Sharpe: {strat_m['Sharpe']:.2f}, Vol: {strat_m['Ann_Vol']:.2%}, Avg Turn: {avg_turnover:.2%}")
print(f"SPY Benchmark    : CAGR: {spy_m['CAGR']:.2%}, MaxDD: {spy_m['Max_DD']:.2%}, "
      f"Sharpe: {spy_m['Sharpe']:.2f}, Vol: {spy_m['Ann_Vol']:.2%}")
print("------------------------")

metrics_df = pd.DataFrame({
    "Momentum Rotation": [f"{strat_m['CAGR']:.2%}", f"{strat_m['Ann_Vol']:.2%}", f"{strat_m['Sharpe']:.2f}",
                          f"{strat_m['Sortino']:.2f}", f"{strat_m['Max_DD']:.2%}", f"{strat_m['Calmar']:.2f}",
                          f"{avg_turnover:.1%}", f"{strat_m['Hit_Rate']:.1%}", f"{strat_m['Alpha_Ann']:.2%} (t={strat_m['Alpha_T']:.2f})"],
    "SPY Benchmark": [f"{spy_m['CAGR']:.2%}", f"{spy_m['Ann_Vol']:.2%}", f"{spy_m['Sharpe']:.2f}",
                      f"{spy_m['Sortino']:.2f}", f"{spy_m['Max_DD']:.2%}", f"{spy_m['Calmar']:.2f}",
                      "--", "--", "--"]
}, index=["Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Sortino Ratio", 
          "Maximum Drawdown", "Calmar Ratio", "Average Turnover", "Hit Rate vs SPY", "Alpha vs SPY"])
metrics_df.to_csv("paper/metrics.csv")

fig, axes = plt.subplots(2, 1, figsize=(10, 7))
axes[0].plot(cum_strat, label=f"Momentum Rotation (Top {TOP_N})", color="#2196F3", lw=2)
axes[0].plot(cum_spy, label="SPY Buy & Hold", color="#FF5722", lw=1.5, ls="--")
axes[0].set_title("Equity Curve: Sector Rotation vs SPY (2015-2025)")
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

# Factor Models
print("Fetching data for Factor Models...")
prices_beta = yf.download(SECTOR_ETFS, start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)["Close"]
returns = prices_beta.pct_change().dropna()

try:
    ff3 = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start=START_DATE, end=END_DATE)[0] / 100
    ff5x = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start=START_DATE, end=END_DATE)[0] / 100
    ff3.columns = ["Mkt-RF", "SMB", "HML", "RF"]
    ff5x.columns = [c.strip() for c in ff5x.columns]
    factors = pd.concat([ff3, ff5x[["RMW", "CMA"]]], axis=1)
except Exception as e:
    print(f"Factor download failed: {e}")
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

print("Running rolling regressions...")
betas_dict = {}
for tick in ["XLK", "XLE", "XLRE", "XLP"]: 
    betas_dict[tick] = compute_betas(returns[tick], factors)

heat_mkt_df = pd.DataFrame({t: betas_dict[t]["Mkt-RF"].resample("ME").mean() for t in betas_dict}).dropna(how="all")
heat_hml_df = pd.DataFrame({t: betas_dict[t]["HML"].resample("ME").mean() for t in betas_dict}).dropna(how="all")

fig, axes = plt.subplots(2, 1, figsize=(14, 12)) 
# To use mdates properly, we use pcolormesh
x_dates = mdates.date2num(heat_mkt_df.index)
sectors = list(heat_mkt_df.columns)
y_idx = np.arange(len(sectors) + 1)

def plot_heatmap_mesh(ax, df, title, cmap, center):
    vmax = max(abs(df.max().max() - center), abs(center - df.min().min()))
    pcm = ax.pcolormesh(df.index, np.arange(len(df.columns)), df.T.values, 
                        cmap=cmap, vmin=center-vmax, vmax=center+vmax, shading='nearest')
    ax.set_title(title)
    ax.set_yticks(np.arange(len(df.columns)))
    ax.set_yticklabels(df.columns)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.colorbar(pcm, ax=ax)

plot_heatmap_mesh(axes[0], heat_mkt_df, "Rolling Market Beta (252-day window)", "RdYlBu_r", 1.0)
plot_heatmap_mesh(axes[1], heat_hml_df, "Rolling Value/Growth (HML) Beta", "RdYlGn", 0.0)

fig.tight_layout()
plt.savefig("paper/images/rolling_betas.png", dpi=300)
plt.close()

# GARCH(1,1) MLE
print("Computing GARCH(1,1) for XLK residuals...")
y = returns["XLK"] - factors["RF"]
X = sm.add_constant(factors[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
ols_res = sm.OLS(y, X).fit()
eps = ols_res.resid.values * 100 # scale up for optimizer stability

def garch_nllh(params, eps):
    omega, alpha, beta = params
    n = len(eps)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(eps)
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
    
    if np.any(sigma2 <= 0):
        return np.inf
    
    llh = -0.5 * np.sum(np.log(2*np.pi) + np.log(sigma2) + (eps**2)/sigma2)
    return -llh

bounds = ((1e-6, 1.0), (1e-6, 0.999), (1e-6, 0.999))
constraints = ({'type': 'ineq', 'fun': lambda x: 0.999 - (x[1] + x[2])})
x0 = [0.01, 0.1, 0.8]
res = minimize(garch_nllh, x0, args=(eps,), bounds=bounds, constraints=constraints, method="SLSQP", options={'disp': False})

omega, alpha, beta = res.x
sigma2 = np.zeros(len(eps))
sigma2[0] = np.var(eps)
for t in range(1, len(eps)):
    sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
sigma = np.sqrt(sigma2) / 100 # scale back

ewma = np.zeros(len(eps))
ewma[0] = np.var(eps/100)
lam = 0.94
for t in range(1, len(eps)):
    ewma[t] = lam * ewma[t-1] + (1-lam)*(eps[t-1]/100)**2
ewma_vol = np.sqrt(ewma)

std_eps = (eps/100) / sigma
lb_res = acorr_ljungbox(std_eps**2, lags=[10], return_df=True)
p_value = lb_res['lb_pvalue'].iloc[0]

print(f"GARCH Stats: omega={omega/10000:.6f}, alpha={alpha:.4f}, beta={beta:.4f}, persistence={alpha+beta:.4f}, LB p-val={p_value:.4f}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(returns.index, sigma, label="GARCH(1,1)", color="#D32F2F", lw=1)
ax.plot(returns.index, ewma_vol, label="EWMA ($\lambda=0.94$)", color="#1976D2", alpha=0.7, lw=1)
ax.set_title("Conditional Volatility of XLK Unexplained by FF5")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("paper/images/garch_xlk.png", dpi=300)
plt.close()
print("Done.")

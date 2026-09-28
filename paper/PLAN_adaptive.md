# Pre-registered plan: adaptive learning rates for volatility management

Written and committed on 2026-09-28, **before any of the experiments below were run**. The grid, the selection rules and the adoption rule are fixed here so that the post-2016 results, which we have already seen for the current version, cannot steer the choice of variant.

## 1. The idea and how it maps onto this strategy

The proposal (from a discussion with an ML-trained roommate): instead of averaging daily data to weekly to cut noise, keep the daily data and set the *learning rate* of each estimator to the horizon it serves, small for daily data, larger for slower quantities. The claim is that the current rule stopped adapting after 2016.

The current rule has three estimators, and each has an implicit learning rate:

| Estimator | Current choice | Implicit learning rate | Adaptive alternative |
|---|---|---|---|
| Variance signal (what the rule reacts to) | Prior calendar month's realized variance | Equal weight on the last ~21 days, zero before | Exponentially weighted daily variance with a chosen half-life; HAR mix of daily, weekly and monthly components |
| Scale constant c (the "normal" exposure level) | Expanding window since 1963 | 1/t: by 2017 each new month moves c by about 1/640, so c has effectively **stopped learning** | Exponentially weighted estimate with constant half-life |
| Combination weights (mean and covariance of original and managed factor) | Expanding window | 1/t | Exponentially weighted mean and covariance |

The roommate's point is exactly right for the scale constant: an expanding window is a stochastic-approximation estimator with step size 1/t, which converges and then cannot track a moving target. A constant step size (exponential weighting) keeps tracking.

## 2. What we expect before looking (honest priors)

1. **Sharpe ratios do not change with leverage.** Adapting only the *level* of exposure (the scale constant) changes a period's Sharpe ratio only through the cap, through costs, and through how exposure is spread across regimes within the period. So baseline variants should matter less than signal variants.
2. **Momentum's own return fell** from 7.3% to 2.9% a year after 2016. No timing rule creates a premium. Adaptation can at most avoid crashes more cheaply or raise the timing covariance.
3. **Faster signals trade more.** Net gains depend on whether the extra reaction outweighs 14 bp per unit of exposure change; break-even costs are reported.
4. Any single variant that looks best after 2016 is likely to be partly luck. That is why the selection rules below never use post-2016 data to pick a variant.

## 3. Fixed grid (Phase 1: monthly rebalancing, all six factors)

Signals, each a forecast of month m's variance made at the end of month m-1 from daily data only:
- **S0** current: realized variance of month m-1 (demeaned daily returns).
- **S1** exponentially weighted daily variance, half-life 5 trading days (about a week), scaled to a month.
- **S2** same, half-life 21 trading days (about a month).
- **S3** same, half-life 63 trading days (about a quarter).
- **S4** HAR (Corsi 2009): next-month log variance regressed in real time (expanding window) on the log of the last day's, last week's and last month's variance; the fitted forecast is the signal.
- **S5** the weekly-averaging idea, tested head to head: realized variance from the last 13 weekly returns, scaled to a month.

Scale constant (same formula as now, sd(f) / sd(f / S)), with weights:
- **B0** current: expanding, equal weights.
- **B1, B2, B3** exponential weights with half-lives of 3, 5 and 10 years.

That is 6 x 4 = **24 managed variants per factor**, each capped at 1.5 and charged 14 bp per unit of exposure change, positions from July 1973 as now.

Combination weights, applied to the selected managed variant:
- **C0** current: expanding mean and covariance.
- **C1, C2** exponential weights with half-lives of 5 and 10 years.

## 4. Selection rules (no post-2016 data is used to choose)

1. **Pre-publication selection.** For each factor, pick the variant with the highest net Sharpe ratio over July 1973 to December 2016. Evaluate it once on January 2017 to August 2026 against the current version (S0-B0) and the original factor.
2. **Real-time selection (the "learning rate on the learning rate").** Each month, hold the variant with the highest trailing 120-month net Sharpe ratio, using only past months. Evaluated from July 1983 (10 years of variant history) to August 2026, and on 2017 onward.
3. **Data-snooping test.** Hansen's (2005) test for superior predictive ability over all 24 variants, with the current version as the benchmark and the monthly Sharpe-ratio contribution as the loss, full sample and 2017 onward.

The full 24-variant grid is reported for every factor (Sharpe by signal and scale constant, before and after 2016), so no variant is hidden.

## 5. Adoption rule (decided now)

A new version replaces the current one as the paper's main specification only if, **for momentum**, rule 2 or rule 3 shows a statistically significant improvement over the current version at 5%, and the improvement has the same sign after 2016 under rule 1. Otherwise the paper keeps the current specification and reports the adaptive analysis as a finding in its own right: adaptive learning rates do (or do not) rescue post-2016 performance. Either outcome is publishable; only the first changes the headline.

## 6. Metrics (same as the current version, so comparisons are like for like)

Net Sharpe ratio; Ledoit-Wolf studentized bootstrap tests against the original factor **and against the current version**; Holm adjustment within families; the M x R decomposition (return channel, risk channel); drawdown at equal volatility; turnover, costs and break-even cost; behaviour in the two stress windows the idea is about (February to April 2020, and March to May 2009).

## 7. Phase 2 (only if Phase 1 shows faster signals help): weekly rebalancing with a no-trade band

For the five Fama-French factors, whose daily files are returns on the same annually formed portfolios as the monthly files, rebalance weekly but trade only when target exposure moves more than a band (0.1 and 0.25) away from current exposure. This tests the "react to COVID without paying costs in normal times" part directly. Momentum is excluded: every French daily momentum file is re-formed daily, so no daily return series exists for the monthly-formed factor.

## 8. Phase 3 (only if Phase 2 is promising): momentum built from CRSP

Construct the monthly-formed momentum factor's daily returns from CRSP via WRDS, to extend Phase 2 to momentum.

## 9. Engineering rules

Every new estimator uses only data dated at or before the end of month m-1 (weekly: the end of the prior week); the no-look-ahead tests are extended to each signal, scale constant and combination rule; all numbers reported in the paper are generated by code.

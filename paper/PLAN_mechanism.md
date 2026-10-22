# Pre-registration: a parameter-free test of momentum's formation beta on CRSP

Committed 2026-09-28, before any CRSP data was downloaded to this machine. The theory is Proposition 6 of the companion project (tanishhky/filtered-risk-timing, docs/THEORY.md, Section 7.2). Its sign and shape were confirmed on French's factors (correlation 0.76 with a theory-free filtered beta), but only after fitting its scale, which came out 2.4 times the prior. This test removes the fit: every input is measured in the cross-section of stocks, before the returns it predicts.

## The prediction
For stocks with market betas b_i (cross-sectional variance sigma_b^2), formation-period log returns r_i = b_i R_F + u_i with u_i independent of b_i, and a winners-minus-losers portfolio of the stocks above c_hi and below c_lo:
beta_WML = sigma_b^2 R_F [f_r(c_hi) / q_hi + f_r(c_lo) / q_lo],
where f_r is the cross-sectional density of formation returns and q_hi, q_lo the shares of stocks beyond each breakpoint. (With equal tail shares q this is Proposition 6; the general form follows from the same Tweedie argument because each tail's term is f_r(c) / q_tail.)

## Data (CRSP monthly via WRDS; licensed, stored locally under data/, never committed)
- Common stocks (share codes 10 and 11 or the CIZ equivalent) on NYSE, AMEX and NASDAQ; returns include delisting returns; a missing delisting return for a performance-related delisting (codes 500 and 520 to 584) is set to -30% (Shumway, 1997).
- Market excess return and risk-free rate: French's monthly Mkt-RF and RF (as in the letter).
- The CRSP release used (last available month) is recorded in the output.

## Construction, for a portfolio formed at the end of month m and held in month m+1
- Formation window: months m-11 to m-1 (French's "2 to 12"); r_i is the sum of log(1 + ret) over it, requiring all 11 returns. R_F is the same sum for the market (Mkt-RF + RF).
- Size: market equity at the end of month m; the size breakpoint is the NYSE median. Prior-return breakpoints: NYSE 30th and 70th percentiles.
- **Replication gate.** The value-weighted 2 x 3 factor, Mom = (SH + BH)/2 - (SL + BL)/2, must correlate at least 0.95 with French's monthly Mom over the overlapping months. If it does not, no test is run until the construction is fixed.
- The theory's object is the **equal-weighted** version of the same 2 x 3 portfolios (primary). French's value-weighted Mom (the letter's traded factor) is secondary, with the prediction applied as an approximation.
- Stock betas b_hat_i: OLS of monthly excess returns on Mkt-RF over months m-71 to m-12 (the 60 months that end before the formation window starts; at least 36 valid), so beta estimation never uses the returns being sorted on.
- sigma_b^2 in each size group: cross-sectional variance of b_hat minus the cross-sectional mean of its squared standard errors (an errors-in-variables correction), floored at zero.
- f_r(c): Gaussian kernel density (Scott's bandwidth) of r_i within each size group, evaluated at that month's NYSE breakpoints; q_hi and q_lo are the group's actual shares beyond them.
- Predicted beta: the average of the two size groups' predictions.

## Tests
- **T1 (the identity on real cross-sections).** The portfolio's formation beta measured directly, the equal-weighted mean b_hat of winners minus losers averaged over size groups, regressed on the predicted beta across months. The prediction holds if the slope is near 1; it measures how much the formula's assumptions (normal betas, independence of betas and idiosyncratic returns) cost.
- **T2 (the economic test, primary).** WML_{m+1} = a + c * beta_pred_m * R_M,m+1 + d * R_M,m+1 + e, Newey-West (12 lags). The size prediction holds with zero fitted parameters if c = 1 and d = 0 (joint Wald test at 5%); the sign and shape hold if c > 0 (one-sided, 5%). Reported with the incremental R^2 over a regression on R_M alone, and against an in-sample Grundy-Martin benchmark (WML on R_M and R_F * R_M with a fitted slope), as the share of that benchmark's explanatory power the parameter-free prediction attains.
- Samples: all holding months with a full beta window (from the early 1930s to the last CRSP month) as primary; July 1963 onward (the letter's sample) as secondary.

## Outcomes, fixed now
- c's 95% interval contains 1 and the Wald test does not reject: **magnitude confirmed**; the paper's headline is a parameter-free prediction of momentum's beta.
- c > 0 significantly but the Wald test rejects: **sign and shape confirmed, magnitude off by the factor c**; the paper reports c and the formula as a shape result.
- c not significantly positive: **the mechanism fails as a monthly prediction**; Proposition 6 leaves the headline, and the paper rests on the letter's evidence, the exposure rule (the market should not be volatility-timed) and the weighted-versus-unweighted reconciliation.

## Merged paper (structure fixed now; results fill it)
1. Puzzle: in real time, volatility management helps momentum and hurts value and the market (letter, M x R channels).
2. Mechanism: momentum's formation beta (Proposition 6) and this CRSP test.
3. Implication: an exposure rule that says what volatility timing should time; the market should not be timed (registered before any result, confirmed).
4. Reconciliation: crash months are predictable (unweighted out-of-sample R^2 5.5% before 2017), but volatility management already captures most of that value (utility-weighted 0.66%).
5. Robustness: better variance forecasts (two-speed filter, adaptive rates) do not change the Sharpe ratio.

## Amendments before any CRSP data (simulated-market validation, 2026-09-28)
The pipeline was run on simulated markets where the formula is exactly true (2,000 stocks, 30 years, normal and t4 idiosyncratic returns; tests/test_mechanism.py). Two specification errors were found and fixed; neither involved any real data.
- **A1.** R_F was defined as the market's total return. The beta multiplies the market's *excess* return; the risk-free part is common to every stock and cancels in the sort. With total returns the predicted beta carried a constant (11 months of the bill rate) that the joint test read as a nonzero d: it rejected the true model in 20 of 20 simulated markets.
- **A2.** Portfolios are sorted on compounded returns, whose first-order coefficient on a stock's beta is R_M (1 - R_M - r_f), summed over the window, not R_M; and a Gaussian kernel understates the density at the breakpoints by about 2 to 3%. R_F now uses the compounded coefficient, and the density at each breakpoint uses the quantile-spacing (sparsity) estimator with the Hall-Sheather bandwidth, the standard estimator of a density at a quantile.
- After both fixes, across 20 simulated markets: mean c = 1.036 (standard deviation 0.060 across markets, mean standard error 0.056), the joint test rejects the true model in 1 of 20 (its nominal size), and the formation-beta slope (T1) averages 1.02. With t4 idiosyncratic returns, where the normal-form formula understates the beta by 15 to 25%, the breakpoint-density form still centres on 1. The remaining upward bias of about 3.6% is reported alongside the real-data estimate.
- **A3 (after the download, before any construction or test).** The current CRSP format (CIZ, `crsp.msf_v2`, through December 2025) incorporates delisting returns into the monthly return: of 23,086 delisting months, 16 have a missing return. The -30% rule would touch those 16 and is not applied; they drop out of that month's portfolio. The legacy table ends in December 2024 and is not used.

## Registered results (run 2026-09-28, CRSP CIZ through December 2025)
- Replication gate: the rebuilt value-weighted factor correlates 0.9989 with French's Mom over 1,181 months (1927-07 to 2025-11). Passed.
- T2 (primary): slope c on the predicted beta is 0.88 (s.e. 0.20) equal-weighted and 0.92 (0.20) for French's factor over 1929 to 2025; 1.00 (0.17) and 1.08 (0.17) from 1963. The level d is -0.15 to -0.22 (s.e. about 0.06). The joint test c = 1, d = 0 rejects in every sample (p < 0.001 over the full sample, 0.014 to 0.026 from 1963). **Registered outcome: sign and shape confirmed; the joint magnitude test fails.** The rejection comes from the level d, not the slope c, whose 95% interval contains 1 in every case. The parameter-free prediction attains 75% to 82% of the incremental R^2 of an in-sample Grundy-Martin regression (0.15 to 0.24 of momentum's monthly variance).
- T1: the directly measured formation beta (mean pre-window beta of winners minus losers) on the prediction has slope 0.55 (full) and 0.49 (from 1963), correlation 0.71 and 0.63. Betas estimated over months m-71 to m-12 are not the betas that drove formation-window returns; if betas drift, this test is attenuated by their persistence, so it is weaker than designed. Reported as registered.

# Follow-up F1, registered after T2 and before computing any idiosyncratic-variance statistic
**Hypothesis (volatility drag).** A compounded 11-month return is lowered by about half the stock's return variance times 11. If idiosyncratic variance s_i^2 rises with beta, the drag pushes high-beta stocks into the loser leg whatever the market did, giving momentum a negative beta at R_F = 0. With the linear projection s_i^2 = a + lambda (b_i - b_bar) + eta_i, the coefficient on b_i in the formation return becomes R_F - (11 / 2) lambda, so
beta_WML^D = sigma_b^2 [R_F - (11 / 2) lambda] [f_r(c_hi) / q_hi + f_r(c_lo) / q_lo].
**Measurement.** s_i^2 is the residual variance of the same 60-month beta regression (months m-71 to m-12); lambda in each size group is Cov(s_hat^2, b_hat) / sigma_b^2 (the estimation noise in b_hat has mean zero given s^2, so it does not bias the covariance). Nothing else changes.
**Validation first.** A simulated market in which idiosyncratic variance rises with beta must show d < 0 with the registered prediction and c = 1, d = 0 with beta^D, before F1 is run on CRSP.
**Pass rule.** F1 explains the level if, with beta^D in place of the registered prediction, the joint test c = 1, d = 0 does not reject at 5% over the full sample; the 1963 sample is reported alongside. If it rejects, the level gap is reported as unexplained. F1 is the only follow-up; no other adjustment to the prediction will be tried.

## F1 validation result (simulation only; F1 was never run on CRSP)
On 16 simulated markets in which idiosyncratic volatility rises with beta, the registered prediction leaves a negative level, as on CRSP (d = -0.039 and -0.079 for the two strengths tested), and lambda is estimated without bias (0.0160 against a true 0.0159). But the first-order drag correction overshoots by about 25% (d = +0.011 and +0.019 after correction; the joint test rejects the corrected model in 38% and 50% of markets). Stocks whose volatility rises with beta are also more dispersed, which violates the independence assumption behind the first-order formula. **Validation failed, so F1 is not run on CRSP**, and, as registered, no other adjustment is tried. The level gap (d about -0.2) is reported as unexplained; simulation supports volatility drag as a mechanism of the right sign, with its magnitude unquantified.

# Link analyses for the merged letter (registered 2026-09-28 before running; both descriptive, neither used to choose anything)
Both use the letter's main specification for momentum (daily-data rule, cap 1.5, net of 14 bp) and the CRSP-predicted beta, over holding months where both exist (July 1973 to December 2025; the CRSP file ends in December 2025).
- **L1 (where the timing gain comes from).** Classify each holding month by the sign of the predicted beta at the end of the prior month: after market declines (beta_pred < 0) or not. Report each group's share of months and its contribution to cov(w, f) (the sum over the group of (w - mean w)(f - mean f), divided by all months), with the group means of f and w. Prediction (THEORY Proposition 7 with the premium proportional to market variance): the beta_pred < 0 group contributes a larger share of the covariance than its share of months.
- **L2 (a mean forecast with no fitted parameter).** mu_hat_t = mu_bar_t + kappa [beta_pred,t F^M_t - mean over s <= t of beta_pred,s F^M_s], with F^M_t the letter's daily-data variance signal for the market, kappa = 4 (the value selected on 1975-2016 data in the companion's registered Phase 2), mu_bar the expanding mean of monthly-formed momentum from July 1963. Forecasts made from June 1973; out-of-sample R^2 against mu_bar, unweighted and weighted by 1 / S^Mom_t (the momentum variance signal; the utility-consistent loss), with one-sided Clark-West tests (whose statistic does not depend on kappa), split at December 2016. Expectation carried over from the companion: the unweighted R^2 exceeds the weighted one, because the forecast's information sits in high-variance months.

## L1 and L2 results (run once, 2026-09-28)
- **L1 (prediction confirmed).** July 1973 to November 2025, 629 months. Months after market declines (beta_pred < 0) are 29% of months and contribute 71% of the covariance between the managed exposure and momentum's return. Momentum earns 1.6% a year in them against 8.6% otherwise, and the managed rule holds 0.56 against 0.82 times the factor.
- **L2 (no forecasting power).** The mean forecast built from the parameter-free beta does not beat the expanding mean: out-of-sample R^2 -0.3% unweighted and -0.7% weighted before 2017 (Clark-West p = 0.15 and 0.24), -1.1% and -1.2% after. The expectation carried over from the companion (unweighted above weighted) does not hold here. The forecast omits the level gap (d about -0.2), whose contribution to the mean scales with market variance; the companion's fitted model, which did forecast, estimated that level (b0 between -0.1 and -0.25). Reported as a negative result.
- Binned figure (deciles of the prediction, French's momentum, 1929 to 2025): realized beta rises from -0.86 to 0.35 as the prediction rises from -0.54 to 0.80; the top decile falls short of the prediction by 0.45.

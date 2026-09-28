# Referee log: "Calm Months, Crash Months" (letter draft)

Each round scrutinizes every step of the method and every claim in the draft. An objection stays open until the approach it targets is the most defensible one available, or the claim is rewritten to say only what the evidence supports. Resolutions record what changed and where.

## Round 1 (2026-09-28): 24 objections

### Data and construction
- **R1-01. Monthly returns are summed daily returns.** Moreira and Muir (2017) and Cederburg et al. (2020) use French's monthly factor returns, which compound. Summing daily long-short returns is an approximation that drifts in volatile months, exactly the months that drive the result. *Fix: take f_m from French's monthly files; use daily data only for realized variance.*
- **R1-02. Realized variance is not demeaned.** Moreira and Muir define RV as the sum of squared deviations of daily returns from the month's mean. The draft sums raw squares. *Fix: use their definition.*
- **R1-03. The factor data are not real-time vintages.** French revises the full history when CRSP and Compustat are updated, so "every input is estimated in real time" overstates what is possible: positions use only past returns, but the returns are today's vintage. *Fix: say so in the data section and soften the abstract.*
- **R1-04. Burn-in, cap level, block length and cost level are researcher choices.** Only the cap and cost have outside anchors (Moreira and Muir's Table IV uses a 1.5 cap and 1, 10, 14 bp). *Fix: cite the anchors, and show the main tests under alternative block lengths.*

### Combination strategy
- **R1-05. The combination is gross of costs while the managed factor is net.** The comparison favours the combination. *Fix: charge the same 14 bp on changes in the combination's total exposure.*
- **R1-06. The combination's exposure is uncapped.** Mean-variance weights on two highly correlated series can imply large gross positions, which contradicts "exposure capped at 1.5". *Fix: write the combination as an effective exposure e_m to the original factor and cap |e_m| at 1.5.*
- **R1-07. The combination weights are estimated on gross managed returns.** An investor would estimate them on the returns actually earned. *Fix: estimate on the net managed series.*

### Inference
- **R1-08. The bootstrap is not studentized.** A percentile bootstrap of a Sharpe difference has poor size in samples like 116 months; Ledoit and Wolf (2008) recommend the studentized circular block bootstrap. The draft cites them for a test it does not run. *Fix: implement their studentized test (HAC standard error on the data, block-based standard error in each resample) and report symmetric studentized confidence intervals.*
- **R1-09. Holm adjustment is applied only to the full sample.** The post-2017 value claim ("remains below 0.01 after adjusting") was computed by hand. *Fix: Holm-adjust each period's family of six tests in code.*
- **R1-10. No sensitivity to block length.** *Fix: report the key tests with 6- and 24-month blocks.*
- **R1-11. The 2017 split was chosen after a 2016 split was examined.** Moving the split strengthened the value result (p = 0.04 at 2016, 0.001 at 2017). The 2017 date has an independent justification (the last month of both published samples), but a reader should see both. *Fix: report the 2016 split as robustness.*

### Claims
- **R1-12. "Momentum's gain shrinks after publication, in line with McLean and Pontiff."** The draft never tests whether the post-2017 gain differs from the pre-2017 gain; it only shows the later gain is not different from zero. *Fix: test the change in the gain across periods and phrase the result as what the test shows.*
- **R1-13. Sharpe comparisons of negative-mean series.** After 2016 value's mean is negative. With negative means, Sharpe rankings can favour the riskier series, so a Sharpe gap alone does not show that timing destroyed value. *Fix: base the value claim on the spanning alpha of the net managed factor on the original, whose sign does not depend on the sign of the mean; report Sharpe ratios as secondary.*
- **R1-14. Drawdowns compared at different volatility.** Managed momentum averages 0.7 times exposure, so a smaller drawdown is partly just less risk. *Fix: compare drawdowns with each series scaled to the original's volatility.*
- **R1-15. "Profitability gains nothing an investor could capture."** Its managed Sharpe is higher (0.46 vs 0.40), just not significantly. *Fix: "no significant real-time gain".*
- **R1-16. "From 1973 to 2026, only momentum improves."** Other factors' point estimates move both ways. *Fix: "only momentum improves significantly".*
- **R1-17. "Momentum is the recurring exception."** Barroso and Detzel find market and momentum partly robust; Cederburg et al. find momentum, ROE and BAB. *Fix: "the most consistent exception", and name the others.*
- **R1-18. "An investor who followed the published advice in 2017 ..."** Moreira and Muir do not advise managing value in particular. *Fix: describe an investor who applied the rule to all six factors.*
- **R1-19. The real-time spanning alpha for profitability is gross of costs.** The draft does not say so. *Fix: say so.*
- **R1-20. The mechanism is asserted, not derived.** The tercile table is descriptive; the draft gives no algebra linking the sign of the timing gain to the covariance between exposure and returns. *Fix: add the decomposition E[w f] = E[w]E[f] + Cov(w, f) and Moreira and Muir's optimal-weight argument (w* = mu_t / (gamma sigma_t^2)), and state when inverse-variance timing is wrong.*

### Presentation and reproducibility
- **R1-21. Numbers in the text are typed by hand.** Every rerun risks a mismatch. *Fix: generate tables and in-text numbers from the code as LaTeX macros.*
- **R1-22. Figure 1 shows the combination as flat before 1978.** It has no returns then; plotting zeros suggests it earned nothing. *Fix: start that line when the combination starts.*
- **R1-23. The paper has one figure and little formal method.** The estimator, test statistic, combination and multiple-testing rule are described in words. *Fix: write out every formula; add figures for (a) Sharpe gains with confidence intervals by period, (b) returns by exposure tercile, (c) the exposure path of momentum and value after 2016.*
- **R1-24. The AI-use declaration is a placeholder.** *Fix: state that generative AI was used to create the simulations and code and to draft the content structure.*

### Round 1 resolutions
All 24 fixed in `src/realtime.py` and the rewritten `paper/letter.tex`:
R1-01 monthly returns now from French's monthly files. **This mattered:** French's *daily* momentum factor is re-formed daily on a day -250 to -21 window, a different factor from the monthly-formed, month -12 to -2 factor the literature uses (correlation of monthly returns 0.97, mean absolute gap 0.68% a month). The first draft and the SSRN preprint 7113079 summed the daily factor. R1-02 RV demeaned as in Moreira and Muir. R1-03 vintage caveat in the data section. R1-04 anchors cited, block-length robustness reported. R1-05 to R1-07 combination is net, capped (|e| <= 1.5) and estimated on net returns. R1-08 Ledoit-Wolf studentized circular block bootstrap with Bartlett HAC standard error, symmetric studentized intervals. R1-09 Holm within every family of six. R1-10 6- and 24-month blocks. R1-11 2016 split reported. R1-12 cross-period change test. R1-13 value claim rests on the net spanning alpha. R1-14 drawdowns at equal volatility. R1-15 to R1-19 wording. R1-20 mean decomposition added. R1-21 all tables and in-text numbers generated as LaTeX. R1-22 combination plotted from its first month. R1-23 formulas and four figures. R1-24 declaration written.

**What the fixes changed:** momentum still improves over the full sample (0.43 to 0.90, Holm p = 0.028), but (a) its real-time combination no longer beats the original once it pays the same costs and cap (p = 0.10); (b) after 2016 managed momentum matches the original (0.21 vs 0.23); (c) the value result weakens: net alpha -1.6% (t = -2.7) but Sharpe test p = 0.06 (Holm 0.36), and t = -1.6 with a 2016 split.

## Round 2 (after Round 1 fixes): 8 objections
- **R2-01. Momentum's realized variance still comes from the daily-formed factor.** No daily series exists for the monthly-formed factor; Moreira and Muir face the same constraint. *Fix: disclose in the data section that RV for momentum is a proxy built from French's daily factor.*
- **R2-02. The headline claims no longer match the evidence.** The abstract and introduction still say the momentum combination wins, that managing value lowered its Sharpe after 2016, and imply a post-2016 momentum gain. *Fix: rewrite every claim from the regenerated numbers; the value result is reported as suggestive and split-sensitive, not as a finding.*
- **R2-03. No test proves the absence of look-ahead.** The repo's own engineering rule requires one. *Fix: add `tests/test_no_lookahead.py`: perturb all data after month t, confirm every exposure up to t is unchanged, for the managed factor and the combination.*
- **R2-04. The robustness table mixes samples without saying so.** Momentum columns are full-sample tests; value columns are post-split. *Fix: label the column groups.*
- **R2-05. The mechanism is shown with terciles only.** Terciles are descriptive and hide the average-exposure effect that explains the post-2016 momentum result (average exposure fell from 0.76 to 0.31). *Fix: add the exact decomposition mean(w f) = mean(w) mean(f) + cov(w, f), minus costs, by factor and period.*
- **R2-06. The combination's cap binds often and this is not reported.** For momentum |e| hits 1.5 in 28% of months. *Fix: report it, since it is part of why the capped combination is weaker than Cederburg et al.'s uncapped one.*
- **R2-07. The data-sample HAC estimator differs from Ledoit and Wolf's prewhitened QS kernel.** *Fix: state the choice (Bartlett, 12 lags) and rely on the block-length robustness, which leaves the conclusions unchanged.*
- **R2-08. Text sometimes quotes unadjusted and sometimes Holm p-values without saying which.** *Fix: label every p-value in the text.*
- (Outside the paper) SSRN 7113079, the site page and LinkedIn report momentum results computed from the daily-formed factor. Flag to the author.

### Round 2 resolutions
R2-01 disclosed in the data section. R2-02 every claim rewritten from regenerated numbers; the value result is reported as a warning. R2-03 `tests/test_no_lookahead.py` (3 tests: managed exposure, combination exposure, and a control that later positions do change). R2-04 column groups labelled. R2-05 exact mean decomposition table. R2-06 reported. R2-07 stated; lag length equals block length. R2-08 every p-value labelled.

## Round 3: 11 objections
- **R3-01. The post-2016 momentum explanation was wrong.** Sharpe ratios are scale-invariant, so low average exposure cannot by itself explain a missing Sharpe gain.
- **R3-02. The mechanism statement was wrong.** With a constant expected return the covariance is zero in expectation, not positive; the Moreira-Muir gain then comes from risk reduction. *Fix (both): the exact two-channel identity SR(f^sigma)/SR(f) = M x R, with M = 1 + (cov - cost)/(w f) and R = w sd(f)/sd(f^sigma); rewritten mechanism text and table.*
- **R3-03.** Figure 1 legend overlapped the data. *Moved above the panels.*
- **R3-04.** Pre-2017 momentum is not significant after the six-factor adjustment in that window (p = 0.116); not stated. *Stated.*
- **R3-05.** "The gain belongs to the years before 2017" overclaims given the change test (p = 0.118). *"Concentrated in"; the decade is too short to prove the gain has gone.*
- **R3-06.** HAC lag wording. *Lag length equals block length.*
- **R3-07.** "The cap is one reason the combination falls short" was untested. *Tested: the cap barely matters; costs remove significance (no-cost, no-cap combination p = 0.03).*
- **R3-08.** Negative macros rendered as hyphens inside math. *\ensuremath{-}.*
- **R3-09.** Conclusion mixed full-sample and post-2016 combination claims. *Separated.*
- **R3-10.** Change test's independence assumption unstated. *Stated.*
- **R3-11.** Word count unverified. *About 1,860 words of main text against FRL's 2,500.*

## Round 4: 4 objections
- **R4-01.** The same test reported different p-values in different tables (baseline robustness 0.006 vs 0.005) because bootstrap draws continued one random stream. *Each test now uses its own fixed-seed generator; identical inputs give identical p-values.*
- **R4-02.** Table 3 caption numbered columns wrongly and said "sum" where cost is subtracted. *Rewritten as an equation.*
- **R4-03.** Figures drifted past the Conclusion. *\FloatBarrier.*
- **R4-04.** "Profitability, a success in spanning tests" lacked a source. *Attributed to Cederburg et al. (their RMW alpha is significant at 1%).*

## Round 5: 4 objections
- **R5-01.** Factual error: the 1.5 cap was called "the tighter" of Moreira and Muir's caps; their caps are 1 and 1.5. *"The higher".*
- **R5-02.** "The cap is not what holds it back" overclaimed: removing the cap moves the Sharpe ratio 0.74 to 0.77, removing costs 0.77 to 0.85. *"Costs, more than the cap".*
- **R5-03.** Profitability was called instructive but not explained with the channels. *M = 0.82, R = 1.49.*
- **R5-04.** Conclusion mixed units ("removes more risk than it costs"). *Stated as M x R > 1.*

## Round 6: 2 objections
- **R6-01.** No figure explained the approach (samples, information timing). *Study-design figure added (Figure 1).*
- **R6-02.** Burn-in length untested. *Resolved by construction: the scale constant uses every month since 1963 whatever the burn-in, so the burn-in only sets the first trading month (verified: a 5-year burn-in gives identical positions from 1973). The meaningful check, a later start (July 1978), is in Table 4.*

## Round 7: 3 objections
- **R7-01.** Table 4 caption said "full sample" for every momentum row, including the 1978 start. *Fixed.*
- **R7-02.** "Published samples start in 1926" for momentum; Cederburg et al.'s starts January 1927. *"Late 1920s".*
- **R7-03.** "The two channels largely cancel" for profitability overstated it: M x R = 1.22. *"Gives back part of ... too small to be statistically distinguishable from zero".*

## Round 8: 0 objections
Checked: no unresolved references; every number in the Results text is generated by `src/realtime.py` except definitional constants (split dates, the 0.2 exposure threshold, 95% intervals, equation numbers); 3/3 look-ahead tests pass; clean LaTeX build (no errors, undefined macros or overfull boxes); no em or en dashes; 11 references, all verified against publisher or index records.

**Outside the paper (for the author):** SSRN preprint 7113079, the site's /research/volatility-managed-factors page and LinkedIn report momentum results computed by summing French's daily-formed momentum factor, a different factor from the monthly-formed one the literature uses.

# Second review cycle: adaptive learning rates (2026-09-28)

After the pre-registered learning-rate study (PLAN_adaptive.md) met its adoption rule, the paper's main specification changed to the daily-data rule (S2-B0) and Section 4 was added. The loop was rerun on the new version.

## Round A1: 9 objections
- **A1-01.** Main text about 2,530 words, over FRL's 2,500. *Trimmed to about 2,460 without removing results or equations.*
- **A1-02.** The paper did not say the grid was designed after the canonical rule's post-2016 failure was known. *Disclosed in Section 4, with that as the reason no rule uses post-2016 data to choose.*
- **A1-03.** Table 5 showed a significant value result (rule-1 choice S4-B1 vs canonical after 2016, p = 0.008; SPA p = 0.008) that the text never discussed. *Discussed and tied to the B1 column of Figure 6.*
- **A1-04.** "Adaptive scale constants lower momentum's pre-2017 Sharpe ratio" did not say these used the canonical signal. *Stated.*
- **A1-05.** "No signal held more than 0.07 of the market in April 2020" was computed over 4 of 6 signals. *Computed over all six.*
- **A1-06.** "It lost -5.3%" read as a double negative. *"Its net return was -5.3%".*
- **A1-07.** Heatmap text too small to read. *Larger figure, compact labels.*
- **A1-08.** Stress-figure legend covered data. *Moved above the panels.*
- **A1-09.** Stress figure did not show the adopted signal. *S2 replaces S3.*

## Round A2: 2 objections
- **A2-01.** "Which is why a significant spanning alpha did not become a significant real-time gain" (profitability) was too causal: M x R still leaves a gain of about a fifth. *"Leaving a gain too small to be statistically significant".*
- **A2-02.** Credit for value's post-2016 improvement to the adaptive scale constant needed its evidence. *Pointed to column B1 of Figure 6.*

## Round A3: 0 objections
Checked: 5/5 look-ahead tests pass (including every adaptive signal, scale constant and combination); S0-B0 reproduces the canonical rule and S2-B0 the main specification exactly (asserted in code); clean build; no unresolved references; no em or en dashes; 13 references, all cited and verified; every result number generated by code; about 2,460 words of main text.

# Merged letter (mechanism-letter branch, 2026-09-28): review loop to zero objections

## Round M1: 10 objections, all fixed
- M1-01. The introduction said every test in Sections 4 and 5 was specified before the CRSP data were downloaded; L1, L2 and F1 were registered after the download (before they were run). Corrected.
- M1-02. The mechanism sample was labelled 1929 to 2025; with a full 60-month beta window the first holding month is July 1932. The start year is now computed from the data.
- M1-03. L1 measures each group's share of the exposure-return covariance, not of the whole timing gain. Abstract and conclusion say so.
- M1-04. "Its size has only been estimated by regression" claimed more about the literature than was checked. Replaced by what we do.
- M1-05. The learning-rate sentence compared the trailing-selection rule with the wrong benchmark; it lost to the canonical rule (0.70 against 0.82).
- M1-06. L2's failure was attributed to the omitted level with "because"; it is a likely reason, not a tested one.
- M1-07. The market's M below one was presented as the premium effect; it is consistent with it.
- M1-08. The conclusion's "when variance is high" was not measured by L1, and the market claim needs Merton's assumption. Both fixed.
- M1-09. The post-2016 sentence reported one period's Sharpe ratios as if they were the change; both periods now given.
- M1-10. Table 3 overran the margin.

## Round M2: 5 objections, all fixed
- M2-01. Abstract and introduction stated "the market should not be timed" without its condition (a premium rising with variance).
- M2-02. "Explains 24% of the variance beyond the market" was imprecise; it is a 24-point rise in R^2 over a regression on the market, 79% of an in-sample fit's rise.
- M2-03. "Crowd the cross-section" was vague: heavier tails put more stocks near the 30th and 70th percentiles at the same variance.
- M2-04. Abstract grammar ("and none of the other factors").
- M2-05. "The slope is right" overstated a slope of 0.92 with standard error 0.19; now "matches the prediction".

## Round M3: 0 objections

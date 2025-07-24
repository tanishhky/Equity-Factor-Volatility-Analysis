# Quantitative Equity Sector Analysis & Strategy Backtesting

This project conducts a comprehensive quantitative analysis of the 11 GICS sectors of the U.S. stock market. It utilizes established academic factor models to deconstruct sector returns, identify risk exposures, and evaluates a dynamic sector rotation strategy through backtesting.

The primary goal is to move beyond simple performance metrics and understand the underlying drivers of sector risk and return, culminating in a practical, rule-based investment strategy.

## Methodology: Using Sector ETFs as Proxies

To analyze sector-level performance, this project uses the 11 Sector SPDR ETFs. These are industry-standard, highly liquid, and investable instruments that serve as proxies for the performance of the Global Industry Classification Standard (GICS) sectors of the S&P 500. This approach allows for the analysis of broad sector trends without the complexity of building and maintaining a custom index from hundreds of individual stocks.

The following ETFs were used:

* **XLK**: Technology
* **XLF**: Financials
* **XLV**: Health Care
* **XLY**: Consumer Discretionary
* **XLP**: Consumer Staples
* **XLE**: Energy
* **XLI**: Industrials
* **XLB**: Materials
* **XLU**: Utilities
* **XLRE**: Real Estate
* **XLC**: Communication Services

---

## Key Analyses

The project is structured into three main parts:

### 1. Expanded Factor Model Analysis

This section applies multiple asset pricing models to determine the drivers of sector returns. The goal is to quantify a sector's sensitivity to systematic risk factors (its Betas) and measure its idiosyncratic, skill-based performance (its Alpha).

* **Models Used:** Fama-French 3-Factor, Carhart 4-Factor, and the Fama-French 5-Factor models.

### 2. Sector Rotation Strategy Backtesting

A quantitative momentum-based sector rotation strategy is developed and backtested against the S&P 500 (SPY) benchmark. The strategy invests in the top 3 sectors with the highest momentum over the preceding 6 months, rebalancing monthly.

### 3. Cross-Sectional & Macroeconomic Risk Analysis

This analysis uses rolling regressions to calculate time-varying factor betas for each sector. The results are visualized as heatmaps to show how sector risk profiles evolve in response to different market regimes.

---

## Findings & Importance

### Factor Model Insights

* **High Explanatory Power**: The Fama-French models explained a very high portion (**92-93%**) of the tech sector's daily returns, validating their use for risk decomposition.
* **Quantified Sector DNA**: The analysis statistically confirmed that the technology sector is **21% more volatile than the market** (`Mkt_RF` beta > 1), has a strong bias toward **large-cap stocks** (negative `SMB` beta), and exhibits a clear **growth characteristic** (negative `HML` beta).
* **Modern Factors Matter**: Adding **Momentum** (`MOM`) and **Profitability** (`RMW`) factors significantly improved the model's accuracy. This demonstrates an understanding of modern asset pricing theory beyond the basic models and highlights that sector performance is also driven by trend-following and corporate financial health.

***Importance for Risk/Quant Roles***: This demonstrates the ability to use econometric models to break down portfolio returns, identify key risk exposures, and determine if performance is due to market factors or genuine alpha—a core task in performance attribution and risk management.

### Sector Rotation Strategy Performance

* **Superior Risk Management**: The most critical finding was that the momentum strategy, while achieving similar returns to the benchmark (16.51% vs. 17.40% CAGR), did so with **significantly less risk**. Its maximum drawdown was **-15.97%** compared to the S&P 500's **-23.97%**.
* **Practical Value**: This result is highly important as it shows the strategy's effectiveness in preserving capital during market downturns, a primary objective for any risk-conscious investor or portfolio manager.

### Rolling Beta & Macroeconomic Insights

* **Dynamic Risk Profiles**: The heatmaps reveal that sector risk profiles are not static. For example, the market beta heatmap shows a market-wide **correlation spike during the 2020 crash**, visually demonstrating how diversification benefits can evaporate during a crisis.
* **Visualizing Market Regimes**: The value (HML) factor heatmap effectively captures the major **market rotation from growth to value stocks in 2022** as inflation and interest rates rose. This ability to visualize and interpret regime changes is crucial for tactical asset allocation and risk modeling.

---

## Technologies Used

* Python 3.9+
* Pandas & NumPy
* yfinance & pandas-datareader
* Statsmodels
* scikit-learn
* Matplotlib & Seaborn

## Setup and Usage

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/Equity-Factor-Volatility-Analysis.git](https://github.com/your-username/Equity-Factor-Volatility-Analysis.git)
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the analysis by executing the cells in the Jupyter Notebook (`main.ipynb`).

# DSA5205 Project 2: Design, Validate, and Analyze an Original Stock-Trading Strategy

## 1. Overview

This project presents a systematic, multi-factor quantitative trading strategy designed to achieve alpha in the U.S. technology stock market. The strategy operates on a monthly rebalancing schedule and leverages a machine learning model—Ridge Regression—to predict stock returns based on a combination of well-established financial factors.

The entire workflow, from data acquisition and feature engineering to model training, backtesting, and performance analysis, is implemented in the accompanying Jupyter Notebook (`DSA5205 Project2.ipynb`). The strategy is benchmarked against a passive buy-and-hold strategy using the SPY (S&P 500 ETF).

-   **Market**: U.S. large-cap technology stocks.
-   **Asset Pool**: 15 selected tech tickers (e.g., AAPL, GOOGL, MSFT, AMZN).
-   **Trading Frequency**: Monthly rebalancing.
-   **Model**: Ridge Regression with hyperparameter tuning.
-   **Backtest Period**: Jan 2018 - Oct 2024.

## 2. Strategy & Methodology

The core of the strategy is to build a predictive model that ranks stocks based on their expected next-month returns. The portfolio is constructed by taking long positions in the top 20% of stocks with the highest predicted returns.

### 2.1. Feature Engineering (Factors)

Five distinct factors are engineered to capture different dimensions of stock characteristics. Each factor is grounded in economic rationale:

1.  **Momentum**: Based on the cumulative returns over the past 11 months (skipping the most recent month). Rationale: stocks that have performed well in the past tend to continue performing well.
2.  **Value**: Measured by the Book-to-Price (B/P) ratio, fetched from the Alpha Vantage API. Rationale: stocks with a high B/P ratio may be undervalued by the market.
3.  **Volatility**: Calculated as the standard deviation of daily returns over the past 6 months. Rationale: low-volatility stocks often provide better risk-adjusted returns.
4.  **Size**: Represented by the natural logarithm of the company's market capitalization. Rationale: captures the size premium effect.
5.  **Short-term Reversal**: Based on the previous month's return. Rationale: captures short-term mean-reversion effects in stock prices.

### 2.2. Predictive Modeling

-   **Model Choice**: **Ridge Regression** was selected for its ability to handle multicollinearity between factors and prevent overfitting through L2 regularization.
-   **Training & Validation**: A **rolling window** approach is used for backtesting to simulate a real-world trading scenario.
    -   **Training Window**: 36 months of historical data.
    -   **Prediction Horizon**: 1 month.
-   **Hyperparameter Tuning**: At each step of the rolling window, **GridSearchCV** with **TimeSeriesSplit** is used to find the optimal `alpha` (regularization strength) for the Ridge model, ensuring the model is always adapted to the most recent data available.

### 2.3. Portfolio Construction & Backtesting

1.  **Signal Generation**: At the end of each month, the trained model predicts the next-month return for all stocks in the pool.
2.  **Portfolio Formation**: Stocks are ranked by their predicted returns. A long-only, **equal-weighted** portfolio is formed with the top 20% of stocks.
3.  **Execution**: The portfolio is rebalanced at the beginning of the next month.
4.  **Transaction Costs**: A realistic transaction cost of **0.1% (10 bps)** is applied to all trades based on the portfolio turnover rate to provide a more accurate performance assessment.

## 3. How to Run the Code

### 3.1. Requirements

The project is built using Python 3. The required libraries are listed in `requirements.txt`.

-   pandas
-   numpy
-   yfinance
-   scikit-learn
-   statsmodels
-   matplotlib
-   requests

### 3.2. Installation

1.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### 3.3. Configuration

Before running the notebook, you must set your Alpha Vantage API key. Open `DSA5205 Project2.ipynb` and replace `"Your API"` with your actual key in the following line:

```python
AV_API_KEY = "Your API"
```

### 3.4. Execution

Simply run all cells in the Jupyter Notebook `DSA5205 Project2.ipynb` from top to bottom. The notebook is structured to perform all steps sequentially:

1.  Data fetching and preprocessing.
2.  Feature calculation.
3.  The final backtesting function `backtest_strategy_final`.
4.  Performance analysis and visualization.

The final output will include performance metrics tables and plots comparing the strategy against the SPY benchmark.

## 4. Project Structure

```
.
├── DSA5205 Project2.ipynb      # The main Jupyter Notebook with all code and analysis.
└── README.md                   # This file.
```

## 5. Performance & Analysis

The notebook provides a comprehensive evaluation of the strategy's performance, covering:

-   **Overall Performance Metrics**: Annualized Return, Volatility, Sharpe Ratio, Max Drawdown, and Turnover.
-   **Cumulative Returns**: A plot showing the strategy's equity curve against the SPY benchmark.
-   **Risk-Adjusted Returns**: CAPM analysis to calculate the strategy's **Alpha** and **Beta**.
-   **Predictive Power**: Out-of-Sample (OOS) R² to measure the model's forecasting accuracy.
-   **Annual Performance**: Year-by-year breakdown of returns and Sharpe ratios.

The results are thoroughly analyzed to determine the strategy's effectiveness, its behavior in different market conditions, and its statistical significance.

## 6. Conclusion & Future Work

This project successfully implements an end-to-end quantitative trading strategy. The analysis reveals that the strategy generated a positive monthly alpha of 1.25%, though it was not statistically significant. Furthermore, the strategy demonstrated significantly higher market risk than the benchmark, with a Beta of approximately 1.5.

Potential areas for future improvement include:
-   Expanding the asset pool to other sectors or markets.
-   Incorporating more diverse factors (e.g., sentiment, alternative data).
-   Experimenting with more advanced non-linear models (e.g., Gradient Boosting, Neural Networks).
-   Implementing a more sophisticated portfolio optimization technique instead of equal weighting.

# Project Notes

## Evaluation Methodology
Expanding-origin cross-validation (rolling origin):
- Start with an initial training window (e.g. 60% of sample observations).
- Train model and make a 1-step-ahead out-of-sample forecast.
- Expand the estimation window by 1 observation, re-estimate parameters, and forecast the next period.
- Evaluate forecast accuracy against actual realizations using Mean Absolute Error (MAE) and Mean Squared Error (MSE).

## Data Sources

### High-Frequency Predictors
- **Bond Yields**: Government of Canada Benchmark Bond Yields (2-Year)  
  https://www.bankofcanada.ca/rates/interest-rates/canadian-bonds/
- **Exchange Rates**: Bank of Canada Daily CAD/USD  
  https://www.bankofcanada.ca/rates/exchange/daily-exchange-rates/
- **Commodity Price Index**: Bank of Canada BCPI (Total, Energy, Non-Energy)  
  https://www.bankofcanada.ca/rates/price-indexes/bcpi/
- **Interest Rates**: Target Overnight Rate and CORRA  
  https://www.bankofcanada.ca/rates/interest-rates/canadian-interest-rates/
- **Gasoline Prices**: FRED US Regular Conventional Gas Price (weekly proxy)  
  https://fred.stlouisfed.org/series/GASREGW/
- **Residential Property Prices**: BIS Canada Property Price Index  
  https://fred.stlouisfed.org/series/QCAR628BIS

### Target Variable
- **Canadian CPI (Monthly)**: Statistics Canada Table 18-10-0006-01  
  https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000601

## Model Specifications
- **Spec 1 (Univariate MIDAS)**: Weekly gas price log-returns
- **Spec 2 (Bivariate MIDAS)**: Weekly gas price log-returns + Target interest rate changes
- **Spec 3 (Commodity Split MIDAS)**: Weekly BCPI energy returns + Weekly BCPI non-energy returns
- **Spec 4 (Trivariate MIDAS)**: Weekly gas prices + Total BCPI + 2Y bond yield changes
- **Benchmarks**: Monthly AR(1), Monthly OLS, ARIMA, ARIMAX, VAR(1), Random Forest, Gradient Boosting
# STAT 499: Mixed Data Sampling (MIDAS) Nowcasting for Canadian CPI

This repository implements Mixed Data Sampling (MIDAS) regression models and benchmark methods (ARIMA, ARIMAX, VAR, OLS, Random Forest, Gradient Boosting) to nowcast monthly Canadian Consumer Price Index (CPI) inflation using high-frequency weekly economic and financial predictors.

## Project Structure

- `clean/`: Data cleaning scripts for high-frequency and low-frequency indicators.
- `cleaned_predictors/`: Processed weekly/monthly series (gasoline prices, interest rates, bond yields, BCPI commodity indexes, exchange rates, property prices).
- `target_variables/`: Canadian monthly CPI and month-over-month inflation.
- `models/`: MIDAS model implementations using exponential Almon lag polynomials (`models/midas/`) and numerical utilities (`models/utils/`).
- `EDA_analysis.py`: Exploratory data analysis and correlation matrix generation.
- `model_comparison.py`: Expanding-window out-of-sample performance evaluation across model families.
- `predictive_value.py`: Evaluation of high-frequency predictive value over low-frequency baselines.
- `plot_model_forecasts.py`: Out-of-sample forecast trajectories and error diagnostic visualizations.

## Requirements

Python environment:
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`

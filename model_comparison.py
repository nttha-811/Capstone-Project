import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from models.utils.math_utils import almon_weights, midas_loss_bivariate, midas_loss_univariate
from models.utils.load_lags import load_weekly_lag_matrix

gas_path = "cleaned_predictors/gas_price.csv"
rate_path = "cleaned_predictors/cleaned_interest_rates.csv"
cpi_path = "target_variables/cleaned_canadian_cpi.csv"

df_gas = load_weekly_lag_matrix(gas_path, "diff_log_gas")
df_rate = load_weekly_lag_matrix(rate_path, "diff_target_rate")

df_cpi = pd.read_csv(cpi_path)
df_cpi.columns = df_cpi.columns.str.strip().str.lower()
cpi_date_col = [c for c in df_cpi.columns if "date" in c][0]
df_cpi[cpi_date_col] = pd.to_datetime(df_cpi[cpi_date_col])
df_cpi = df_cpi.sort_values(cpi_date_col).set_index(cpi_date_col)

df_merged = df_gas.join(df_rate, lsuffix="_gas", rsuffix="_rate", how="inner")
df_merged = df_merged.join(df_cpi[["inflation"]], how="inner").dropna()

df_merged["monthly_avg_gas"] = df_merged[["Lag_0_gas", "Lag_1_gas", "Lag_2_gas", "Lag_3_gas"]].mean(axis=1)
df_merged["monthly_avg_rate"] = df_merged[["Lag_0_rate", "Lag_1_rate", "Lag_2_rate", "Lag_3_rate"]].mean(axis=1)
df_merged["lag_1_inflation"] = df_merged["inflation"].shift(1)

df_clean = df_merged.dropna().copy()

print(f"Timeline: {df_clean.index.min().date()} to {df_clean.index.max().date()}")
print(f"Observations: {len(df_clean)}\n")

y = df_clean["inflation"].values
y_lag1 = df_clean["lag_1_inflation"].values

X_gas_weekly = df_clean[["Lag_0_gas", "Lag_1_gas", "Lag_2_gas", "Lag_3_gas"]].values
X_rate_weekly = df_clean[["Lag_0_rate", "Lag_1_rate", "Lag_2_rate", "Lag_3_rate"]].values
X_combined_weekly = np.hstack([X_gas_weekly, X_rate_weekly])

X_gas_monthly = df_clean["monthly_avg_gas"].values
X_rate_monthly = df_clean["monthly_avg_rate"].values
X_combined_monthly = np.column_stack([X_gas_monthly, X_rate_monthly])

initial_train_size = int(len(y) * 0.6)

preds_ar1 = []
preds_ols_monthly = []
preds_midas_gas = []
preds_midas_bivariate = []
preds_rf = []
preds_gbm = []
actuals = []

init_uni = [0.001, 0.05, -0.1, -0.05]
init_bi = [0.001, 0.05, 0.02, -0.1, -0.05, -0.1, -0.05]

for t in range(initial_train_size, len(y)):
    y_tr = y[:t]
    y_lag1_tr = y_lag1[:t]
    X_gas_w_tr = X_gas_weekly[:t]
    X_rate_w_tr = X_rate_weekly[:t]
    X_comb_w_tr = X_combined_weekly[:t]
    X_comb_m_tr = X_combined_monthly[:t]

    y_te = y[t:t+1]
    y_lag1_te = y_lag1[t:t+1]
    X_gas_w_te = X_gas_weekly[t:t+1]
    X_rate_w_te = X_rate_weekly[t:t+1]
    X_comb_w_te = X_combined_weekly[t:t+1]
    X_comb_m_te = X_combined_monthly[t:t+1]

    # AR(1)
    A_ar1 = np.vstack([np.ones(len(y_tr)), y_lag1_tr]).T
    coeff_ar1, _, _, _ = np.linalg.lstsq(A_ar1, y_tr, rcond=None)
    pred_ar1 = coeff_ar1[0] + coeff_ar1[1] * y_lag1_te[0]
    preds_ar1.append(pred_ar1)

    # Monthly OLS
    A_ols = np.vstack([np.ones(len(y_tr)), X_comb_m_tr[:, 0], X_comb_m_tr[:, 1]]).T
    coeff_ols, _, _, _ = np.linalg.lstsq(A_ols, y_tr, rcond=None)
    pred_ols = coeff_ols[0] + coeff_ols[1] * X_comb_m_te[0, 0] + coeff_ols[2] * X_comb_m_te[0, 1]
    preds_ols_monthly.append(pred_ols)

    # Univariate MIDAS
    res_u = minimize(midas_loss_univariate, init_uni, args=(X_gas_w_tr, y_tr), method="Nelder-Mead")
    w_u = almon_weights(res_u.x[2], res_u.x[3], max_lag=4)
    pred_u = res_u.x[0] + res_u.x[1] * np.dot(X_gas_w_te, w_u)
    preds_midas_gas.append(pred_u[0])

    # Bivariate MIDAS
    res_b = minimize(midas_loss_bivariate, init_bi, args=(X_gas_w_tr, X_rate_w_tr, y_tr), method="Nelder-Mead")
    w1_b = almon_weights(res_b.x[3], res_b.x[4], max_lag=4)
    w2_b = almon_weights(res_b.x[5], res_b.x[6], max_lag=4)
    pred_b = res_b.x[0] + (res_b.x[1] * np.dot(X_gas_w_te, w1_b)) + (res_b.x[2] * np.dot(X_rate_w_te, w2_b))
    preds_midas_bivariate.append(pred_b[0])

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    rf.fit(X_comb_w_tr, y_tr)
    preds_rf.append(rf.predict(X_comb_w_te)[0])

    # Gradient Boosting
    gbm = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42)
    gbm.fit(X_comb_w_tr, y_tr)
    preds_gbm.append(gbm.predict(X_comb_w_te)[0])

    actuals.append(y_te[0])

models = {
    "AR(1) Baseline": preds_ar1,
    "Monthly Averaged OLS": preds_ols_monthly,
    "Random Forest": preds_rf,
    "Gradient Boosting": preds_gbm,
    "Univariate MIDAS (Gas)": preds_midas_gas,
    "Bivariate MIDAS (Gas + Rate)": preds_midas_bivariate,
}

mae_ar1 = mean_absolute_error(actuals, preds_ar1)

print("Out-of-Sample Nowcasting Performance Comparison:")
print(f"{'Model':<30} | {'Input Frequency':<18} | {'OOS MAE':<10} | {'OOS MSE':<12} | {'Gain vs AR(1)':<12}")
print("-" * 92)

for name, preds in models.items():
    mae = mean_absolute_error(actuals, preds)
    mse = mean_squared_error(actuals, preds)
    gain = ((mae_ar1 - mae) / mae_ar1) * 100
    
    freq = "Monthly Only" if "AR(1)" in name else "Monthly Aggregated" if "OLS" in name else "Weekly High-Freq"
    gain_str = "Baseline" if "AR(1)" in name else f"{gain:+.2f}%"
    
    print(f"{name:<30} | {freq:<18} | {mae:.6f}   | {mse:.8e}   | {gain_str:<12}")




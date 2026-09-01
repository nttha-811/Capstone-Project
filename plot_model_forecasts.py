import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from models.utils.load_lags import load_weekly_lag_matrix
from models.utils.math_utils import (
    almon_weights,
    ar_midas_loss_bivariate,
    ar_midas_loss_univariate,
    midas_loss_bivariate,
    midas_loss_univariate,
)

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 9

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

df_merged["inflation_pct"] = df_merged["inflation"] * 100.0
df_merged["monthly_avg_gas"] = df_merged[["Lag_0_gas", "Lag_1_gas", "Lag_2_gas", "Lag_3_gas"]].mean(axis=1) * 100.0
df_merged["monthly_avg_rate"] = df_merged[["Lag_0_rate", "Lag_1_rate", "Lag_2_rate", "Lag_3_rate"]].mean(axis=1)
df_merged["lag_1_inflation_pct"] = df_merged["inflation_pct"].shift(1)

df_clean = df_merged.dropna().copy()
dates = df_clean.index

y = df_clean["inflation_pct"].values
y_lag1 = df_clean["lag_1_inflation_pct"].values

X_gas_w = df_clean[["Lag_0_gas", "Lag_1_gas", "Lag_2_gas", "Lag_3_gas"]].values * 100.0
X_rate_w = df_clean[["Lag_0_rate", "Lag_1_rate", "Lag_2_rate", "Lag_3_rate"]].values
X_comb_w = np.hstack([X_gas_w, X_rate_w])

X_gas_m = df_clean["monthly_avg_gas"].values
X_rate_m = df_clean["monthly_avg_rate"].values
X_comb_m = np.column_stack([X_gas_m, X_rate_m])
X_ml = np.column_stack([y_lag1, X_comb_w])

df_var = df_clean[["inflation_pct", "monthly_avg_gas", "monthly_avg_rate"]].copy()

initial_train_size = 48
total_obs = len(y)

model_names = [
    "ARIMA(1,0,0) Baseline",
    "ARIMAX(1,0,0)",
    "VAR(1) System",
    "Monthly Averaged OLS",
    "Univariate MIDAS (Gas)",
    "Bivariate MIDAS (Gas + Rate)",
    "AR-MIDAS Univariate (Gas)",
    "AR-MIDAS Bivariate (Gas + Rate)",
    "Random Forest",
    "Gradient Boosting",
]

preds = {name: [] for name in model_names}
actual_vals = []
oos_dates = []

init_uni = [0.1, 0.05, -0.1, -0.05]
init_bi = [0.1, 0.05, 0.02, -0.1, -0.05, -0.1, -0.05]
init_ar_uni = [0.1, 0.2, 0.05, -0.1, -0.05]
init_ar_bi = [0.1, 0.2, 0.05, 0.02, -0.1, -0.05, -0.1, -0.05]

print(f"Generating forecasts for {total_obs - initial_train_size} out-of-sample months...")

for t in range(initial_train_size, total_obs):
    y_tr, y_lag1_tr = y[:t], y_lag1[:t]
    X_gas_w_tr, X_rate_w_tr = X_gas_w[:t], X_rate_w[:t]
    X_comb_m_tr, X_ml_tr = X_comb_m[:t], X_ml[:t]
    df_var_tr = df_var.iloc[:t]

    y_te, y_lag1_te = y[t], y_lag1[t]
    X_gas_w_te, X_rate_w_te = X_gas_w[t : t + 1], X_rate_w[t : t + 1]
    X_comb_m_te, X_ml_te = X_comb_m[t : t + 1], X_ml[t : t + 1]

    # ARIMA
    try:
        p_ar1 = ARIMA(y_tr, order=(1, 0, 0)).fit().forecast(1)[0]
    except Exception:
        p_ar1 = np.mean(y_tr)
    preds["ARIMA(1,0,0) Baseline"].append(p_ar1)

    # ARIMAX
    try:
        p_arimax = ARIMA(y_tr, exog=X_comb_m_tr, order=(1, 0, 0)).fit().forecast(1, exog=X_comb_m_te)[0]
    except Exception:
        p_arimax = p_ar1
    preds["ARIMAX(1,0,0)"].append(p_arimax)

    # VAR
    try:
        var_f = VAR(df_var_tr).fit(maxlags=1).forecast(df_var_tr.values[-1:], steps=1)
        p_var = var_f[0, 0]
    except Exception:
        p_var = p_ar1
    preds["VAR(1) System"].append(p_var)

    # OLS
    A_ols = np.vstack([np.ones(len(y_tr)), X_comb_m_tr[:, 0], X_comb_m_tr[:, 1]]).T
    c_ols, _, _, _ = np.linalg.lstsq(A_ols, y_tr, rcond=None)
    p_ols = c_ols[0] + c_ols[1] * X_comb_m_te[0, 0] + c_ols[2] * X_comb_m_te[0, 1]
    preds["Monthly Averaged OLS"].append(p_ols)

    # Univariate MIDAS
    ru = minimize(midas_loss_univariate, init_uni, args=(X_gas_w_tr, y_tr), method="Nelder-Mead")
    wu = almon_weights(ru.x[2], ru.x[3], max_lag=4)
    preds["Univariate MIDAS (Gas)"].append(ru.x[0] + ru.x[1] * np.dot(X_gas_w_te, wu)[0])

    # Bivariate MIDAS
    rb = minimize(midas_loss_bivariate, init_bi, args=(X_gas_w_tr, X_rate_w_tr, y_tr), method="Nelder-Mead")
    wb1, wb2 = almon_weights(rb.x[3], rb.x[4], 4), almon_weights(rb.x[5], rb.x[6], 4)
    preds["Bivariate MIDAS (Gas + Rate)"].append(
        rb.x[0] + rb.x[1] * np.dot(X_gas_w_te, wb1)[0] + rb.x[2] * np.dot(X_rate_w_te, wb2)[0]
    )

    # AR-MIDAS Univariate
    rar_u = minimize(ar_midas_loss_univariate, init_ar_uni, args=(y_lag1_tr, X_gas_w_tr, y_tr), method="Nelder-Mead")
    war_u = almon_weights(rar_u.x[3], rar_u.x[4], 4)
    preds["AR-MIDAS Univariate (Gas)"].append(rar_u.x[0] + rar_u.x[1] * y_lag1_te + rar_u.x[2] * np.dot(X_gas_w_te, war_u)[0])

    # AR-MIDAS Bivariate
    rar_b = minimize(ar_midas_loss_bivariate, init_ar_bi, args=(y_lag1_tr, X_gas_w_tr, X_rate_w_tr, y_tr), method="Nelder-Mead")
    war_b1, war_b2 = almon_weights(rar_b.x[4], rar_b.x[5], 4), almon_weights(rar_b.x[6], rar_b.x[7], 4)
    preds["AR-MIDAS Bivariate (Gas + Rate)"].append(
        rar_b.x[0] + rar_b.x[1] * y_lag1_te + rar_b.x[2] * np.dot(X_gas_w_te, war_b1)[0] + rar_b.x[3] * np.dot(X_rate_w_te, war_b2)[0]
    )

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    rf.fit(X_ml_tr, y_tr)
    preds["Random Forest"].append(rf.predict(X_ml_te)[0])

    # Gradient Boosting
    gbm = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42)
    gbm.fit(X_ml_tr, y_tr)
    preds["Gradient Boosting"].append(gbm.predict(X_ml_te)[0])

    actual_vals.append(y_te)
    oos_dates.append(dates[t])

actual_vals = np.array(actual_vals)
oos_dates = pd.to_datetime(oos_dates)

df_results = pd.DataFrame(preds, index=oos_dates)
df_results["Actual CPI Inflation"] = actual_vals

fig, axes = plt.subplots(5, 2, figsize=(15, 16), sharex=True, sharey=True)
axes = axes.flatten()

colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"
]

for idx, model_name in enumerate(model_names):
    ax = axes[idx]
    y_hat = df_results[model_name].values
    mae = mean_absolute_error(actual_vals, y_hat)
    rmse = np.sqrt(mean_squared_error(actual_vals, y_hat))

    ax.plot(oos_dates, actual_vals, color="black", label="Actual Inflation", linewidth=1.5, alpha=0.9)
    ax.plot(oos_dates, y_hat, color=colors[idx], linestyle="--", label=f"Forecast ({model_name})", linewidth=1.3)
    ax.fill_between(oos_dates, actual_vals, y_hat, color="crimson", alpha=0.15, label="Error")

    ax.set_title(f"{model_name} (MAE: {mae:.3f}%, RMSE: {rmse:.3f}%)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("MoM (%)", fontsize=8)
    ax.legend(loc="upper left", fontsize=7.5, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

plt.suptitle("Out-of-Sample Canadian CPI Inflation Forecasts vs Actual (2020-2026)", fontsize=12, fontweight="bold", y=0.995)
plt.tight_layout()

os.makedirs("assets", exist_ok=True)
grid_path = "assets/model_forecasts_grid.png"
plt.savefig(grid_path, dpi=300, bbox_inches="tight")
plt.close()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})

ax1.plot(oos_dates, actual_vals, color="black", linewidth=2.0, label="Actual CPI Inflation", zorder=5)
ax1.plot(oos_dates, df_results["ARIMA(1,0,0) Baseline"], color="#7f7f7f", linestyle=":", label="ARIMA(1,0,0)", linewidth=1.4)
ax1.plot(oos_dates, df_results["Monthly Averaged OLS"], color="#ff7f0e", linestyle="-.", label="Monthly Averaged OLS", linewidth=1.4)
ax1.plot(oos_dates, df_results["Univariate MIDAS (Gas)"], color="#9467bd", linestyle="--", label="Univariate MIDAS (Gas)", linewidth=1.5)
ax1.plot(oos_dates, df_results["AR-MIDAS Univariate (Gas)"], color="#1f77b4", linestyle="-", label="AR-MIDAS (Gas)", linewidth=1.6)
ax1.plot(oos_dates, df_results["Random Forest"], color="#2ca02c", linestyle="-", label="Random Forest", linewidth=1.4)

ax1.set_title("Forecast Trajectory Comparison Across Models", fontsize=11, fontweight="bold")
ax1.set_ylabel("Monthly Inflation (%)", fontsize=9)
ax1.legend(loc="upper left", fontsize=8, frameon=True, ncol=2)
ax1.grid(True, linestyle=":", alpha=0.6)

for model_name in ["ARIMA(1,0,0) Baseline", "Monthly Averaged OLS", "Univariate MIDAS (Gas)", "AR-MIDAS Univariate (Gas)", "Random Forest"]:
    err = np.abs(actual_vals - df_results[model_name].values)
    ax2.plot(oos_dates, err, label=f"{model_name}", linewidth=1.3, alpha=0.85)

ax2.axhline(0, color="gray", linewidth=0.8)
ax2.set_title("Absolute Forecast Error (|Actual - Forecast|)", fontsize=10, fontweight="bold")
ax2.set_ylabel("Absolute Error (%)", fontsize=9)
ax2.set_xlabel("Date", fontsize=9)
ax2.legend(loc="upper left", fontsize=8, frameon=True, ncol=2)
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
timeline_path = "assets/forecast_error_timeline.png"
plt.savefig(timeline_path, dpi=300, bbox_inches="tight")
plt.close()


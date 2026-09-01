import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models.utils.math_utils import almon_weights, midas_loss_bivariate
from models.utils.load_lags import load_weekly_lag_matrix


def run_energy_policy_midas():
    gas_path = (
        "cleaned_predictors/gas_price.csv"
        if os.path.exists("cleaned_predictors/gas_price.csv")
        else "predictors/cleaned_gas_price.csv"
    )
    rate_path = (
        "cleaned_predictors/cleaned_interest_rates.csv"
        if os.path.exists("cleaned_predictors/cleaned_interest_rates.csv")
        else "predictors/cleaned_interest_rates.csv"
    )
    cpi_path = (
        "target_variables/cleaned_canadian_cpi.csv"
        if os.path.exists("target_variables/cleaned_canadian_cpi.csv")
        else "cleaned_canadian_cpi.csv"
    )

    df_gas = load_weekly_lag_matrix(gas_path, "diff_log_gas")
    df_rate = load_weekly_lag_matrix(rate_path, "diff_target_rate")

    df_cpi = pd.read_csv(cpi_path)
    df_cpi.columns = df_cpi.columns.str.strip().str.lower()
    cpi_date_col = [c for c in df_cpi.columns if "date" in c][0]
    df_cpi[cpi_date_col] = pd.to_datetime(df_cpi[cpi_date_col])
    df_cpi = df_cpi.sort_values(cpi_date_col).set_index(cpi_date_col)

    df_merged = df_gas.join(df_rate, lsuffix="_gas", rsuffix="_rate", how="inner")
    df_merged = df_merged.join(df_cpi[["inflation"]], how="inner").dropna()

    print(f"Timeline: {df_merged.index.min().date()} to {df_merged.index.max().date()}")
    print(f"Observations: {len(df_merged)}\n")

    X1 = df_merged[
        ["Lag_0_gas", "Lag_1_gas", "Lag_2_gas", "Lag_3_gas"]
    ].values
    X2 = df_merged[
        ["Lag_0_rate", "Lag_1_rate", "Lag_2_rate", "Lag_3_rate"]
    ].values
    y = df_merged["inflation"].values

    initial_guess = [0.001, 0.05, 0.02, -0.1, -0.05, -0.1, -0.05]

    res_full = minimize(
        midas_loss_bivariate,
        initial_guess,
        args=(X1, X2, y),
        method="Nelder-Mead",
    )
    alpha, beta1, beta2, t1_1, t2_1, t1_2, t2_2 = res_full.x

    w1_opt = almon_weights(t1_1, t2_1, max_lag=4)
    w2_opt = almon_weights(t1_2, t2_2, max_lag=4)

    y_pred_is = (
        alpha + (beta1 * np.dot(X1, w1_opt)) + (beta2 * np.dot(X2, w2_opt))
    )
    is_mae = mean_absolute_error(y, y_pred_is)
    is_mse = mean_squared_error(y, y_pred_is)

    initial_train_size = int(len(y) * 0.6)
    oos_preds, oos_actuals = [], []

    for t in range(initial_train_size, len(y)):
        X1_train, X2_train, y_train = X1[:t], X2[:t], y[:t]
        X1_test, X2_test, y_test = X1[t : t + 1], X2[t : t + 1], y[t : t + 1]

        res_step = minimize(
            midas_loss_bivariate,
            initial_guess,
            args=(X1_train, X2_train, y_train),
            method="Nelder-Mead",
        )
        a_s, b1_s, b2_s, t1_1_s, t2_1_s, t1_2_s, t2_2_s = res_step.x

        w1_s = almon_weights(t1_1_s, t2_1_s, max_lag=4)
        w2_s = almon_weights(t1_2_s, t2_2_s, max_lag=4)

        pred_step = (
            a_s
            + (b1_s * np.dot(X1_test, w1_s))
            + (b2_s * np.dot(X2_test, w2_s))
        )
        oos_preds.append(pred_step[0])
        oos_actuals.append(y_test[0])

    oos_mae = mean_absolute_error(oos_actuals, oos_preds)
    oos_mse = mean_squared_error(oos_actuals, oos_preds)

    print("Bivariate MIDAS (Gas Price + Policy Rate):")
    print(f"Alpha: {alpha:.6f}")
    print(f"Beta 1 (Gas): {beta1:.6f}")
    print(f"Beta 2 (Rate): {beta2:.6f}\n")

    print(f"{'Lag':<6} {'Gas Weight':<15} {'Rate Weight':<15}")
    print("-" * 38)
    for i in range(4):
        print(f"t-{i:<4} {w1_opt[i]:<15.4f} {w2_opt[i]:<15.4f}")

    print("\nPerformance Summary:")
    print(f"{'Metric':<20} {'In-Sample':<15} {'Out-of-Sample (OOS)':<15}")
    print("-" * 50)
    print(f"{'MAE':<20} {is_mae:<15.6f} {oos_mae:<15.6f}")
    print(f"{'MSE':<20} {is_mse:<15.6f} {oos_mse:<15.6f}\n")


if __name__ == "__main__":
    run_energy_policy_midas()

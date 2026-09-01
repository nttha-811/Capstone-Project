import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models.utils.math_utils import almon_weights, midas_loss_univariate
from models.utils.load_lags import load_weekly_lag_matrix


def run_gas_midas():
    gas_path = "cleaned_predictors/gas_price.csv"
    cpi_path = "target_variables/cleaned_canadian_cpi.csv"

    df_X = load_weekly_lag_matrix(gas_path, "diff_log_gas")

    df_target = pd.read_csv(cpi_path)
    df_target.columns = df_target.columns.str.strip().str.lower()
    target_date_col = [c for c in df_target.columns if "date" in c][0]
    df_target[target_date_col] = pd.to_datetime(df_target[target_date_col])
    df_target = df_target.sort_values(target_date_col).set_index(target_date_col)

    df_merged = df_X.join(df_target[["inflation"]], how="inner").dropna()
    df_merged = df_merged.loc["2016-08-01":].copy()

    print(f"Timeline: {df_merged.index.min().date()} to {df_merged.index.max().date()}")
    print(f"Observations: {len(df_merged)}\n")

    X_final = df_merged[["Lag_0", "Lag_1", "Lag_2", "Lag_3"]].values
    y_final = df_merged["inflation"].values

    initial_guess = [0.001, 0.05, -0.1, -0.05]
    res_full = minimize(
        midas_loss_univariate,
        initial_guess,
        args=(X_final, y_final),
        method="Nelder-Mead",
    )

    alpha_opt, beta_opt, theta1_opt, theta2_opt = res_full.x
    final_weights = almon_weights(theta1_opt, theta2_opt, max_lag=4)

    X_aggregated_is = np.dot(X_final, final_weights)
    y_pred_is = alpha_opt + beta_opt * X_aggregated_is

    is_mae = mean_absolute_error(y_final, y_pred_is)
    is_mse = mean_squared_error(y_final, y_pred_is)

    initial_train_size = int(len(y_final) * 0.6)
    total_observations = len(y_final)

    out_of_sample_predictions = []
    actual_inflation_values = []

    for t in range(initial_train_size, total_observations):
        X_train, y_train = X_final[:t], y_final[:t]
        X_test, y_test = X_final[t : t + 1], y_final[t : t + 1]

        res_step = minimize(
            midas_loss_univariate,
            initial_guess,
            args=(X_train, y_train),
            method="Nelder-Mead",
        )
        alpha_step, beta_step, theta1_step, theta2_step = res_step.x
        step_weights = almon_weights(theta1_step, theta2_step, max_lag=4)

        X_test_aggregated = np.dot(X_test, step_weights)
        y_pred_step = alpha_step + beta_step * X_test_aggregated

        out_of_sample_predictions.append(y_pred_step[0])
        actual_inflation_values.append(y_test[0])

    y_pred_oos = np.array(out_of_sample_predictions)
    y_true_oos = np.array(actual_inflation_values)

    oos_mae = mean_absolute_error(y_true_oos, y_pred_oos)
    oos_mse = mean_squared_error(y_true_oos, y_pred_oos)

    print("Univariate MIDAS (Gasoline) Results:")
    print(f"Alpha: {alpha_opt:.6f}")
    print(f"Beta:  {beta_opt:.6f}")
    for idx, w in enumerate(final_weights):
        print(f"  Lag {idx} weight: {w:.4f}")

    print("\nPerformance Summary:")
    print(f"{'Metric':<20} {'In-Sample':<15} {'Out-of-Sample (OOS)':<15}")
    print("-" * 50)
    print(f"{'MAE':<20} {is_mae:<15.6f} {oos_mae:<15.6f}")
    print(f"{'MSE':<20} {is_mse:<15.6f} {oos_mse:<15.6f}\n")


if __name__ == "__main__":
    run_gas_midas()
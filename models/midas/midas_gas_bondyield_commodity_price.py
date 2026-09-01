import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models.utils.load_lags import load_weekly_lag_matrix
from models.utils.math_utils import almon_weights, midas_loss_trivariate


def run_trivariate_midas():
    df_gas = load_weekly_lag_matrix(
        file_path="cleaned_predictors/gas_price.csv",
        target_col="diff_log_gas",
    )
    df_bcpi = load_weekly_lag_matrix(
        file_path="cleaned_predictors/commodity_prices_W.BCPI.csv",
        target_col="diff_log_bcpi",
    )
    df_yield = load_weekly_lag_matrix(
        file_path="cleaned_predictors/bond_yields.csv",
        target_col="diff_yield_2yr",
    )

    cpi_path = "target_variables/cleaned_canadian_cpi.csv"
    if not os.path.exists(cpi_path):
        cpi_path = "cleaned_predictors/cleaned_canadian_cpi.csv"

    df_target = pd.read_csv(cpi_path)
    df_target.columns = df_target.columns.str.strip().str.lower()
    target_date_col = "date" if "date" in df_target.columns else df_target.columns[0]
    df_target[target_date_col] = pd.to_datetime(df_target[target_date_col])
    df_target = df_target.sort_values(target_date_col).set_index(target_date_col)

    df_merged = df_target[["inflation"]].join(
        df_gas.add_prefix("gas_"), how="inner"
    ).join(
        df_bcpi.add_prefix("bcpi_"), how="inner"
    ).join(
        df_yield.add_prefix("yield_"), how="inner"
    ).dropna()

    print(f"Timeline: {df_merged.index.min().date()} to {df_merged.index.max().date()}")
    print(f"Observations: {len(df_merged)}\n")

    X_gas = df_merged[["gas_Lag_0", "gas_Lag_1", "gas_Lag_2", "gas_Lag_3"]].values
    X_bcpi = df_merged[["bcpi_Lag_0", "bcpi_Lag_1", "bcpi_Lag_2", "bcpi_Lag_3"]].values
    X_yield = df_merged[["yield_Lag_0", "yield_Lag_1", "yield_Lag_2", "yield_Lag_3"]].values
    y_final = df_merged["inflation"].values

    initial_guess = [0.01, 0.3, 0.2, 0.1, -0.1, -0.02, -0.1, -0.02, -0.1, -0.02]

    res_full = minimize(
        midas_loss_trivariate,
        initial_guess,
        args=(X_gas, X_bcpi, X_yield, y_final),
        method="Nelder-Mead",
        options={"maxiter": 5000},
    )

    params_opt = res_full.x
    alpha_opt = params_opt[0]
    beta_g, beta_b, beta_y = params_opt[1], params_opt[2], params_opt[3]

    w_gas_opt = almon_weights(params_opt[4], params_opt[5])
    w_bcpi_opt = almon_weights(params_opt[6], params_opt[7])
    w_yield_opt = almon_weights(params_opt[8], params_opt[9])

    y_pred_is = (
        alpha_opt
        + (beta_g * np.dot(X_gas, w_gas_opt))
        + (beta_b * np.dot(X_bcpi, w_bcpi_opt))
        + (beta_y * np.dot(X_yield, w_yield_opt))
    )

    is_mae = mean_absolute_error(y_final, y_pred_is)
    is_mse = mean_squared_error(y_final, y_pred_is)

    initial_train_size = min(50, int(len(y_final) * 0.6))
    total_obs = len(y_final)

    oos_preds = []
    actuals = []

    for t in range(initial_train_size, total_obs):
        X1_tr, X2_tr, X3_tr, y_tr = X_gas[:t], X_bcpi[:t], X_yield[:t], y_final[:t]
        X1_te, X2_te, X3_te, y_te = X_gas[t:t+1], X_bcpi[t:t+1], X_yield[t:t+1], y_final[t:t+1]

        res_step = minimize(
            midas_loss_trivariate,
            initial_guess,
            args=(X1_tr, X2_tr, X3_tr, y_tr),
            method="Nelder-Mead",
            options={"maxiter": 3000},
        )

        p = res_step.x
        a_step = p[0]
        b1_step, b2_step, b3_step = p[1], p[2], p[3]

        wg_s = almon_weights(p[4], p[5])
        wb_s = almon_weights(p[6], p[7])
        wy_s = almon_weights(p[8], p[9])

        y_hat = (
            a_step
            + (b1_step * np.dot(X1_te, wg_s))
            + (b2_step * np.dot(X2_te, wb_s))
            + (b3_step * np.dot(X3_te, wy_s))
        )

        oos_preds.append(y_hat[0])
        actuals.append(y_te[0])

    oos_mae = mean_absolute_error(actuals, oos_preds)
    oos_mse = mean_squared_error(actuals, oos_preds)

    print("Trivariate MIDAS (Gas + BCPI + 2Y Yield):")
    print(f"Alpha:               {alpha_opt:.6f}")
    print(f"Beta 1 (Gas):        {beta_g:.6f}")
    print(f"Beta 2 (BCPI):       {beta_b:.6f}")
    print(f"Beta 3 (2Y Yield):   {beta_y:.6f}\n")

    print(f"{'Lag':<8} {'Gas Weight':<14} {'BCPI Weight':<14} {'Yield Weight':<14}")
    print("-" * 52)
    for i in range(4):
        print(f"t-{i:<6} {w_gas_opt[i]:<14.4f} {w_bcpi_opt[i]:<14.4f} {w_yield_opt[i]:<14.4f}")

    print("\nPerformance Summary:")
    print(f"{'Metric':<20} {'In-Sample':<15} {'Out-of-Sample (OOS)':<15}")
    print("-" * 50)
    print(f"{'MAE':<20} {is_mae:<15.6f} {oos_mae:<15.6f}")
    print(f"{'MSE':<20} {is_mse:<15.6f} {oos_mse:<15.6f}\n")


if __name__ == "__main__":
    run_trivariate_midas()

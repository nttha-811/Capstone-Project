import os
import numpy as np
import pandas as pd


def clean_exchange_rates(
    input_path="predictors/DailyExchangeRates_2017.csv",
    output_path="cleaned_predictors/exchange_rates.csv",
):
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    target_col = "FXUSDCAD"
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").ffill()

    df_weekly = df[[target_col]].resample("W-FRI").last()
    df_weekly["log_fx"] = np.log(df_weekly[target_col])
    df_weekly["diff_log_fx"] = df_weekly["log_fx"].diff()
    df_weekly = df_weekly.dropna()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_weekly.to_csv(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    clean_exchange_rates()
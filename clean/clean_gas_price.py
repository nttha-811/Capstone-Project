import os
import numpy as np
import pandas as pd


def clean_gas_prices(
    input_path="predictors/gas_price_us_1990.csv",
    output_path="cleaned_predictors/gas_price.csv",
):
    df_raw = pd.read_csv(input_path)
    df_raw["observation_date"] = pd.to_datetime(df_raw["observation_date"])
    df_raw = df_raw.rename(columns={"observation_date": "date"})
    df_raw = df_raw.sort_values("date").set_index("date")

    df_raw["log_gas"] = np.log(df_raw["GASREGW"])
    df_raw["diff_log_gas"] = df_raw["log_gas"].diff()
    df_cleaned = df_raw[["GASREGW", "diff_log_gas"]].dropna()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cleaned.to_csv(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    clean_gas_prices()


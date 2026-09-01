import os
import numpy as np
import pandas as pd


def clean_commodity_prices_split(
    input_path="predictors/commodityPriceIndex_1972.csv",
    output_path="cleaned_predictors/commodity_prices_split.csv",
):
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    energy_col = "W.ENER"
    non_energy_col = "W.BCNE"

    df[energy_col] = pd.to_numeric(df[energy_col], errors="coerce").ffill()
    df[non_energy_col] = pd.to_numeric(df[non_energy_col], errors="coerce").ffill()

    df["log_energy"] = np.log(df[energy_col])
    df["diff_log_energy"] = df["log_energy"].diff()

    df["log_non_energy"] = np.log(df[non_energy_col])
    df["diff_log_non_energy"] = df["log_non_energy"].diff()

    df_cleaned = df[[energy_col, "diff_log_energy", non_energy_col, "diff_log_non_energy"]].dropna()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cleaned.to_csv(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    clean_commodity_prices_split()
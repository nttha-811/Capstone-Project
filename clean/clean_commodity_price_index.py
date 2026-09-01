import os
import numpy as np
import pandas as pd


def clean_commodity_prices(
    input_path="predictors/commodityPriceIndex_1972.csv",
    output_path="cleaned_predictors/commodity_prices_W.BCPI.csv",
):
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    target_col = "W.BCPI"
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").ffill()

    df["log_bcpi"] = np.log(df[target_col])
    df["diff_log_bcpi"] = df["log_bcpi"].diff()
    df_cleaned = df[[target_col, "diff_log_bcpi"]].dropna()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cleaned.to_csv(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    clean_commodity_prices()
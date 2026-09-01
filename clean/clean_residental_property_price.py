import os
import numpy as np
import pandas as pd


def clean_property_prices(
    input_path="predictors/Residential Property Prices for Canada_monthly_1970.csv",
    output_path="cleaned_predictors/property_price.csv",
):
    df = pd.read_csv(input_path)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.rename(columns={"observation_date": "date"})
    df = df.sort_values("date").set_index("date")

    target_col = "QCAR628BIS"
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").ffill()

    df["log_property"] = np.log(df[target_col])
    df["property_inflation"] = df["log_property"].diff()
    df_cleaned = df[[target_col, "property_inflation"]].dropna()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cleaned.to_csv(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    clean_property_prices()
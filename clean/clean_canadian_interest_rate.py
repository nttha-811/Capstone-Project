import os
import numpy as np
import pandas as pd


def clean_interest_rates(
    input_path="predictors/CanadianInterestRates_2016.csv",
    output_path="cleaned_predictors/cleaned_interest_rates.csv",
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find raw interest rates at: {input_path}")

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip().str.lower()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").set_index("date")

    target_rate_col = "v39079"
    corra_col = "canadian overnight repo rate average (corra) (%)"

    df_selected = df[[target_rate_col, corra_col]].copy()

    for col in [target_rate_col, corra_col]:
        df_selected[col] = df_selected[col].replace(r"^\s*$", np.nan, regex=True)
        df_selected[col] = df_selected[col].replace("Not available", np.nan)
        df_selected[col] = pd.to_numeric(df_selected[col], errors="coerce").ffill()

    df_weekly = df_selected.resample("W-FRI").last()
    df_weekly["diff_target_rate"] = df_weekly[target_rate_col].diff()
    df_weekly["diff_corra"] = df_weekly[corra_col].diff()
    df_weekly = df_weekly.dropna()

    df_weekly = df_weekly.rename(
        columns={
            target_rate_col: "raw_target_rate",
            corra_col: "raw_corra",
        }
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_weekly.to_csv(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    clean_interest_rates()
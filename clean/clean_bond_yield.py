import os
import numpy as np
import pandas as pd


def clean_bond_yields(
    input_path="predictors/bond_yields_all_2001.csv",
    output_path="cleaned_predictors/bond_yields.csv",
):
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = df.sort_values("date").set_index("date")

    target_col = "BD.CDN.2YR.DQ.YLD"
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").ffill()

    df_weekly = df[[target_col]].resample("W-FRI").last()
    df_weekly["diff_yield_2yr"] = df_weekly[target_col].diff()
    df_weekly = df_weekly.dropna()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_weekly.to_csv(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    clean_bond_yields()
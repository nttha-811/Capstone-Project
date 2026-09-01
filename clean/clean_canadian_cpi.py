import os
import numpy as np
import pandas as pd


def clean_canadian_cpi(filepath="target_variables/canadian_cpi.csv", output_path="target_variables/cleaned_canadian_cpi.csv"):
    df_raw = pd.read_csv(filepath, skiprows=8)
    df_raw = df_raw.dropna(subset=[df_raw.columns[0]])
    df_raw = df_raw[~df_raw[df_raw.columns[0]].astype(str).str.startswith("Footnotes")]

    date_columns = df_raw.iloc[0].values[1:]
    date_columns = [d for d in date_columns if pd.notna(d)]

    df_raw.iloc[:, 0] = df_raw.iloc[:, 0].astype(str).str.strip().str.replace('"', "")
    cpi_row_mask = df_raw.iloc[:, 0].str.startswith("All-items 8")

    if not cpi_row_mask.any():
        raise ValueError("Could not find 'All-items 8' row in CPI data.")

    cpi_values = df_raw[cpi_row_mask].iloc[0, 1 : len(date_columns) + 1].values
    cpi_values = pd.to_numeric([str(v).replace('"', "").strip() for v in cpi_values], errors="coerce")

    df_cpi = pd.DataFrame({"CPI": cpi_values}, index=pd.to_datetime(date_columns))
    df_cpi.index.name = "Date"
    df_cpi = df_cpi.sort_index().dropna()

    df_cpi["log_cpi"] = np.log(df_cpi["CPI"])
    df_cpi["inflation"] = df_cpi["log_cpi"].diff()
    df_cpi = df_cpi.dropna()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cpi.to_csv(output_path)
    print(f"Saved: {output_path}")
    return df_cpi[["CPI", "inflation"]]


if __name__ == "__main__":
    df_target = clean_canadian_cpi()
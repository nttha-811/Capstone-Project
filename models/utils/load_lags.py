import os
import pandas as pd


def load_weekly_lag_matrix(file_path, target_col):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not locate file: {file_path}")

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()

    date_col = [c for c in df.columns if "date" in c][0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    target_col = target_col.lower()
    if target_col not in df.columns:
        raise KeyError(
            f"Column '{target_col}' not found. Available: {list(df.columns)}"
        )

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").ffill()

    grouped = df.groupby([df.index.year, df.index.month])

    aligned_rows = []
    aligned_months = []

    for (year, month), group in grouped:
        if len(group) >= 4:
            recent_4_weeks = group[target_col].values[-4:]
            aligned_rows.append(recent_4_weeks[::-1])
            aligned_months.append(pd.Timestamp(year=year, month=month, day=1))

    df_lag = pd.DataFrame(
        aligned_rows,
        columns=["Lag_0", "Lag_1", "Lag_2", "Lag_3"],
        index=aligned_months,
    )
    df_lag.index.name = "Date"
    return df_lag


def load_monthly_lag_matrix(file_path, value_col, date_col="date"):
    return load_weekly_lag_matrix(file_path, value_col)
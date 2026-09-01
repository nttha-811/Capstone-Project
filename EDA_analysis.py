import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

cpi_path = "target_variables/cleaned_canadian_cpi.csv"
if not os.path.exists(cpi_path):
    cpi_path = "cleaned_predictors/cleaned_canadian_cpi.csv"

df_target = pd.read_csv(cpi_path)
df_target.columns = df_target.columns.str.strip().str.lower()

date_col = "date" if "date" in df_target.columns else df_target.columns[0]
df_target[date_col] = pd.to_datetime(df_target[date_col])
df_target = df_target.set_index(date_col)[["inflation"]]
df_target = df_target.rename(columns={"inflation": "CPI Inflation MoM"})

predictors_map = {
    "gas_price.csv": (
        "diff_log_gas",
        "Gas Price Return",
        "Weekly",
    ),
    "cleaned_interest_rates.csv": (
        "diff_target_rate",
        "Policy Rate Change",
        "Daily->Weekly",
    ),
    "bond_yields.csv": (
        "diff_yield_2yr",
        "2Y Bond Yield Change",
        "Daily->Weekly",
    ),
    "commodity_prices_W.BCPI.csv": (
        "diff_log_bcpi",
        "Total Commodity Index (BCPI)",
        "Weekly",
    ),
    "commodity_prices_split.csv": (
        "diff_log_energy",
        "Energy Commodity Return",
        "Weekly",
    ),
    "commodity_prices_split_non_en": (
        "diff_log_non_energy",
        "Non-Energy Commodity Return",
        "Weekly",
    ),
    "exchange_rates.csv": (
        "diff_log_fx",
        "USD/CAD Return",
        "Daily->Weekly",
    ),
    "property_price.csv": (
        "property_inflation",
        "Property Price Inflation",
        "Quarterly",
    ),
}

aligned_series = [df_target]

for key, (col_name, clean_label, freq) in predictors_map.items():
    actual_filename = "commodity_prices_split.csv" if "commodity_prices_split" in key else key
    file_path = f"cleaned_predictors/{actual_filename}"

    if os.path.exists(file_path):
        df_pred = pd.read_csv(file_path)
        df_pred.columns = df_pred.columns.str.strip().str.lower()

        pred_date_col = "date" if "date" in df_pred.columns else df_pred.columns[0]
        df_pred[pred_date_col] = pd.to_datetime(df_pred[pred_date_col])
        df_pred = df_pred.set_index(pred_date_col)

        if col_name not in df_pred.columns:
            continue

        df_selected = df_pred[[col_name]]

        if "property" in key:
            df_monthly = df_selected.resample("MS").ffill().rename(columns={col_name: clean_label})
        else:
            df_monthly = df_selected.resample("MS").mean().rename(columns={col_name: clean_label})

        aligned_series.append(df_monthly)

df_master = aligned_series[0]
for series in aligned_series[1:]:
    df_master = df_master.join(series, how="inner")

target_col_name = "CPI Inflation MoM"

print(f"Timeline: {df_master.index.min().date()} to {df_master.index.max().date()}")
print(f"Observations: {len(df_master)}\n")

stats_df = df_master.describe().T[["count", "mean", "std", "min", "50%", "max"]]
stats_df = stats_df.rename(columns={"50%": "median"})
print("Summary Statistics:")
print(stats_df.to_string())
print("\n")

corr_matrix = df_master.corr()
target_corr = corr_matrix[[target_col_name]].drop(index=target_col_name)
target_corr = target_corr.rename(columns={target_col_name: "Correlation (r)"})
target_corr["Abs Correlation"] = target_corr["Correlation (r)"].abs()
target_corr = target_corr.sort_values(by="Abs Correlation", ascending=False)

print("Correlations with Target Inflation:")
print(target_corr[["Correlation (r)"]].to_string())

os.makedirs("clean", exist_ok=True)
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    linewidths=0.8,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 9},
)
plt.title("Correlation Matrix of Predictors and Target CPI", fontsize=11, fontweight="bold", pad=12)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()

heatmap_path = "clean/master_system_heatmap.png"
plt.savefig(heatmap_path, dpi=300)
plt.close()

fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
axes = axes.flatten()

feature_cols = [c for c in df_master.columns if c != target_col_name]
for idx, col in enumerate(feature_cols[:8]):
    ax = axes[idx]
    ax.plot(df_master.index, df_master[col], color="navy", label=col, linewidth=1.2)
    ax.set_title(col, fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
ts_plot_path = "clean/macro_indicators_timeseries.png"
plt.savefig(ts_plot_path, dpi=300)
plt.close()


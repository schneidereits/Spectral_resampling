# visualize_resampled_enmap_library.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------
# Paths to resampled CSV outputs
# ------------------------
RESAMPLED_RAW_PATH = "spectral_library_resampled_enmap.csv"
RESAMPLED_INTERP_PATH = "spectral_library_resampled_enmap_interpolated.csv"
ORIGINAL_LIBRARY_PATH = "99_library_joined_filtered.csv"

# ------------------------
# Load DataFrames
# ------------------------
resampled_raw_df = pd.read_csv(RESAMPLED_RAW_PATH)
resampled_interp_df = pd.read_csv(RESAMPLED_INTERP_PATH)
spectral_lib = pd.read_csv(ORIGINAL_LIBRARY_PATH)

# ------------------------
# Helper: Format wavelength columns to exactly 2 decimals
# ------------------------
def format_wavelength_cols(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for c in df.columns:
        try:
            new_cols.append(f"{float(c):.2f}")
        except ValueError:
            new_cols.append(c)
    df.columns = new_cols
    return df

resampled_raw_df = format_wavelength_cols(resampled_raw_df)
resampled_interp_df = format_wavelength_cols(resampled_interp_df)

# ------------------------
# Wavelength columns (numeric columns)
# ------------------------
wavelength_cols = [c for c in resampled_interp_df.columns if str(c).replace('.', '').isdigit()]

print(f"Rows with NaN values: {np.isnan(resampled_raw_df[wavelength_cols]).any(axis=1).sum()}")

# ------------------------
# 1 Plot NaN counts per wavelength
# ------------------------

def plot_nan_counts(df, wavelength_cols):
    # Ensure column names are cleaned and converted to float
    clean_cols = []
    for c in wavelength_cols:
        try:
            clean_cols.append(float(str(c).strip()))
        except ValueError:
            continue

    # Count NaNs per column
    na_counts = df[wavelength_cols].isna().sum()
    na_counts.index = clean_cols

    # Sort by wavelength
    na_counts = na_counts.sort_index()

    # Plot
    plt.figure(figsize=(12, 5))
    plt.bar(na_counts.index, na_counts.values, width=1.0, edgecolor='black')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Number of NaNs")
    plt.title("Number of NaNs per Wavelength (sorted by wavelength)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Use it
plot_nan_counts(resampled_raw_df, wavelength_cols)
plot_nan_counts(resampled_interp_df, wavelength_cols)

# ------------------------
# 2 Mean Percent Error between raw and interpolated
# ------------------------
def plot_mean_percent_error(resampled_raw, resampled_interp, wavelength_cols):
    mpe_per_band = []
    for col in wavelength_cols:
        arr1 = resampled_raw[col].astype(float).values
        arr2 = resampled_interp[col].astype(float).values
        valid_mask = (arr2 != 0) & np.isfinite(arr1) & np.isfinite(arr2)
        mpe = np.nanmean(((arr1[valid_mask] - arr2[valid_mask]) / arr2[valid_mask]) * 100)
        mpe_per_band.append(mpe)

    overall_mpe = np.nanmean(mpe_per_band)

    plt.figure(figsize=(12, 5))
    plt.plot(sorted([float(w) for w in wavelength_cols]), mpe_per_band,
             marker='o', linestyle='-', alpha=0.7, label="Mean % Error")
    plt.axhline(overall_mpe, color='orange', linestyle='--', label=f"Overall MPE = {overall_mpe:.2f}%")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Mean Percent Error (%)")
    plt.ticklabel_format(style='plain', axis='y')
    plt.title("Mean Percent Error Across Bands")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_mean_percent_error(resampled_raw_df, resampled_interp_df, wavelength_cols)

# ------------------------
# 3 Reshape to long format for FacetGrid
# ------------------------
meta_cols = ["class", "category_1", "category_2", "id", "source"]
df_long = resampled_interp_df.melt(
    id_vars=meta_cols,
    value_vars=wavelength_cols,
    var_name="wavelength",
    value_name="reflectance"
)
df_long["wavelength"] = pd.to_numeric(df_long["wavelength"], errors="coerce")

# ------------------------
# 4 Spectral library plot (FacetGrid)
# ------------------------
def plot_spectral_library(df_long, hue_col='source', meta_cols=None, class_col='class'):
    g = sns.FacetGrid(df_long, col=class_col, col_wrap=3, sharey=False, sharex=True)
    g.map_dataframe(
        sns.lineplot,
        x="wavelength",
        y="reflectance",
        hue=hue_col,
        units="id",
        estimator=None,
        alpha=0.2
    )
    g.set_axis_labels("Wavelength (nm)", "Reflectance")
    g.set_titles("{col_name}")
    g.add_legend(title=hue_col, bbox_to_anchor=(0.3, -0.05), loc="upper center", ncol=2)
    plt.subplots_adjust(top=0.9, bottom=0.15)
    g.fig.suptitle("Spectral Library")
    plt.show()

plot_spectral_library(df_long, hue_col="source", meta_cols=meta_cols, class_col="class")

# ------------------------
# 5 NRMSE between raw and interpolated
# ------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_nrmse_vs_original(resampled_df: pd.DataFrame,
                           original_df: pd.DataFrame,
                           wavelength_cols: list,
                           id_col: str = "id",
                           title: str = "Normalized RMSE Across Bands"):
    """
    Calculate and plot NRMSE between a resampled spectral library and the original library.
    Each resampled wavelength is compared to the closest wavelength in the original library.

    Parameters:
        resampled_df (pd.DataFrame): Resampled spectral library with 'id' and spectral columns.
        original_df (pd.DataFrame): Original spectral library with 'id' and spectral columns.
        wavelength_cols (list): List of wavelength columns in the resampled_df.
        id_col (str): Column name for unique spectrum IDs. Default is "id".
        title (str): Title for the plot.
    """

    # --- Identify numeric wavelength columns in the original spectral library ---
    original_wls = [c for c in original_df.columns if str(c).replace(".", "").replace("-", "").isdigit()]

    # --- Convert to float for matching ---
    resampled_wls = np.array([float(w) for w in wavelength_cols])
    original_wls_float = np.array([float(w) for w in original_wls])

    # --- Find closest spectral library column for each resampled wavelength ---
    closest_cols = [original_wls[np.argmin(np.abs(original_wls_float - w))] for w in resampled_wls]

    # --- Find matching IDs ---
    matching_ids = list(set(resampled_df[id_col]).intersection(set(original_df[id_col])))

    # --- Filter and set index ---
    resampled_matched = resampled_df[resampled_df[id_col].isin(matching_ids)].set_index(id_col)
    original_matched = original_df[original_df[id_col].isin(matching_ids)].set_index(id_col)

    # --- Compute NRMSE per band ---
    nrmse_per_band = []
    for res_col, orig_col in zip(wavelength_cols, closest_cols):
        arr1 = resampled_matched[res_col].astype(float).values
        arr2 = original_matched[orig_col].astype(float).values
        rmse = np.sqrt(np.nanmean((arr1 - arr2) ** 2))
        nrmse = rmse / (np.nanmax(arr2) - np.nanmin(arr2))
        nrmse_per_band.append(nrmse)

    overall_nrmse = np.nanmean(nrmse_per_band)

    # --- Plot ---
    plt.figure(figsize=(12, 5))
    plt.plot([float(w) for w in closest_cols], nrmse_per_band,
             marker='o', linestyle='-', alpha=0.7, label="NRMSE")
    plt.axhline(overall_nrmse, color='orange', linestyle='--',
                label=f"Overall NRMSE = {overall_nrmse:.5f}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized RMSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_nrmse_vs_original(resampled_raw_df, spectral_lib, wavelength_cols, "id", "Normalized RMSE Across Bands: RAW vs Original")
plot_nrmse_vs_original(resampled_interp_df, spectral_lib, wavelength_cols, "id", "Normalized RMSE Across Bands: interpolated vs Original")

# ------------------------
# 6 Spectra comparison across libraries
# ------------------------
def plot_spectra_comparison(resampled_df, ref_df, original_df, common_wavelengths, classes):
    for cls in classes:
        cls_ids = resampled_df[resampled_df['class'] == cls]['id']
        cls_matching_ids = cls_ids[:5]  # Pick first 5
        if not len(cls_matching_ids):
            continue

        plt.figure(figsize=(12, 6))
        all_vals = []
        for sid in cls_matching_ids:
            spec1 = resampled_df.set_index('id').loc[sid, common_wavelengths].astype(float).values
            plt.plot([float(w) for w in common_wavelengths], spec1, alpha=0.9, linestyle=':', color="blue", linewidth=2.5)

            spec2 = ref_df.set_index('id').loc[sid, common_wavelengths].astype(float).values
            plt.plot([float(w) for w in common_wavelengths], spec2, alpha=0.5, linestyle='--', color='red', linewidth=2.5)

            spec3 = original_df.set_index('id').loc[sid, [str(int(float(w))) for w in common_wavelengths]].astype(float).values
            plt.plot([float(w) for w in common_wavelengths], spec3, alpha=0.5, linestyle='-', color='green', linewidth=1.5)

            all_vals.extend(spec1)
            all_vals.extend(spec2)
            all_vals.extend(spec3)

        plt.legend(["Resampled (enmapbox)", "Reference Sigma Library", "Original ASD Library"])
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.title(f"Spectra Comparison ({cls})")
        plt.ylim(np.nanmin(all_vals), np.nanmax(all_vals))
        plt.tight_layout()
        plt.show()

classes = resampled_raw_df['class'].unique()
common_wavelengths = wavelength_cols
plot_spectra_comparison(resampled_raw_df, resampled_interp_df, spectral_lib, common_wavelengths, classes)

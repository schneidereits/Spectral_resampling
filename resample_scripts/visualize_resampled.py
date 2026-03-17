"""
Parameterized Spectral Library Visualization Script
---------------------------------------------------
Visualizes resampled spectral libraries for any sensor.
Plots NaN distributions, mean percent error, spectral library patterns, 
and computes NRMSE analysis.

Usage:
    1. Edit the "User Configuration" section below
    2. Run: python visualize_resampled_parameterized.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (where wavelength/ folder is)
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


# ============================================================================
# USER CONFIGURATION
# ============================================================================
# Edit these parameters before running the script

SENSOR = "landsat-9"  # Options: AVIRIS-3, AVIRIS-NG, enmap, landsat-8, landsat-9
ORIGINAL_LIB_PATH = r"E:\Project_EnFireMap\01_data\03_spectral_libraries\99_library_joined_with_lake.csv"  # Original spectral library
RESAMPLED_DIR = r"C:\Users\schnesha\Downloads\resample_test"  # Directory where resampled files are located

# ============================================================================
# Sensor Configuration Registry
# ============================================================================
SENSOR_CONFIG = {
    "AVIRIS-3": {
        "output_prefix": "spectral_library_resampled_AVIRIS-3",
        "description": "AVIRIS-3"
    },
    "AVIRIS-NG": {
        "output_prefix": "spectral_library_resampled_AVIRIS-NG",
        "description": "AVIRIS-NG"
    },
    "enmap": {
        "output_prefix": "spectral_library_resampled_enmap",
        "description": "EnMAP"
    },
    "landsat-8": {
        "output_prefix": "spectral_library_resampled_landsat8",
        "description": "Landsat-8 OLI"
    },
    "landsat-9": {
        "output_prefix": "spectral_library_resampled_landsat9",
        "description": "Landsat-9 OLI-2"
    },
}


# ============================================================================
# Utility Functions
# ============================================================================
def load_sensor_config(sensor_name):
    """Load sensor configuration from registry."""
    if sensor_name not in SENSOR_CONFIG:
        available = ", ".join(SENSOR_CONFIG.keys())
        raise ValueError(f"Unknown sensor '{sensor_name}'. Available sensors: {available}")
    return SENSOR_CONFIG[sensor_name]


def format_wavelength_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Format wavelength columns to exactly 2 decimals."""
    new_cols = []
    for c in df.columns:
        try:
            new_cols.append(f"{float(c):.2f}")
        except ValueError:
            new_cols.append(c)
    df.columns = new_cols
    return df


def get_wavelength_cols(df: pd.DataFrame):
    """Extract numeric wavelength column names."""
    return [c for c in df.columns if str(c).replace('.', '').isdigit()]


def plot_nan_counts(df, wavelength_cols, title_suffix=""):
    """
    Plot number of NaN values per wavelength.
    
    Args:
        df: DataFrame with wavelength columns
        wavelength_cols: List of wavelength column names
        title_suffix: Additional text for plot title
    """
    clean_cols = []
    for c in wavelength_cols:
        try:
            clean_cols.append(float(str(c).strip()))
        except ValueError:
            continue

    na_counts = df[wavelength_cols].isna().sum()
    na_counts.index = clean_cols
    na_counts = na_counts.sort_index()

    plt.figure(figsize=(12, 5))
    plt.bar(na_counts.index, na_counts.values, width=1.0, edgecolor='black')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Number of NaNs")
    plt.title(f"NaN Distribution per Wavelength {title_suffix}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_mean_percent_error(resampled_raw, resampled_interp, wavelength_cols, sensor_desc=""):
    """
    Plot mean percent error between raw and interpolated spectra.
    
    Args:
        resampled_raw: Raw resampled DataFrame
        resampled_interp: Interpolated resampled DataFrame
        wavelength_cols: List of wavelength column names
        sensor_desc: Sensor description for title
    """
    mpe_per_band = []
    wl_sorted = sorted([float(w) for w in wavelength_cols])
    
    for col in wavelength_cols:
        arr1 = resampled_raw[col].astype(float).values
        arr2 = resampled_interp[col].astype(float).values
        valid_mask = (arr2 != 0) & np.isfinite(arr1) & np.isfinite(arr2)
        if np.any(valid_mask):
            mpe = np.nanmean(((arr1[valid_mask] - arr2[valid_mask]) / arr2[valid_mask]) * 100)
        else:
            mpe = np.nan
        mpe_per_band.append(mpe)

    overall_mpe = np.nanmean(mpe_per_band)

    plt.figure(figsize=(12, 5))
    plt.plot(wl_sorted, mpe_per_band, marker='o', linestyle='-', alpha=0.7, label="Mean % Error")
    plt.axhline(overall_mpe, color='orange', linestyle='--', label=f"Overall MPE = {overall_mpe:.2f}%")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Mean Percent Error (%)")
    plt.ticklabel_format(style='plain', axis='y')
    plt.title(f"Mean Percent Error Across Bands - {sensor_desc}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_spectral_library(df_long, sensor_desc="", class_col='class', source_col='source'):
    """
    Plot spectral library by class using FacetGrid.
    
    Args:
        df_long: Long-format DataFrame with wavelength, reflectance columns
        sensor_desc: Sensor description for title
        class_col: Column name for faceting (e.g., 'class')
        source_col: Column name for color/hue (e.g., 'source')
    """
    # Check if required columns exist
    if class_col not in df_long.columns:
        class_col = None
    if source_col not in df_long.columns:
        source_col = None

    if class_col:
        g = sns.FacetGrid(df_long, col=class_col, col_wrap=3, sharey=False, sharex=True)
    else:
        plt.figure(figsize=(12, 6))
        g = None

    if g:
        if source_col:
            g.map_dataframe(
                sns.lineplot,
                x="wavelength",
                y="reflectance",
                hue=source_col,
                units="id_lib",
                estimator=None,
                alpha=0.2
            )
            g.add_legend(title=source_col, bbox_to_anchor=(0.3, -0.05), loc="upper center", ncol=2)
        else:
            g.map(
                plt.plot,
                "wavelength",
                "reflectance",
                alpha=0.2
            )
        g.set_axis_labels("Wavelength (nm)", "Reflectance")
        g.set_titles("{col_name}")
        plt.subplots_adjust(top=0.9, bottom=0.15)
        g.fig.suptitle(f"Spectral Library - {sensor_desc}")
    else:
        if source_col and source_col in df_long.columns:
            for source in df_long[source_col].unique():
                df_sub = df_long[df_long[source_col] == source]
                for lib_id in df_sub["id_lib"].unique():
                    df_lib = df_sub[df_sub["id_lib"] == lib_id]
                    plt.plot(df_lib["wavelength"], df_lib["reflectance"], alpha=0.2, label=source)
        else:
            for lib_id in df_long["id_lib"].unique():
                df_lib = df_long[df_long["id_lib"] == lib_id]
                plt.plot(df_lib["wavelength"], df_lib["reflectance"], alpha=0.2)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.title(f"Spectral Library - {sensor_desc}")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def compute_nrmse(resampled_df, spectral_lib, wavelength_cols, original_wavelengths):
    """
    Compute Normalized RMSE between original and resampled library.
    
    Args:
        resampled_df: Resampled reflectance DataFrame
        spectral_lib: Original spectral library DataFrame
        wavelength_cols: Resampled wavelength columns
        original_wavelengths: Original wavelength array
    
    Returns:
        Tuple of (nrmse_per_band, overall_nrmse)
    """
    spectral_lib_wavelength_cols = [
        c for c in spectral_lib.columns 
        if str(c).replace(".", "").replace("-", "").isdigit()
    ]
    
    reflectance_resampled_wls = np.array([float(w) for w in wavelength_cols])
    spec_lib_wls = np.array([float(w) for w in spectral_lib_wavelength_cols])
    
    # Find closest original wavelengths for each resampled wavelength
    closest_spec_lib_cols = [
        spectral_lib_wavelength_cols[np.argmin(np.abs(spec_lib_wls - w))] 
        for w in reflectance_resampled_wls
    ]
    
    # Find matching library IDs
    matching_ids = list(
        set(resampled_df['id_lib']).intersection(set(spectral_lib['id_lib']))
    )
    
    if not matching_ids:
        print("  Warning: No matching library IDs found. NRMSE computation skipped.")
        return [], np.nan
    
    resampled_matched = resampled_df[resampled_df['id_lib'].isin(matching_ids)].set_index('id_lib')
    spectral_lib_matched = spectral_lib[spectral_lib['id_lib'].isin(matching_ids)].set_index('id_lib')

    nrmse_per_band = []
    for out_col, spec_col in zip(wavelength_cols, closest_spec_lib_cols):
        arr1 = resampled_matched.loc[matching_ids, out_col].astype(float).values
        arr2 = spectral_lib_matched.loc[matching_ids, spec_col].astype(float).values
        rmse = np.sqrt(np.nanmean((arr1 - arr2) ** 2))
        range_ref = np.nanmax(arr2) - np.nanmin(arr2)
        nrmse_per_band.append(rmse / range_ref if range_ref != 0 else np.nan)

    overall_nrmse = np.nanmean(nrmse_per_band)
    return nrmse_per_band, overall_nrmse


def plot_nrmse(wavelength_cols, nrmse_per_band, overall_nrmse, sensor_desc=""):
    """Plot NRMSE across bands."""
    plt.figure(figsize=(12, 5))
    wl_sorted = sorted([float(w) for w in wavelength_cols])
    plt.plot(wl_sorted, nrmse_per_band, marker='o', linestyle='-', alpha=0.7, label="NRMSE")
    plt.axhline(overall_nrmse, color='orange', linestyle='--', 
                label=f"Overall NRMSE = {overall_nrmse:.4f}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized RMSE")
    plt.title(f"NRMSE Across Bands - {sensor_desc}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Main Processing
# ============================================================================
def main(sensor, original_lib_path, resampled_dir):
    """
    Main visualization workflow.
    
    Args:
        sensor: Sensor name (e.g., 'AVIRIS-3', 'enmap', 'landsat-8')
        original_lib_path: Path to original spectral library CSV
        resampled_dir: Directory containing resampled CSV files
    """
    print(f"\n{'='*70}")
    print(f"Spectral Library Visualization")
    print(f"{'='*70}")
    print(f"Sensor: {sensor}")
    print(f"Original library: {original_lib_path}")
    print(f"Resampled directory: {resampled_dir}\n")
    
    # Load sensor configuration
    sensor_cfg = load_sensor_config(sensor)
    sensor_desc = sensor_cfg["description"]
    output_prefix = sensor_cfg["output_prefix"]
    
    # Construct file paths
    resampled_raw_path = os.path.join(resampled_dir, f"{output_prefix}.csv")
    resampled_interp_path = os.path.join(resampled_dir, f"{output_prefix}_interpolated.csv")
    
    # Check that files exist
    if not os.path.exists(resampled_raw_path):
        raise FileNotFoundError(f"Resampled raw file not found: {resampled_raw_path}")
    if not os.path.exists(resampled_interp_path):
        raise FileNotFoundError(f"Resampled interpolated file not found: {resampled_interp_path}")
    if not os.path.exists(original_lib_path):
        raise FileNotFoundError(f"Original library not found: {original_lib_path}")
    
    print(f"[1] Loading spectral library files...")
    resampled_raw_df = pd.read_csv(resampled_raw_path)
    resampled_interp_df = pd.read_csv(resampled_interp_path)
    spectral_lib = pd.read_csv(original_lib_path, low_memory=False)
    
    # Format wavelength columns
    resampled_raw_df = format_wavelength_cols(resampled_raw_df)
    resampled_interp_df = format_wavelength_cols(resampled_interp_df)
    
    wavelength_cols = get_wavelength_cols(resampled_interp_df)
    print(f"  Loaded {len(resampled_raw_df)} spectra with {len(wavelength_cols)} bands")
    
    raw_nans = np.isnan(resampled_raw_df[wavelength_cols]).any(axis=1).sum()
    interp_nans = np.isnan(resampled_interp_df[wavelength_cols]).any(axis=1).sum()
    print(f"  Raw: {raw_nans} rows with NaN values")
    print(f"  Interpolated: {interp_nans} rows with NaN values")
    
    # --- Plot 1: NaN counts ---
    print(f"\n[2] Plotting NaN distribution...")
    plot_nan_counts(resampled_raw_df, wavelength_cols, f"(Raw) - {sensor_desc}")
    plot_nan_counts(resampled_interp_df, wavelength_cols, f"(Interpolated) - {sensor_desc}")
    
    # --- Plot 2: Mean Percent Error ---
    print(f"\n[3] Plotting Mean Percent Error...")
    plot_mean_percent_error(resampled_raw_df, resampled_interp_df, wavelength_cols, sensor_desc)
    
    # --- Plot 3: Spectral Library ---
    print(f"\n[4] Plotting spectral library...")
    meta_cols = [c for c in resampled_interp_df.columns if not str(c).replace('.', '').isdigit()]
    df_long = resampled_interp_df.melt(
        id_vars=meta_cols,
        value_vars=wavelength_cols,
        var_name="wavelength",
        value_name="reflectance"
    )
    df_long["wavelength"] = pd.to_numeric(df_long["wavelength"], errors="coerce")
    
    # Try to use 'class' for faceting if available
    class_col = 'class' if 'class' in df_long.columns else None
    source_col = 'source' if 'source' in df_long.columns else None
    plot_spectral_library(df_long, sensor_desc, class_col=class_col, source_col=source_col)
    
    # --- Plot 4: NRMSE ---
    print(f"\n[5] Computing and plotting NRMSE...")
    nrmse_per_band, overall_nrmse = compute_nrmse(
        resampled_interp_df, 
        spectral_lib, 
        wavelength_cols,
        None
    )
    
    if len(nrmse_per_band) > 0:
        plot_nrmse(wavelength_cols, nrmse_per_band, overall_nrmse, sensor_desc)
        print(f"  Overall NRMSE: {overall_nrmse:.4f}")
    
    print(f"\n{'='*70}")
    print(f"Visualization complete!")
    print(f"{'='*70}\n")


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    try:
        main(
            sensor=SENSOR,
            original_lib_path=ORIGINAL_LIB_PATH,
            resampled_dir=RESAMPLED_DIR
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

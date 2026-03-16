"""
Spectral Library Resampling Script
----------------------------------
Resamples spectral libraries using Landsat band response functions,
interpolates missing data, and performs visualizations and NRMSE analysis.
"""

# ------------------------
# Imports
# ------------------------
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------
# Configuration
# ------------------------

# Choose Landsat version: 8 or 9
landsat_version = 9  # Change to 8 or 9 as needed

# Set response path conditionally based on Landsat version
if landsat_version == 8:
    landsat_response_path = "wavelength/L8_OLI_Ball_BA_RSR.v1.1-1.xlsx"
elif landsat_version == 9:
    landsat_response_path = "wavelength/L9_OLI2_Ball_BA_RSR.v2-1.xlsx"
else:
    raise ValueError("Invalid Landsat version. Please specify 8 or 9.")

landsat_bands = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
wavelengths = np.arange(350, 2501)  # adjust as needed
interpolate_lib = True  # whether to interpolate missing values
spectral_lib_path = "merged_lib.csv"
resampled_out_path = "spectral_library_resampled_landsat_response.csv"

# ------------------------
# Helper / Core Functions
# ------------------------

def load_spectral_library(path, wavelengths):
    """Load spectral library and report missing values."""
    spectral_lib = pd.read_csv(path)
    reflectance_values = spectral_lib[wavelengths.astype(str)].values
    na_count = np.isnan(reflectance_values).any(axis=1).sum()
    print(f"Number of rows with NaN in any wavelength column: {na_count}")
    return spectral_lib, reflectance_values

def load_landsat_responses(response_path, bands):
    """Load Landsat band response functions from Excel."""
    responses = {}
    for band in bands:
        df = pd.read_excel(response_path, sheet_name=band)
        wavelengths_landsat = df["Wavelength"].astype(int).tolist()
        weights = df["BA RSR [watts]"].astype(float).tolist()
        responses[band] = list(zip(wavelengths_landsat, weights))
    # Load band centers
    df_summary = pd.read_excel(response_path, sheet_name="Band summary")
    band_dict = dict(zip(df_summary['Band'], df_summary['Center Wavelength [nm]']))
    return responses, band_dict

def resample_data(array, marray, wavelength, responses, noDataValue=np.nan, feedback=None, isFirstBlock=True):
    """Resample spectral data using band response functions."""
    wavelength = [int(round(v)) for v in wavelength]
    outarray = []

    for name in responses:
        weightsByWavelength = dict(responses[name])
        indices, weights = [], []

        for idx, wl in enumerate(wavelength):
            weight = weightsByWavelength.get(wl)
            if weight is not None:
                indices.append(idx)
                weights.append(weight)

        if not indices:
            if isFirstBlock:
                message = (
                    f'No source bands ({min(wavelength)}–{max(wavelength)} nm) '
                    f'are covered by target band "{name}" '
                    f'({min(weightsByWavelength.keys())}–{max(weightsByWavelength.keys())} nm).'
                )
                if feedback:
                    feedback.pushWarning(message)
                else:
                    print("Warning:", message)
            outarray.append(np.full_like(array[0], noDataValue, dtype=array[0].dtype))
            continue

        tmparray = np.asarray(array, np.float32)[indices]
        tmpmarray = np.asarray(marray)[indices]
        warray = np.array(weights).reshape((-1, 1, 1)) * np.ones_like(tmparray)

        for tmparr, warr, marr in zip(tmparray, warray, tmpmarray):
            invalid = np.logical_not(marr)
            tmparr[invalid] = np.nan
            warr[invalid] = np.nan

        outarr = np.nansum(tmparray * warray, 0) / np.nansum(warray, 0)
        outarr[np.isnan(outarr)] = noDataValue
        outarray.append(outarr)

    return outarray

def interpolate_nan_rows(reflectance_values):
    """Interpolate NaNs in each row of the reflectance array."""
    interp_values = reflectance_values.copy()
    for i in range(reflectance_values.shape[0]):
        row = reflectance_values[i]
        nans = np.isnan(row)
        if np.any(nans):
            not_nan_idx = np.where(~nans)[0]
            nan_idx = np.where(nans)[0]
            if len(not_nan_idx) >= 2:
                interp_values[i, nans] = np.interp(nan_idx, not_nan_idx, row[not_nan_idx])
    na_count = np.isnan(interp_values).any(axis=1).sum()
    print(f"Number of rows with NaN after interpolation: {na_count}")
    return interp_values

def save_resampled(df, path):
    """Save the resampled DataFrame to CSV."""
    df.to_csv(path, index=False)
    print(f"Resampled spectral library saved to:\n{path}")

def plot_spectral_library(df_long):
    """Plot spectral library per class using Seaborn."""
    g = sns.FacetGrid(df_long, col="class", col_wrap=3, sharey=False, sharex=True)
    g.map_dataframe(
        sns.lineplot,
        x="wavelength",
        y="reflectance",
        hue="source",
        units="id_lib",
        estimator=None,
        alpha=0.2
    )
    g.set_axis_labels("Wavelength (nm)", "Reflectance")
    g.set_titles("{col_name}")
    g.add_legend(title="Source", bbox_to_anchor=(0.3, -0.05), loc="upper center", ncol=2)
    plt.subplots_adjust(top=0.9, bottom=0.15)
    g.fig.suptitle("Spectral Library (Water Bands Removed)")
    plt.show()

def compute_nrmse(resampled_df, spectral_lib, wavelength_cols, wavelengths):
    """Compute NRMSE between resampled and original spectral library."""
    spectral_lib_wavelength_cols = [
        c for c in spectral_lib.columns if str(c).replace(".", "").replace("-", "").isdigit()
    ]
    reflectance_resampled_wls = np.array([float(w) for w in wavelength_cols])
    spec_lib_wls = np.array([float(w) for w in spectral_lib_wavelength_cols])
    closest_spec_lib_cols = [
        spectral_lib_wavelength_cols[np.argmin(np.abs(spec_lib_wls - w))] for w in reflectance_resampled_wls
    ]
    matching_ids = list(set(resampled_df['id_lib']).intersection(set(spectral_lib['id_lib'])))
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

    plt.figure(figsize=(12, 5))
    plt.plot(sorted([float(w) for w in closest_spec_lib_cols]), nrmse_per_band, marker='o', linestyle='-', alpha=0.7, label="NRMSE")
    plt.axhline(overall_nrmse, color='orange', linestyle='--', label=f"Overall NRMSE = {overall_nrmse:.4f}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized RMSE")
    plt.title("Normalized RMSE Across Bands (Matched IDs)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return nrmse_per_band, overall_nrmse

# ------------------------
# Main Execution
# ------------------------
def main():
    class DummyFeedback:
        def pushWarning(self, msg):
            print("Warning:", msg)

    feedback = DummyFeedback()

    # Load data
    spectral_lib, reflectance_values = load_spectral_library(spectral_lib_path, wavelengths)
    responses, band_dict = load_landsat_responses(landsat_response_path, landsat_bands)

    # Resample original
    array = np.expand_dims(reflectance_values.T, axis=2)
    marray = np.ones_like(array, dtype=bool)
    resampled = resample_data(array, marray, wavelengths, responses, feedback=feedback)
    reflectance_resampled = pd.DataFrame({band: arr.squeeze() for band, arr in zip(landsat_bands, resampled)})
    reflectance_resampled = reflectance_resampled.rename(columns=band_dict)

    meta_cols = spectral_lib.drop(columns=[str(w) for w in wavelengths])
    reflectance_resampled = pd.concat([meta_cols.reset_index(drop=True), reflectance_resampled.reset_index(drop=True)], axis=1)

    # Interpolation
    if interpolate_lib:
        reflectance_values_interp = interpolate_nan_rows(reflectance_values)
        array = np.expand_dims(reflectance_values_interp.T, axis=2)
        resampled_interp = resample_data(array, marray, wavelengths, responses, feedback=feedback)
        reflectance_interpolated_resampled = pd.DataFrame({band: arr.squeeze() for band, arr in zip(landsat_bands, resampled_interp)})
        reflectance_interpolated_resampled = reflectance_interpolated_resampled.rename(columns=band_dict)
        reflectance_interpolated_resampled = pd.concat([meta_cols.reset_index(drop=True), reflectance_interpolated_resampled.reset_index(drop=True)], axis=1)
        reflectance_resampled = reflectance_interpolated_resampled

    # Save resampled
    save_resampled(reflectance_resampled, resampled_out_path)

    # Prepare long-format for plotting
    meta_cols_for_plot = ["class", "category_1", "category_2", "id_lib", "source"]
    wavelength_cols_for_plot = [c for c in reflectance_resampled.columns if str(c).replace(".", "").isdigit()]
    df_long = reflectance_resampled.melt(
        id_vars=meta_cols_for_plot,
        value_vars=wavelength_cols_for_plot,
        var_name="wavelength",
        value_name="reflectance"
    )
    df_long["wavelength"] = pd.to_numeric(df_long["wavelength"], errors="coerce")

    plot_spectral_library(df_long)
    compute_nrmse(reflectance_resampled, spectral_lib, wavelength_cols_for_plot, wavelengths)


if __name__ == "__main__":
    main()

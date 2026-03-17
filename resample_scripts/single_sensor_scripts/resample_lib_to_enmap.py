
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Configuration (User Parameters)
# ------------------------------------------------------------
WAVELENGTHS = np.arange(350, 2501)
SPECTRAL_LIB_PATH = "99_library_joined_with_lake.csv"
ENMAP_BAND_SPEC_PATH = "wavelength/enmap_spectral_config.csv"
OUT_DIR = "."

# Output files
RESAMPLED_RAW_PATH = os.path.join(OUT_DIR, "spectral_library_resampled_enmap.csv")
RESAMPLED_INTERP_PATH = os.path.join(OUT_DIR, "spectral_library_resampled_enmap_interpolated.csv")




# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
def interpolate_reflectance(reflectance_values: np.ndarray) -> np.ndarray:
    """Interpolates NaNs in each reflectance spectrum row-wise."""
    interpolated = reflectance_values.copy()
    for i in range(reflectance_values.shape[0]):
        row = reflectance_values[i]
        nans = np.isnan(row)
        if np.any(nans):
            not_nan_idx = np.where(~nans)[0]
            nan_idx = np.where(nans)[0]
            if len(not_nan_idx) >= 2:
                interpolated_vals = np.interp(nan_idx, not_nan_idx, row[not_nan_idx])
                interpolated[i, nans] = interpolated_vals
    return interpolated


def resample_spectrum(reflectance, wavelengths, band_centers, band_fwhm):
    """Resample a single spectrum using Gaussian convolution matching EnMAP conventions."""
    resampled = np.zeros(len(band_centers))
    for i, (center, fwhm) in enumerate(zip(band_centers, band_fwhm)):
        sigma = fwhm / 2.355
        a = 2 * sigma ** 2
        b = sigma * np.sqrt(2 * np.pi)
        xs = np.arange(int(center - sigma * 3), int(center + sigma * 3) + 2)
        weights = np.exp(-((xs - center) ** 2) / a) / b
        weights = weights / np.max(weights)
        xs_clipped = np.clip(xs, wavelengths[0], wavelengths[-1])
        interp_ref = np.interp(xs_clipped, wavelengths, reflectance)
        resampled[i] = np.sum(interp_ref * weights) / np.sum(weights)
    return resampled


def resample_library(reflectance_values, wavelengths, band_centers, band_fwhm, n_jobs=10):
    """Parallelized resampling for an entire spectral library."""
    return np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(resample_spectrum)(reflectance_values[i], wavelengths, band_centers, band_fwhm)
            for i in range(reflectance_values.shape[0])
        )
    )


# ------------------------------------------------------------
# Main Processing
# ------------------------------------------------------------
def main():
    # --- Load spectral library ---
    print("Loading spectral library...")
    spectral_lib = pd.read_csv(SPECTRAL_LIB_PATH)
    reflectance_values = spectral_lib[WAVELENGTHS.astype(str)].values

    print(f"Rows with NaN values: {np.isnan(reflectance_values).any(axis=1).sum()}")

    # --- Load EnMAP band configuration ---
    enmap = pd.read_csv(ENMAP_BAND_SPEC_PATH, header=None)
    band_centers = enmap.iloc[:, 0].values
    band_fwhm = enmap.iloc[:, 1].values

    # --- Resample Uninterpolated Library ---
    print("Resampling uninterpolated spectra...")
    resampled_raw = resample_library(reflectance_values, WAVELENGTHS, band_centers, band_fwhm, n_jobs=10)
    meta_cols = spectral_lib.drop(columns=WAVELENGTHS.astype(str))
    band_cols = [f"{float(b):.2f}" for b in np.round(band_centers, 2)]
    # Concatenate metadata and resampled spectra
    resampled_raw_df = pd.concat(
        [
            meta_cols.reset_index(drop=True),
            pd.DataFrame(resampled_raw, columns=band_cols)
        ],
        axis=1,
    )
    resampled_raw_df.to_csv(RESAMPLED_RAW_PATH, index=False)
    print(f"Saved uninterpolated resampled library → {RESAMPLED_RAW_PATH}")

    # --- Interpolate and Resample ---
    print("Interpolating missing values...")
    reflectance_interpolated = interpolate_reflectance(reflectance_values)

    print("Resampling interpolated spectra...")
    resampled_interp = resample_library(reflectance_interpolated, WAVELENGTHS, band_centers, band_fwhm, n_jobs=10)
    resampled_interp_df = pd.concat(
        [meta_cols.reset_index(drop=True),
         pd.DataFrame(resampled_interp, columns=band_cols)
         ],
        axis=1,
    )
    resampled_interp_df.to_csv(RESAMPLED_INTERP_PATH, index=False)
    print(f"Saved interpolated resampled library → {RESAMPLED_INTERP_PATH}")

    print("Processing complete")


# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()

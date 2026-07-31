"""
Parameterized Spectral Library Resampling Script
-------------------------------------------------
Resamples spectral libraries to different sensors using Gaussian convolution.
Sensor configuration is automatically detected and loaded.

Usage:
    1. Edit the "User Configuration" section below
    2. Run: python resample_lib_parameterized.py
"""

import os
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (where wavelength/ folder is)
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


# ============================================================================
# USER CONFIGURATION
# ============================================================================
# Edit these parameters before running the script

OUTPUT_DIR = r"C:\Users\schnesha\Downloads\resample_test"  # Output directory for resampled files
N_JOBS = 7  # Number of parallel jobs
SENSOR = "sentinel-2a"  # Options: AVIRIS-3, AVIRIS-NG, enmap, PRISMA, EMIT, landsat-8, landsat-9, sentinel-2a, sentinel-2b, sentinel-2c

# ============================================================================
# Input Lib CONFIGURATION
# ============================================================================
INPUT_LIB_PATH = r"E:\Project_EnFireMap\01_data\03_spectral_libraries\99_library_joined_with_lake.csv"  # Path to your spectral library

# For Spectral libraries with 1 nm res 
INPUT_LIB_WAVELENGTHS = np.arange(350, 2501)

# If enmap image endmembers are to be resampled 
if true:
    INPUT_LIB_WAVELENGTHS =   np.array([
        418.416, 424.043, 429.457, 434.686, 439.757, 444.699, 449.539, 454.306, 459.031, 463.73,
        468.411, 473.08, 477.744, 482.411, 487.087, 491.78, 496.497, 501.243, 506.02, 510.828,
        515.672, 520.55, 525.467, 530.424, 535.422, 540.463, 545.551, 550.687, 555.873, 561.112,
        566.405, 571.756, 577.166, 582.636, 588.171, 593.773, 599.446, 605.193, 611.017, 616.923,
        622.92, 628.987, 635.112, 641.294, 647.537, 653.841, 660.207, 666.637, 673.131, 679.691,
        686.319, 693.014, 699.78, 706.617, 713.524, 720.501, 727.545, 734.654, 741.826, 749.06,
        756.353, 763.703, 771.108, 778.567, 786.078, 793.639, 801.248, 808.905, 816.608, 824.355,
        832.145, 839.976, 847.847, 855.757, 863.703, 871.683, 879.692, 887.729, 895.789, 901.962,
        911.572, 921.32, 931.204, 941.218, 951.361, 961.629, 972.017, 982.524, 993.145, 1003.88,
        1014.72, 1025.66, 1036.7, 1047.84, 1059.07, 1070.39, 1081.79, 1093.26, 1104.81,
        1116.43, 1128.11, 1139.84, 1151.62, 1163.44, 1175.31, 1187.2, 1199.11, 1211.05,
        1223, 1234.97, 1246.95, 1258.93, 1270.92, 1282.92, 1294.91, 1306.9, 1318.88,
        1330.86, 1461.11, 1472.74, 1484.34, 1495.89, 1507.4, 1518.87, 1530.29, 1541.68,
        1553.01, 1564.31, 1575.55, 1586.76, 1597.91, 1609.02, 1620.09, 1631.11, 1642.08,
        1653, 1663.87, 1674.7, 1685.47, 1696.2, 1706.87, 1717.5, 1728.08, 1738.6,
        1749.08, 1759.51, 1967.66, 1977.08, 1986.45, 1995.79, 2005.08, 2014.33, 2023.54,
        2032.71, 2041.83, 2050.92, 2059.96, 2068.97, 2077.93, 2086.86, 2095.74, 2104.59,
        2113.4, 2122.17, 2130.9, 2139.6, 2148.26, 2156.88, 2165.47, 2174.02, 2182.53,
        2191.01, 2199.46, 2207.86, 2216.24, 2224.58, 2232.89, 2241.16, 2249.4, 2257.61,
        2265.79, 2273.93, 2282.04, 2290.12, 2298.17, 2306.19, 2314.18, 2322.13, 2330.05,
        2337.94, 2345.81, 2353.64, 2361.44, 2369.21, 2376.95, 2384.66, 2392.34, 2400,
        2407.62, 2415.21, 2422.78, 2430.32, 2437.83, 2445.3
    ], dtype=np.float64)

# ============================================================================
# Sensor Configuration Registry
# ============================================================================
SENSOR_CONFIG = {
    "AVIRIS-3": {
        "config_file": os.path.join(PROJECT_DIR, "wavelength", "AVIRIS-3_spectral_config.csv"),
        "output_prefix": "spectral_library_resampled_AVIRIS-3",
        "description": "AVIRIS-3 sensor configuration",
        "resampling_method": "gaussian"
    },
    "AVIRIS-NG": {
        "config_file": os.path.join(PROJECT_DIR, "wavelength", "AVIRIS-NG_spectral_config.csv"),
        "output_prefix": "spectral_library_resampled_AVIRIS-NG",
        "description": "AVIRIS-NG sensor configuration",
        "resampling_method": "gaussian"
    },
    "enmap": {
        "config_file": os.path.join(PROJECT_DIR, "wavelength", "enmap_spectral_config.csv"),
        "output_prefix": "spectral_library_resampled_enmap",
        "description": "EnMAP sensor configuration",
        "resampling_method": "gaussian"
    },
    "PRISMA": {
        "config_file": os.path.join(PROJECT_DIR, "wavelength", "PRISMA_spectral_config.csv"),
        "output_prefix": "spectral_library_resampled_PRISMA",
        "description": "PRISMA sensor configuration",
        "resampling_method": "gaussian"
    },
    "EMIT": {
        "config_file": os.path.join(PROJECT_DIR, "wavelength", "EMIT_spectral_config.csv"),
        "output_prefix": "spectral_library_resampled_EMIT",
        "description": "EMIT sensor configuration",
        "resampling_method": "gaussian"
    },
    "landsat-8": {
        "response_file": os.path.join(PROJECT_DIR, "wavelength", "L8_OLI_Ball_BA_RSR.v1.1-1.xlsx"),
        "output_prefix": "spectral_library_resampled_landsat8",
        "description": "Landsat 8 OLI sensor configuration",
        "resampling_method": "response_function",
        "bands": ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
    },
    "landsat-9": {
        "response_file": os.path.join(PROJECT_DIR, "wavelength", "L9_OLI2_Ball_BA_RSR.v2-1.xlsx"),
        "output_prefix": "spectral_library_resampled_landsat9",
        "description": "Landsat 9 OLI-2 sensor configuration",
        "resampling_method": "response_function",
        "bands": ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
    },
    "sentinel-2a": {
        "response_file": os.path.join(PROJECT_DIR, "wavelength", "COPE-GSEG-EOPG-TN-15-0007 - Sentinel-2 Spectral Response Functions 2024 - 4.0.xlsx"),
        "sheet_name": "Spectral Responses (S2A)",
        "output_prefix": "spectral_library_resampled_sentinel2a",
        "description": "Sentinel-2A sensor configuration",
        "resampling_method": "sentinel_response_function"
    },
    "sentinel-2b": {
        "response_file": os.path.join(PROJECT_DIR, "wavelength", "COPE-GSEG-EOPG-TN-15-0007 - Sentinel-2 Spectral Response Functions 2024 - 4.0.xlsx"),
        "sheet_name": "Spectral Responses (S2B)",
        "output_prefix": "spectral_library_resampled_sentinel2b",
        "description": "Sentinel-2B sensor configuration",
        "resampling_method": "sentinel_response_function"
    },
    "sentinel-2c": {
        "response_file": os.path.join(PROJECT_DIR, "wavelength", "COPE-GSEG-EOPG-TN-15-0007 - Sentinel-2 Spectral Response Functions 2024 - 4.0.xlsx"),
        "sheet_name": "Spectral Responses (S2C)",
        "output_prefix": "spectral_library_resampled_sentinel2c",
        "description": "Sentinel-2C sensor configuration",
        "resampling_method": "sentinel_response_function"
    },
}


# ============================================================================
# Utility Functions
# ============================================================================
def interpolate_reflectance(reflectance_values: np.ndarray) -> np.ndarray:
    """
    Interpolates NaNs in each reflectance spectrum row-wise.
    
    Args:
        reflectance_values: 2D array of shape (n_spectra, n_wavelengths)
    
    Returns:
        Interpolated reflectance array with NaNs filled where possible
    """
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
    """
    Resample a single spectrum using Gaussian convolution.
    
    Convolves the input spectrum with a Gaussian response function for each
    band, matching industry conventions (EnMAP-style).
    
    Args:
        reflectance: 1D array of reflectance values at input wavelengths
        wavelengths: 1D array of input wavelengths
        band_centers: 1D array of output Band center wavelengths
        band_fwhm: 1D array of Full-Width-Half-Maximum for each band
    
    Returns:
        1D array of resampled reflectance values
    """
    resampled = np.zeros(len(band_centers))
    for i, (center, fwhm) in enumerate(zip(band_centers, band_fwhm)):
        sigma = fwhm / 2.355  # Convert FWHM to standard deviation
        a = 2 * sigma ** 2
        b = sigma * np.sqrt(2 * np.pi)
        
        # Define Gaussian kernel points
        xs = np.arange(int(center - sigma * 3), int(center + sigma * 3) + 2)
        weights = np.exp(-((xs - center) ** 2) / a) / b
        weights = weights / np.max(weights)
        
        # Clip to valid wavelength range
        xs_clipped = np.clip(xs, wavelengths[0], wavelengths[-1])
        
        # Interpolate reflectance at kernel points
        interp_ref = np.interp(xs_clipped, wavelengths, reflectance)
        
        # Apply weighted convolution
        resampled[i] = np.sum(interp_ref * weights) / np.sum(weights)
    
    return resampled


def resample_library(reflectance_values, wavelengths, band_centers, band_fwhm, n_jobs=10):
    """
    Parallelized resampling for an entire spectral library.
    
    Args:
        reflectance_values: 2D array of shape (n_spectra, n_wavelengths)
        wavelengths: 1D array of input wavelengths
        band_centers: 1D array of output band center wavelengths
        band_fwhm: 1D array of FWHM for each band
        n_jobs: Number of parallel jobs (default: 10)
    
    Returns:
        2D array of resampled reflectance with shape (n_spectra, n_bands)
    """
    return np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(resample_spectrum)(reflectance_values[i], wavelengths, band_centers, band_fwhm)
            for i in range(reflectance_values.shape[0])
        )
    )


def load_landsat_responses(response_file, band_names):
    """
    Load Landsat band response functions from Excel file.
    
    Args:
        response_file: Path to Landsat response function Excel file
        band_names: List of band names to load
    
    Returns:
        Tuple of (responses dict, band_centers dict)
        - responses: {band_name: [(wavelength, weight), ...], ...}
        - band_centers: {band_name: center_wavelength, ...}
    
    Raises:
        FileNotFoundError: If response file does not exist
    """
    if not os.path.exists(response_file):
        raise FileNotFoundError(f"Landsat response file not found: {response_file}")
    
    responses = {}
    for band in band_names:
        df = pd.read_excel(response_file, sheet_name=band)
        wavelengths_ls = df["Wavelength"].astype(int).tolist()
        weights = df["BA RSR [watts]"].astype(float).tolist()
        responses[band] = list(zip(wavelengths_ls, weights))
    
    # Load band center wavelengths
    df_summary = pd.read_excel(response_file, sheet_name="Band summary")
    band_centers = dict(zip(df_summary['Band'], df_summary['Center Wavelength [nm]']))
    
    print(f"  Loaded {len(band_names)} bands from {response_file}")
    return responses, band_centers


def resample_spectrum_landsat(reflectance, wavelengths, responses):
    """
    Resample a single spectrum using Landsat response functions.
    
    Args:
        reflectance: 1D array of reflectance values
        wavelengths: 1D array of input wavelengths (integer nm)
        responses: Dict of {band_name: [(wavelength, weight), ...], ...}
    
    Returns:
        Dict of {band_name: resampled_value}
    """
    wavelengths_int = wavelengths.astype(int)
    resampled = {}
    
    for band_name, response_pairs in responses.items():
        weights_by_wl = dict(response_pairs)
        indices = []
        weights = []
        
        for idx, wl in enumerate(wavelengths_int):
            weight = weights_by_wl.get(wl)
            if weight is not None:
                indices.append(idx)
                weights.append(weight)
        
        if not indices:
            # No overlap between input and response function wavelengths
            resampled[band_name] = np.nan
        else:
            # Weighted average
            ref_subset = reflectance[indices]
            w_array = np.array(weights)
            
            # Handle NaNs in reflectance
            valid_mask = ~np.isnan(ref_subset)
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                resampled[band_name] = np.sum(ref_subset[valid_indices] * w_array[valid_indices]) / np.sum(w_array[valid_indices])
            else:
                resampled[band_name] = np.nan
    
    return resampled


def resample_library_landsat(reflectance_values, wavelengths, responses, band_names, n_jobs=10):
    """
    Parallelized Landsat resampling for entire spectral library.
    
    Args:
        reflectance_values: 2D array of shape (n_spectra, n_wavelengths)
        wavelengths: 1D array of input wavelengths
        responses: Dict of response functions
        band_names: List of band names in correct order
        n_jobs: Number of parallel jobs (default: 10)
    
    Returns:
        2D array of resampled reflectance with shape (n_spectra, n_bands)
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(resample_spectrum_landsat)(reflectance_values[i], wavelengths, responses)
        for i in range(reflectance_values.shape[0])
    )
    
    # Convert list of dicts to 2D array
    resampled = np.array([[result[band] for band in band_names] for result in results])
    return resampled


def load_sentinel_responses(response_file, sheet_names):
    """
    Load Sentinel-2 band response functions from Excel file.
    
    Args:
        response_file: Path to Sentinel-2 response function Excel file
        sheet_names: List of sheet names (e.g., ['Spectral responses (S2A)', 'Spectral responses (S2B)'])
    
    Returns:
        Tuple of (responses dict, band_centers dict)
        - responses: {sheet_name: {band_name: (wavelengths, weights), ...}, ...}
        - band_centers: {sheet_name: {band_name: center_wavelength, ...}, ...}
    
    Raises:
        FileNotFoundError: If response file does not exist
    """
    if not os.path.exists(response_file):
        raise FileNotFoundError(f"Sentinel-2 response file not found: {response_file}")
    
    all_responses = {}
    all_band_centers = {}
    
    for sheet_name in sheet_names:
        df = pd.read_excel(response_file, sheet_name=sheet_name)
        wavelengths_s2 = df["SR_WL"].values
        
        responses = {}
        band_centers = {}
        
        # Each column (except SR_WL) is a band's response function
        for col in df.columns:
            if col != "SR_WL":
                weights = df[col].values
                # Store as tuple of (wavelengths, weights) for interpolation
                responses[col] = (wavelengths_s2, weights)
                # Calculate band center as weighted mean
                valid_weights = weights[weights > 0]
                if len(valid_weights) > 0:
                    band_centers[col] = np.average(wavelengths_s2[weights > 0], weights=valid_weights)
                else:
                    band_centers[col] = np.mean(wavelengths_s2)
        
        all_responses[sheet_name] = responses
        all_band_centers[sheet_name] = band_centers
        print(f"  Loaded {len(responses)} bands from {sheet_name}")
    
    return all_responses, all_band_centers


def resample_spectrum_sentinel(reflectance, wavelengths, responses):
    """
    Resample a single spectrum using Sentinel-2 response functions.
    
    Args:
        reflectance: 1D array of reflectance values
        wavelengths: 1D array of input wavelengths
        responses: Dict of {band_name: (response_wavelengths, response_weights), ...}
    
    Returns:
        Dict of {band_name: resampled_value}
    """
    resampled = {}
    
    for band_name, (response_wl, response_weights) in responses.items():
        # Interpolate response function to input wavelengths
        interp_weights = np.interp(wavelengths, response_wl, response_weights, left=0, right=0)
        
        # Find wavelengths with non-zero response
        valid_mask = interp_weights > 0
        
        if not np.any(valid_mask):
            # No overlap between input and response function wavelengths
            resampled[band_name] = np.nan
        else:
            # Weighted average using interpolated response
            valid_indices = np.where(valid_mask)[0]
            ref_subset = reflectance[valid_indices]
            w_subset = interp_weights[valid_indices]
            
            # Handle NaNs in reflectance
            ref_valid_mask = ~np.isnan(ref_subset)
            if np.any(ref_valid_mask):
                ref_valid_indices = np.where(ref_valid_mask)[0]
                resampled[band_name] = np.sum(ref_subset[ref_valid_indices] * w_subset[ref_valid_indices]) / np.sum(w_subset[ref_valid_indices])
            else:
                resampled[band_name] = np.nan
    
    return resampled


def resample_library_sentinel(reflectance_values, wavelengths, responses, band_names, n_jobs=10):
    """
    Parallelized Sentinel-2 resampling for entire spectral library.
    
    Args:
        reflectance_values: 2D array of shape (n_spectra, n_wavelengths)
        wavelengths: 1D array of input wavelengths
        responses: Dict of response functions for a single sensor
        band_names: List of band names in correct order
        n_jobs: Number of parallel jobs (default: 10)
    
    Returns:
        2D array of resampled reflectance with shape (n_spectra, n_bands)
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(resample_spectrum_sentinel)(reflectance_values[i], wavelengths, responses)
        for i in range(reflectance_values.shape[0])
    )
    
    # Convert list of dicts to 2D array
    resampled = np.array([[result[band] for band in band_names] for result in results])
    return resampled


def load_sensor_config(sensor_name):
    """
    Load sensor configuration from registry.
    
    Args:
        sensor_name: Name of the sensor (e.g., 'AVIRIS-3', 'enmap')
    
    Returns:
        Dictionary with configuration paths and metadata
    
    Raises:
        ValueError: If sensor is not in the registry
    """
    if sensor_name not in SENSOR_CONFIG:
        available = ", ".join(SENSOR_CONFIG.keys())
        raise ValueError(f"Unknown sensor '{sensor_name}'. Available sensors: {available}")
    return SENSOR_CONFIG[sensor_name]


def load_band_config(config_file):
    """
    Load band center and FWHM from spectral config file.
    
    Args:
        config_file: Path to spectral configuration CSV
    
    Returns:
        Tuple of (band_centers, band_fwhm) as numpy arrays
    
    Raises:
        FileNotFoundError: If config file does not exist
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    config = pd.read_csv(config_file, header=None)
    band_centers = config.iloc[:, 0].values
    band_fwhm = config.iloc[:, 1].values
    
    print(f"  Loaded {len(band_centers)} bands from {config_file}")
    return band_centers, band_fwhm


# ============================================================================
# Main Processing
# ============================================================================

def main(sensor, input_lib_path, output_dir=".", wavelengths=INPUT_LIB_WAVELENGTHS, n_jobs=10):
    """
    Main resampling workflow.
    
    Args:
        sensor: Sensor name (e.g., 'AVIRIS-3', 'enmap', 'AVIRIS-NG', 'landsat-8', 'landsat-9', 'sentinel-2a', 'sentinel-2b', 'sentinel-2c')
        input_lib_path: Path to input spectral library CSV
        output_dir: Directory for output files (default: current directory)
        wavelengths: Input wavelength array (default: 350-2500 nm)
        n_jobs: Number of parallel jobs (default: 10)
    """
    print(f"\n{'='*70}")
    print(f"Spectral Library Resampling")
    print(f"{'='*70}")
    print(f"Sensor: {sensor}")
    print(f"Input: {input_lib_path}")
    print(f"Output: {output_dir}\n")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Load sensor configuration ---
    print(f"[1] Loading sensor configuration...")
    sensor_cfg = load_sensor_config(sensor)
    resampling_method = sensor_cfg.get("resampling_method", "gaussian")
    
    # --- Load spectral library ---
    print(f"\n[2] Loading spectral library...")
    if not os.path.exists(input_lib_path):
        raise FileNotFoundError(f"Input library not found: {input_lib_path}")
    
    spectral_lib = pd.read_csv(input_lib_path)
    reflectance_values = spectral_lib[wavelengths.astype(str)].values
    
    n_rows, n_cols = reflectance_values.shape
    nan_count = np.isnan(reflectance_values).any(axis=1).sum()
    print(f"  Loaded {n_rows} spectra × {n_cols} wavelengths")
    print(f"  Rows with NaN values: {nan_count}")
    
    # Extract metadata columns (non-wavelength columns)
    meta_cols = spectral_lib.drop(columns=wavelengths.astype(str))
    
    # --- Resampling: Gaussian (AVIRIS/EnMAP) ---
    if resampling_method == "gaussian":
        band_centers, band_fwhm = load_band_config(sensor_cfg["config_file"])
        
        # --- Resample Uninterpolated Library ---
        print(f"\n[3] Resampling uninterpolated spectra...")
        resampled_raw = resample_library(reflectance_values, wavelengths, band_centers, band_fwhm, n_jobs=n_jobs)
        
        band_cols = [f"{float(b):.2f}" for b in np.round(band_centers, 2)]
        
        resampled_raw_df = pd.concat(
            [
                meta_cols.reset_index(drop=True),
                pd.DataFrame(resampled_raw, columns=band_cols)
            ],
            axis=1,
        )
        
        # Use basename of input file + prefix
        base_name = os.path.splitext(os.path.basename(input_lib_path))[0]
        raw_out_path = os.path.join(output_dir, f"{base_name}_{sensor_cfg['output_prefix']}.csv")
        resampled_raw_df.to_csv(raw_out_path, index=False)
        print(f"  Saved → {raw_out_path}")
        
        # --- Interpolate and Resample ---
        print(f"\n[4] Interpolating missing values and resampling...")
        reflectance_interpolated = interpolate_reflectance(reflectance_values)
        resampled_interp = resample_library(reflectance_interpolated, wavelengths, band_centers, band_fwhm, n_jobs=n_jobs)
        
        resampled_interp_df = pd.concat(
            [
                meta_cols.reset_index(drop=True),
                pd.DataFrame(resampled_interp, columns=band_cols)
            ],
            axis=1,
        )
        
        interp_out_path = os.path.join(output_dir, f"{base_name}_{sensor_cfg['output_prefix']}_interpolated.csv")
        resampled_interp_df.to_csv(interp_out_path, index=False)
        print(f"  Saved → {interp_out_path}")
    
    # --- Resampling: Response Function (Landsat) ---
    elif resampling_method == "response_function":
        band_names = sensor_cfg["bands"]
        responses, band_centers_dict = load_landsat_responses(sensor_cfg["response_file"], band_names)
        
        # --- Resample Uninterpolated Library ---
        print(f"\n[3] Resampling uninterpolated spectra...")
        resampled_raw = resample_library_landsat(reflectance_values, wavelengths, responses, band_names, n_jobs=n_jobs)
        
        # Use band names sorted by center wavelength for column naming
        band_cols = [f"{band_centers_dict[band]:.1f}" for band in band_names]
        
        resampled_raw_df = pd.concat(
            [
                meta_cols.reset_index(drop=True),
                pd.DataFrame(resampled_raw, columns=band_cols)
            ],
            axis=1,
        )
        
        # Use basename of input file + prefix
        base_name = os.path.splitext(os.path.basename(input_lib_path))[0]
        raw_out_path = os.path.join(output_dir, f"{base_name}_{sensor_cfg['output_prefix']}.csv")
        resampled_raw_df.to_csv(raw_out_path, index=False)
        print(f"  Saved → {raw_out_path}")
        
        # --- Interpolate and Resample ---
        print(f"\n[4] Interpolating missing values and resampling...")
        reflectance_interpolated = interpolate_reflectance(reflectance_values)
        resampled_interp = resample_library_landsat(reflectance_interpolated, wavelengths, responses, band_names, n_jobs=n_jobs)
        
        resampled_interp_df = pd.concat(
            [
                meta_cols.reset_index(drop=True),
                pd.DataFrame(resampled_interp, columns=band_cols)
            ],
            axis=1,
        )
        
        interp_out_path = os.path.join(output_dir, f"{base_name}_{sensor_cfg['output_prefix']}_interpolated.csv")
        resampled_interp_df.to_csv(interp_out_path, index=False)
        print(f"  Saved → {interp_out_path}")
    
    # --- Resampling: Response Function (Sentinel-2) ---
    elif resampling_method == "sentinel_response_function":
        sheet_name = sensor_cfg["sheet_name"]
        all_responses, all_band_centers = load_sentinel_responses(sensor_cfg["response_file"], [sheet_name])
        responses = all_responses[sheet_name]
        band_centers_dict = all_band_centers[sheet_name]
        # Sort band names by their center wavelengths (smallest to largest)
        band_names = sorted(responses.keys(), key=lambda x: band_centers_dict[x])
        
        # --- Resample Uninterpolated Library ---
        print(f"\n[3] Resampling uninterpolated spectra...")
        resampled_raw = resample_library_sentinel(reflectance_values, wavelengths, responses, band_names, n_jobs=n_jobs)
        
        # Use band center wavelengths for column naming
        band_cols = [f"{band_centers_dict[band]:.1f}" for band in band_names]
        
        resampled_raw_df = pd.concat(
            [
                meta_cols.reset_index(drop=True),
                pd.DataFrame(resampled_raw, columns=band_cols)
            ],
            axis=1,
        )
        
        # Use basename of input file + prefix
        base_name = os.path.splitext(os.path.basename(input_lib_path))[0]
        raw_out_path = os.path.join(output_dir, f"{base_name}_{sensor_cfg['output_prefix']}.csv")
        resampled_raw_df.to_csv(raw_out_path, index=False)
        print(f"  Saved → {raw_out_path}")
        
        # --- Interpolate and Resample ---
        print(f"\n[4] Interpolating missing values and resampling...")
        reflectance_interpolated = interpolate_reflectance(reflectance_values)
        resampled_interp = resample_library_sentinel(reflectance_interpolated, wavelengths, responses, band_names, n_jobs=n_jobs)
        
        resampled_interp_df = pd.concat(
            [
                meta_cols.reset_index(drop=True),
                pd.DataFrame(resampled_interp, columns=band_cols)
            ],
            axis=1,
        )
        
        interp_out_path = os.path.join(output_dir, f"{base_name}_{sensor_cfg['output_prefix']}_interpolated.csv")
        resampled_interp_df.to_csv(interp_out_path, index=False)
        print(f"  Saved → {interp_out_path}")
    
    else:
        raise ValueError(f"Unknown resampling method: {resampling_method}")
    
    print(f"\n{'='*70}")
    print(f"Processing complete!")
    print(f"{'='*70}\n")


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    try:
        main(
            sensor=SENSOR,
            input_lib_path=INPUT_LIB_PATH,
            output_dir=OUTPUT_DIR,
            wavelengths=INPUT_LIB_WAVELENGTHS,
            n_jobs=N_JOBS
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

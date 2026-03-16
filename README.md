# Spectral Resampling Tools

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7+-green.svg)](https://www.python.org/)

A collection of Python scripts and Jupyter notebooks for resampling spectral libraries to match the spectral response functions of various remote sensing sensors, including EnMAP, AVIRIS-3, and Landsat.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Spectral Library Resampling**: Resample spectral reflectance data to sensor-specific band configurations
- **Multiple Sensor Support**:
  - EnMAP (Environmental Mapping and Analysis Program)
  - AVIRIS-3 (Airborne Visible/Infrared Imaging Spectrometer)
  - Landsat 8 and 9 (using response functions or FWHM)
- **Interpolation**: Handle missing data in spectral libraries with interpolation
- **Parallel Processing**: Utilize joblib for efficient parallel computation
- **Visualization**: Generate plots and analysis of resampled spectra
- **NRMSE Analysis**: Calculate Normalized Root Mean Square Error for quality assessment

## Installation

### Prerequisites

- Python 3.7 or higher
- Required Python packages:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - joblib
  - astropy (for convolution in Landsat resampling)
  - openpyxl (for reading Landsat response function Excel files)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Spectral_resampling.git
   cd Spectral_resampling
   ```

2. Install required packages:
   ```bash
   pip install numpy pandas matplotlib seaborn joblib astropy openpyxl
   ```

3. For Jupyter notebooks, ensure Jupyter is installed:
   ```bash
   pip install jupyter
   ```

## Usage

### Python Scripts

Run the scripts from the repository root directory. Each script resamples spectral data to a specific sensor:

- `resample_lib_to_enmap.py`: Resample to EnMAP bands
- `resample_lib_to_AVIRIS-3.py`: Resample to AVIRIS-3 bands
- `resample_lib_to_landsat.py`: Resample to Landsat bands using response functions

### Jupyter Notebooks

Open and run the notebooks for interactive analysis:

- `resample_lib_to_landsat_fwmh.ipynb`: Landsat resampling using Full Width at Half Maximum (FWHM)
- `resample_lib_to_landsat_response_function.ipynb`: Landsat resampling using response functions

### Configuration

Edit the configuration section at the top of each script/notebook to set:

- Input spectral library path
- Output directory
- Wavelength range
- Interpolation settings

## Project Structure

```
Spectral_resampling/
├── LICENSE
├── README.md
├── resample_scripts/
│   ├── resample_lib_to_AVIRIS-3.py
│   ├── resample_lib_to_AVIRIS-3_vis.py
│   ├── resample_lib_to_enmap.py
│   ├── resample_lib_to_enmap_vis.py
│   ├── resample_lib_to_landsat.py
│   ├── resample_lib_to_landsat_fwmh.ipynb
│   └── resample_lib_to_landsat_response_function.ipynb
└── wavelength/
    ├── AVIRIS-3_spectral_config.csv
    ├── AVIRIS-3_spectral_config_extract.py
    └── enmap_spectral_config.csv
```

- `resample_scripts/`: Main resampling scripts and notebooks
- `wavelength/`: Sensor spectral configuration files and extraction script

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Spectral configurations based on official sensor specifications
- Landsat response functions from USGS
- Inspired by various remote sensing spectral resampling techniques from the [enmap-box](https://github.com/EnMAP-Box/enmap-box)

## Contact

For questions or issues, please open an issue on GitHub.
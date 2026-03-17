import csv
import re

# Input HDR file and output CSV file paths
hdr_file = r"D:\Spectral_resampling\wavelength\ang20220318t192455_rfl_v2aa1_img.hdr"
csv_out = r"D:\Spectral_resampling\wavelength\AVIRIS-NG_spectral_config.csv"

# Read entire HDR file
with open(hdr_file, 'r') as f:
    hdr_text = f.read()

def extract_values(label, text):
    """Extract list of numeric values from an ENVI header array field."""
    match = re.search(rf'{label}\s*=\s*{{([^}}]+)}}', text, re.IGNORECASE)
    if not match:
        raise ValueError(f"{label} not found in HDR file.")
    return [float(v.strip()) for v in match.group(1).split(',') if v.strip()]

# Extract wavelength and fwhm
wavelengths = extract_values("wavelength", hdr_text)
fwhm = extract_values("fwhm", hdr_text)

# Check consistency
if len(wavelengths) != len(fwhm):
    raise ValueError(f"Length mismatch: wavelength={len(wavelengths)}, fwhm={len(fwhm)}")

# Write to CSV (no header)
with open(csv_out, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for w, f in zip(wavelengths, fwhm):
        writer.writerow([w, f])

print(f"Wrote {len(wavelengths)} rows to:\n{csv_out}")

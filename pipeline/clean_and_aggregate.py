import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from tqdm import tqdm

# data path
DATA_DIR = r"C:\HSI_Data\DOI-10-13012-b2idb-8141497_v1\all_data\all_data"
OUTPUT_CSV = "hyperspectral_features.csv"

def find_cube(mat_dict):
    """
    Automatically find the hyperspectral cube inside .mat file
    (ignores metadata keys like __header__)
    """
    for key, value in mat_dict.items():
        if not key.startswith("__") and isinstance(value, np.ndarray):
            if value.ndim == 3:
                return value, key
    raise ValueError("No 3D cube found in .mat file")

def ensure_band_first(cube):
    """
    Convert cube to shape: (bands, height, width)
    """
    if cube.shape[0] == 300:
        return cube  # already correct
    elif cube.shape[-1] == 300:
        return np.transpose(cube, (2, 0, 1))
    else:
        raise ValueError(f"Unexpected cube shape: {cube.shape}")

def extract_features(cube):
    """
    Extract per-wavelength features
    """
    bands, h, w = cube.shape
    features = {}

    # reshape: (bands, pixels)
    cube_reshaped = cube.reshape(bands, -1)

    for b in range(bands):
        band_data = cube_reshaped[b]

        features[f"band_{b}_mean"] = np.mean(band_data)
        features[f"band_{b}_std"] = np.std(band_data)
        features[f"band_{b}_min"] = np.min(band_data)
        features[f"band_{b}_max"] = np.max(band_data)
        features[f"band_{b}_median"] = np.median(band_data)
        features[f"band_{b}_q25"] = np.percentile(band_data, 25)
        features[f"band_{b}_q75"] = np.percentile(band_data, 75)
        features[f"band_{b}_skew"] = skew(band_data)
        features[f"band_{b}_kurtosis"] = kurtosis(band_data)

    # spectral slope (trend across wavelengths)
    mean_spectrum = np.mean(cube_reshaped, axis=1)
    features["spectrum_slope"] = np.polyfit(range(len(mean_spectrum)), mean_spectrum, 1)[0]

    # total energy
    features["total_intensity"] = np.sum(mean_spectrum)

    # normalized spectrum stats
    norm_spec = mean_spectrum / (np.linalg.norm(mean_spectrum) + 1e-8)
    features["norm_spec_mean"] = np.mean(norm_spec)
    features["norm_spec_std"] = np.std(norm_spec)
    return features

def main():
    all_rows = []

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".mat")]

    for file in tqdm(files):
        path = os.path.join(DATA_DIR, file)
        try:
            mat = loadmat(path)
            cube, cube_name = find_cube(mat)
            cube = ensure_band_first(cube)

            features = extract_features(cube)
            features["file_name"] = file

            all_rows.append(features)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved features to {OUTPUT_CSV}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    main()
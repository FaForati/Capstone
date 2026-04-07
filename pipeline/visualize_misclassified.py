import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

DATA_DIR = r"C:\HSI_Data\DOI-10-13012-b2idb-8141497_v1\all_data\all_data"
#MISCLASSIFIED_CSV = "misclassified_eggs_all_features.csv"
#MISCLASSIFIED_CSV = "high_confidence_eggs_all_features_LR.csv"
MISCLASSIFIED_CSV = "high_confidence_eggs_spectra_LR.csv"
#MISCLASSIFIED_CSV = "misclassified_eggs_spectra_LR.csv"

OUTPUT_DIR = "outputs"

# wavelengths
WAVELENGTHS = np.linspace(374, 1015, 300)

TARGET_BANDS = {
    "R": 656,
    "G": 767,
    "B": 860
}

def find_nearest_band(target):
    return np.argmin(np.abs(WAVELENGTHS - target))

band_indices = {k: find_nearest_band(v) for k, v in TARGET_BANDS.items()}

def find_cube(mat_dict):
    for key, value in mat_dict.items():
        if not key.startswith("__") and isinstance(value, np.ndarray):
            if value.ndim == 3:
                return value
    raise ValueError("No cube found")

def ensure_band_first(cube):
    if cube.shape[0] == 300:
        return cube
    elif cube.shape[-1] == 300:
        return np.transpose(cube, (2, 0, 1))
    else:
        raise ValueError("Unexpected shape")

def get_rgb_image(cube):
    r = cube[band_indices["R"]]
    g = cube[band_indices["G"]]
    b = cube[band_indices["B"]]

    rgb = np.stack([r, g, b], axis=-1)

    rgb = rgb - rgb.min()
    rgb = rgb / (rgb.max() + 1e-8)

    return rgb

# =========================
# LOAD IDS
# =========================
df = pd.read_csv(MISCLASSIFIED_CSV)

fertile = df[df["true_label"] == 1]
infertile = df[df["true_label"] == 0]

# =========================
# SAVE GRID IMAGE
# =========================
def save_grid(samples, title, save_path):
    n = len(samples)
    cols = 5
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(15, 3 * rows))

    for i, row in enumerate(samples.itertuples()):
        file_name = row.file_name[:-1] + ".mat" #[:-1]
        path = os.path.join(DATA_DIR, file_name)

        try:
            mat = loadmat(path)
            cube = ensure_band_first(find_cube(mat))
            rgb = get_rgb_image(cube)

            plt.subplot(rows, cols, i + 1)
            plt.imshow(rgb)
            plt.title(f"{row.file_name} confidence: {row.confidence:.2f}", fontsize=4)
            plt.axis("off")

        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    plt.suptitle(title)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()

# =========================
# RUN
# =========================

print("Saving grid visualizations...")
#save_grid(fertile, "Misclassified Fertile Eggs", os.path.join(OUTPUT_DIR, "fertile_misclassified_all_features_LR.png"))
#save_grid(infertile, "Misclassified Infertile Eggs", os.path.join(OUTPUT_DIR, "infertile_misclassified_all_features_LR.png"))
#save_grid(fertile, "High Confidence Fertile Eggs", os.path.join(OUTPUT_DIR, "fertile_grid_high_confidence_all_features_LR.png"))
#save_grid(infertile, "High Confidence Infertile Eggs", os.path.join(OUTPUT_DIR, "infertile_grid_high_confidence_all_features_LR.png"))
#save_grid(fertile, "Misclassified Fertile Eggs (Spectra Only)", os.path.join(OUTPUT_DIR, "fertile_misclassified_spectra_LR.png"))
#save_grid(infertile, "Misclassified Infertile Eggs (Spectra Only)", os.path.join(OUTPUT_DIR, "infertile_misclassified_spectra_LR.png"))    
save_grid(fertile, "High Confidence Fertile Eggs (Spectra Only)", os.path.join(OUTPUT_DIR, "fertile_grid_high_confidence_spectra_LR.png"))
save_grid(infertile, "High Confidence Infertile Eggs (Spectra Only)", os.path.join(OUTPUT_DIR, "infertile_grid_high_confidence_spectra_LR.png"))

print("Done! Outputs saved in:", OUTPUT_DIR)
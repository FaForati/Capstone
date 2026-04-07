import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# =========================
# CONFIG
# =========================
DATA_CSV = r"C:\HSI_Data\DOI-10-13012-b2idb-8141497_v1\spectra and reference parameters.csv"

LED_WAVELENGTHS = [520, 680, 940]
FWHM = 30  # nm (try 20, 30, 50)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_CSV)

df["label"] = df["Fertility status"].map({
    "Fertile": 1,
    "Infertile": 0
})

df = df.dropna(subset=["label"])

# =========================
# EXTRACT SPECTRAL DATA
# =========================
spectral_cols = [col for col in df.columns if "nm" in col or col.replace('.', '', 1).isdigit()]

wavelengths = np.array([float(col.replace("nm", "")) for col in spectral_cols])
X_spectrum = df[spectral_cols].values
y = df["label"].values

# =========================
# LED SIMULATION
# =========================
def gaussian_weights(wavelengths, center, fwhm):
    sigma = fwhm / 2.355
    weights = np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)
    return weights / np.sum(weights)

def simulate_leds(X_spectrum, wavelengths, led_centers, fwhm):
    X_led = []

    for center in led_centers:
        weights = gaussian_weights(wavelengths, center, fwhm)
        intensity = np.dot(X_spectrum, weights)
        X_led.append(intensity)

    return np.vstack(X_led).T

X_led = simulate_leds(X_spectrum, wavelengths, LED_WAVELENGTHS, FWHM)

print("Simulated LED feature shape:", X_led.shape)

# =========================
# ADD RATIO FEATURES
# =========================
def add_ratios(X):
    features = [X]
    n = X.shape[1]

    for i in range(n):
        for j in range(i + 1, n):
            ratio = X[:, i] / (X[:, j] + 1e-8)
            features.append(ratio.reshape(-1, 1))

    return np.hstack(features)

X_features = add_ratios(X_led)

# =========================
# MODEL
# =========================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

# =========================
# EVALUATION
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X_features, y, cv=cv, scoring="roc_auc")

print("\n=== LED Simulation Results ===")
print(f"Wavelengths: {LED_WAVELENGTHS}")
print(f"FWHM: {FWHM} nm")
print(f"ROC AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
import pandas as pd
import numpy as np
from itertools import combinations

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# =========================
# CONFIG
# =========================
DATA_CSV = r"C:\HSI_Data\InHouseData\InHouse_fertility_data.csv"#r"C:\HSI_Data\DOI-10-13012-b2idb-8141497_v1\spectra and reference parameters.csv"

# Candidate wavelengths (reduce search space for speed)
CANDIDATE_WAVELENGTHS = np.arange(450, 970, 20)  # 450–950 nm every 20 nm

NUM_LEDS_LIST = [6] #3,4, 5,
FWHM = 30

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_CSV)

df["label"] = df["Fertility status"].map({
    "Fertile": 1,
    "Infertile": 0,
    1: 1,
    0: 0
})

df = df.dropna(subset=["label"])

# Extract spectral columns
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

# =========================
# FEATURE ENGINEERING
# =========================
def add_ratios(X):
    features = [X]
    n = X.shape[1]
    for i in range(n):
        for j in range(i + 1, n):
            ratio = X[:, i] / (X[:, j] + 1e-8)
            features.append(ratio.reshape(-1, 1))
    return np.hstack(features)

# =========================
# MODEL
# =========================
def evaluate_model(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    aucs = []
    bal_accs = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression(
            max_iter=2000,
            #class_weight="balanced"
        )

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]

        # ROC AUC
        auc = roc_auc_score(y_test, y_prob)
        aucs.append(auc)

        # Optimal threshold (maximize balanced accuracy)
        thresholds = np.linspace(0.2, 0.8, 50)
        best_bal_acc = 0

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            best_bal_acc = max(best_bal_acc, bal_acc)

        bal_accs.append(best_bal_acc)

    return np.mean(aucs), np.std(aucs), np.mean(bal_accs)

# =========================
# SEARCH
# =========================
results = []
try:
    for num_leds in NUM_LEDS_LIST:
        print(f"\n=== Searching {num_leds} LEDs ===")

        for combo in combinations(CANDIDATE_WAVELENGTHS, num_leds):
            combo = list(combo)

            # simulate LEDs
            X_led = simulate_leds(X_spectrum, wavelengths, combo, FWHM)

            # add ratios (best feature)
            X_feat = add_ratios(X_led)

            auc_mean, auc_std, bal_acc = evaluate_model(X_feat, y)

            results.append({
                "num_leds": num_leds,
                "wavelengths": combo,
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "balanced_acc": bal_acc
            })

            print(f"{combo} → AUC: {auc_mean:.3f}, BalAcc: {bal_acc:.3f}")
except:
    print("Search interrupted. Saving results so far...")
    results_df = pd.DataFrame(results).sort_values(by="auc_mean", ascending=False)

    results_df.to_csv("partial_optimized_led_wavelengths_6_20_30.csv", index=False)

    print("\n=== TOP RESULTS ===")
    print(results_df.head(10))
# =========================
# SAVE RESULTS
# =========================
results_df = pd.DataFrame(results).sort_values(by="auc_mean", ascending=False)

results_df.to_csv("optimized_led_wavelengths_6_20_30.csv", index=False)

print("\n=== TOP RESULTS ===")
print(results_df.head(10))
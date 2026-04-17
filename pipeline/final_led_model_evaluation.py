import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)

# =========================
# CONFIG (PUT YOUR BEST HERE)
# =========================
DATA_CSV = r"C:\HSI_Data\DOI-10-13012-b2idb-8141497_v1\spectra and reference parameters.csv"#r"C:\HSI_Data\InHouseData\train.csv"

BEST_WAVELENGTHS = [530, 610, 670, 950]  # [530, 610, 670, 950]
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
# PREPARE FEATURES
# =========================
X_led = simulate_leds(X_spectrum, wavelengths, BEST_WAVELENGTHS, FWHM)
X = add_ratios(X_led)

# =========================
# TRAIN / VAL / TEST SPLIT
# =========================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

# =========================
# SCALE
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# =========================
# MODEL
# =========================
model = LogisticRegression(
    max_iter=2000,
    #class_weight="balanced"
)

model.fit(X_train, y_train)

# =========================
# FIND BEST THRESHOLD (on validation)
# =========================
y_val_prob = model.predict_proba(X_val)[:, 1]

thresholds = np.linspace(0.1, 0.9, 100)
best_threshold = 0.5
best_bal_acc = 0

for t in thresholds:
    y_val_pred = (y_val_prob >= t).astype(int)
    bal_acc = balanced_accuracy_score(y_val, y_val_pred)

    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        best_threshold = t

print(f"\nBest threshold: {best_threshold:.3f}")
print(f"Validation Balanced Accuracy: {best_bal_acc:.3f}")

# =========================
# FINAL TEST EVALUATION
# =========================
y_test_prob = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= best_threshold).astype(int)

auc = roc_auc_score(y_test, y_test_prob)
bal_acc = balanced_accuracy_score(y_test, y_test_pred)

print("\n=== TEST RESULTS ===")
print("ROC AUC:", auc)
print("Balanced Accuracy:", bal_acc)

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Infertile", "Fertile"],
            yticklabels=["Infertile", "Fertile"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =========================
# ROC CURVE
# =========================
fpr, tpr, _ = roc_curve(y_test, y_test_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# =========================
# FEATURE IMPORTANCE
# =========================
feature_names = []

# original intensities
for w in BEST_WAVELENGTHS:
    feature_names.append(f"I_{w}")

# ratios
for i in range(len(BEST_WAVELENGTHS)):
    for j in range(i + 1, len(BEST_WAVELENGTHS)):
        feature_names.append(f"{BEST_WAVELENGTHS[i]}/{BEST_WAVELENGTHS[j]}")

coeffs = model.coef_[0]

feat_df = pd.DataFrame({
    "feature": feature_names,
    "importance": coeffs
}).sort_values(by="importance", key=abs, ascending=False)

print("\nTop features:")
print(feat_df.head(10))

# Plot feature importance
plt.figure(figsize=(8, 5))
sns.barplot(data=feat_df.head(10), x="importance", y="feature")
plt.title("Top Feature Importances (LogReg)")
plt.show()
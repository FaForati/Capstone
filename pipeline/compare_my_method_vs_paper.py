import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from scipy.signal import savgol_filter

# =========================
# CONFIG
# =========================
DATA_CSV = r"C:\HSI_Data\DOI-10-13012-b2idb-8141497_v1\spectra and reference parameters.csv"#r"C:\HSI_Data\InHouseData\InHouse_fertility_data.csv"#

BEST_WAVELENGTHS = [530, 610, 670, 950]#[710, 750, 850, 870]#
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

spectral_cols = [col for col in df.columns if "nm" in col or col.replace('.', '', 1).isdigit()]
wavelengths = np.array([float(col.replace("nm", "")) for col in spectral_cols])

X_full = df[spectral_cols].values
y = df["label"].values

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.25, stratify=y, random_state=42
)

# =========================
# CLASS BALANCING (TRAIN ONLY)
# =========================
from imblearn.over_sampling import SMOTE

print("Before balancing:", np.bincount(y_train))

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After balancing:", np.bincount(y_train))

# =========================
# -------- My METHOD -----
# =========================

def gaussian_weights(wavelengths, center, fwhm):
    sigma = fwhm / 2.355
    w = np.exp(-0.5 * ((wavelengths - center)/sigma)**2)
    return w / np.sum(w)

def simulate_leds(X, wavelengths, centers):
    X_led = []
    for c in centers:
        w = gaussian_weights(wavelengths, c, FWHM)
        X_led.append(np.dot(X, w))
    return np.vstack(X_led).T

def add_ratios(X):
    feats = [X]
    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            feats.append((X[:, i] / (X[:, j]+1e-8)).reshape(-1,1))
    return np.hstack(feats)

# LED simulation
X_train_led = simulate_leds(X_train, wavelengths, BEST_WAVELENGTHS)
X_test_led = simulate_leds(X_test, wavelengths, BEST_WAVELENGTHS)

X_train_my = add_ratios(X_train_led)
X_test_my = add_ratios(X_test_led)

scaler = StandardScaler()
X_train_my = scaler.fit_transform(X_train_my)
X_test_my = scaler.transform(X_test_my)

my_model = LogisticRegression(max_iter=2000)
my_model.fit(X_train_my, y_train)

y_prob_my = my_model.predict_proba(X_test_my)[:,1]
y_pred_my = (y_prob_my >= 0.6).astype(int)

# =========================
# -------- PAPER METHOD ----
# =========================

# First derivative (Savitzky-Golay)
def first_derivative(X):
    return savgol_filter(X, window_length=11, polyorder=2, deriv=1)
def snv(X):
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
X_train_fd =  first_derivative(snv(X_train))
X_test_fd =  first_derivative(snv(X_test))
'''
scaler2 = StandardScaler()
X_train_fd = scaler2.fit_transform(X_train_fd)
X_test_fd = scaler2.transform(X_test_fd)
'''
models = {
    "CatBoost": CatBoostClassifier(verbose=0),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "SVM": SVC(probability=True),
    "RandomForest": RandomForestClassifier()
}

paper_results = {}

for name, model in models.items():
    model.fit(X_train_fd, y_train)
    y_prob = model.predict_proba(X_test_fd)[:,1]
    y_pred = model.predict(X_test_fd)

    paper_results[name] = {
        "auc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "y_prob": y_prob,
        "y_pred": y_pred
    }

# =========================
# METRICS FUNCTION
# =========================
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred)
    }

my_metrics = compute_metrics(y_test, y_pred_my, y_prob_my)

# =========================
# RESULTS TABLE
# =========================
rows = []

rows.append({"Method": "Proposed Method (4 LEDs)", **my_metrics})

for name, res in paper_results.items():
    rows.append({
        "Method": f"Paper - {name}",
        "AUC": res["auc"],
        "Accuracy": res["accuracy"],
        "Precision": res["precision"],
        "Recall": res["recall"],
        "F1": res["f1"]
    })

results_df = pd.DataFrame(rows)
print("\n=== COMPARISON TABLE ===")
print(results_df)

# =========================
# ROC CURVES
# =========================
plt.figure()

fpr, tpr, _ = roc_curve(y_test, y_prob_my)
plt.plot(fpr, tpr, label="Proposed Method")

for name, res in paper_results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    plt.plot(fpr, tpr, label=name)

plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title("ROC Comparison")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()

# =========================
# CONFUSION MATRICES
# =========================
def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(title)
    plt.show()

plot_cm(y_test, y_pred_my, "Proposed Method")

for name, res in paper_results.items():
    plot_cm(y_test, res["y_pred"], f"{name}")

# =========================
# BAR PLOT COMPARISON
# =========================
results_df.set_index("Method")[["Accuracy","F1","AUC"]].plot(kind="bar")
plt.title("Performance Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =========================
# FEATURE IMPORTANCE (MY MODEL)
# =========================
coef = my_model.coef_[0]

plt.figure()
plt.bar(range(len(coef)), np.abs(coef))
plt.title("Proposed Model Feature Importance")
plt.show()
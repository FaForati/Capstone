import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

# External models
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# =========================
# CONFIG
# =========================
DATA_CSV = r"C:\HSI_Data\DOI-10-13012-b2idb-8141497_v1\spectra and reference parameters.csv"

SELECTED_WAVELENGTHS = [520, 680, 940]  # EDIT THIS

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_CSV)

df["Sample ID"] = df["Sample ID"].str.replace("'", "", regex=False)

df["label"] = df["Fertility status"].map({
    "Fertile": 1,
    "Infertile": 0
})

df = df.dropna(subset=["label"])

# =========================
# EXTRACT SPECTRAL COLUMNS
# =========================
spectral_cols = [col for col in df.columns if "nm" in col or col.replace('.', '', 1).isdigit()]

col_to_wavelength = {}
for col in spectral_cols:
    try:
        col_to_wavelength[col] = float(col.replace("nm", ""))
    except:
        continue

# Select closest wavelengths
selected_cols = []
for w in SELECTED_WAVELENGTHS:
    closest = min(col_to_wavelength, key=lambda c: abs(col_to_wavelength[c] - w))
    selected_cols.append(closest)

print("Selected columns:", selected_cols)

X_base = df[selected_cols].values
y = df["label"].values

# =========================
# PREPROCESSING FUNCTIONS
# =========================
def baseline_correction(X):
    # subtract minimum per sample
    return X - np.min(X, axis=1, keepdims=True)

def msc(X):
    mean_spectrum = np.mean(X, axis=0)
    X_corr = np.zeros_like(X)

    for i in range(X.shape[0]):
        fit = np.polyfit(mean_spectrum, X[i], 1)
        X_corr[i] = (X[i] - fit[1]) / (fit[0] + 1e-8)

    return X_corr

def normalize(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

def snv(X):
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

def first_derivative(X):
    return np.diff(X, axis=1, prepend=X[:, [0]])

def add_ratios(X):
    features = [X]
    n = X.shape[1]

    for i in range(n):
        for j in range(i + 1, n):
            ratio = X[:, i] / (X[:, j] + 1e-8)
            features.append(ratio.reshape(-1, 1))

    return np.hstack(features)

# =========================
# FEATURE SETS
# =========================
feature_sets = {
    "raw": X_base,
    "BC": baseline_correction(X_base),
    "MSC": msc(X_base),
    "Norm": normalize(X_base),
    "SNV": snv(X_base),
    "SNV+derivative": first_derivative(snv(X_base)),
    "ratios": add_ratios(X_base),
    "SNV+ratios": add_ratios(snv(X_base)),
    "MSC+ratios": add_ratios(msc(X_base)),
}

# =========================
# MODELS
# =========================
models = {
    "LogReg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, class_weight="balanced"))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False
    ),
    "CatBoost": CatBoostClassifier(
        iterations=300,
        depth=4,
        learning_rate=0.05,
        verbose=0
    )
}

# =========================
# EVALUATION
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for feat_name, X_feat in feature_sets.items():
    print(f"\n=== Feature Set: {feat_name} ===")

    for model_name, model in models.items():
        try:
            scores = cross_val_score(model, X_feat, y, cv=cv, scoring="roc_auc")

            mean_score = scores.mean()
            std_score = scores.std()

            print(f"{model_name}: {mean_score:.4f} +/- {std_score:.4f}")

            results.append({
                "features": feat_name,
                "model": model_name,
                "roc_auc_mean": mean_score,
                "roc_auc_std": std_score
            })
        except Exception as e:
            print(f"{model_name} failed: {e}")

# =========================
# SAVE RESULTS
# =========================
results_df = pd.DataFrame(results).sort_values(by="roc_auc_mean", ascending=False)

results_df.to_csv("few_wavelength_advanced_results.csv", index=False)

print("\nTop results:")
print(results_df.head(10))

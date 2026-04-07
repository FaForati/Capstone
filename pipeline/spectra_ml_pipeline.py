import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# =========================
# FILE
# =========================
DATA_PATH = r"C:\HSI_Data\DOI-10-13012-b2idb-8141497_v1\spectra and reference parameters.csv"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

# =========================
# LABEL
# =========================
df["label"] = df["Fertility status"].map({
    "Fertile": 1,
    "Infertile": 0
})

df = df.dropna(subset=["label"])

# =========================
# SELECT ONLY WAVELENGTH COLUMNS
# =========================
# Columns that are numeric wavelengths (e.g., "374.14", "376.18", ...)
wavelength_cols = []

for col in df.columns:
    try:
        float(col)  # check if column name is numeric
        wavelength_cols.append(col)
    except:
        continue

# Sort columns by wavelength
wavelength_cols = sorted(wavelength_cols, key=lambda x: float(x))

X = df[wavelength_cols]
y = df["label"]

# Clean
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

print("Number of spectral features:", len(wavelength_cols))

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# MODEL 1: Logistic Regression
# =========================
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
])

lr_pipe.fit(X_train, y_train)

y_pred_lr = lr_pipe.predict(X_test)
y_prob_lr = lr_pipe.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression (Spectra Only) ===")
print(classification_report(y_test, y_pred_lr))
print("ROC AUC:", roc_auc_score(y_test, y_prob_lr))

# =========================
# FIND MISCLASSIFIED EGGS
# =========================
# Keep index alignment
X_test_df = X_test.copy()
X_test_df["true_label"] = y_test.values
X_test_df["pred_label"] = y_pred_lr
X_test_df["confidence"] = y_prob_lr

# Add file names back (important!)
# We must use original df indexing
test_indices = X_test.index
file_names = df.loc[test_indices, "Sample ID"].values
X_test_df["file_name"] = file_names

# Misclassified samples
misclassified = X_test_df[X_test_df["true_label"] != X_test_df["pred_label"]]

print("Number of misclassified eggs:", len(misclassified))

# Save only IDs + labels
misclassified_ids = misclassified[["file_name", "true_label", "pred_label", "confidence"]]
misclassified_ids.to_csv("misclassified_eggs_spectra_LR.csv", index=False)

print("Saved misclassified eggs to misclassified_eggs_spectra_LR.csv")


# =========================
# FIND highest confidence EGGS
# =========================
# Keep index alignment
X_test_df = X_test.copy()
X_test_df["true_label"] = y_test.values
X_test_df["pred_label"] = y_pred_lr
X_test_df["confidence"] = y_prob_lr

# Add file names back (important!)
# We must use original df indexing
test_indices = X_test.index
file_names = df.loc[test_indices, "Sample ID"].values
X_test_df["file_name"] = file_names

# high confidence samples
high_confidence = pd.concat([X_test_df.loc[X_test_df["confidence"].nlargest(10).index], X_test_df.loc[X_test_df["confidence"].nsmallest(10).index]])

print("Number of high confidence samples:", len(high_confidence))

# Save only IDs + labels
high_confidence_ids = high_confidence[["file_name", "true_label", "pred_label", "confidence"]]
high_confidence_ids.to_csv("high_confidence_eggs_spectra_LR.csv", index=False)

print("Saved high confidence samples to high_confidence_eggs_spectra_LR.csv")
# =========================
# MODEL 2: Random Forest
# =========================
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("\n=== Random Forest (Spectra Only) ===")
print(classification_report(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, y_prob_rf))

# =========================
# CROSS VALIDATION
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_cv = cross_val_score(lr_pipe, X, y, cv=cv, scoring="roc_auc")
rf_cv = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")

print("\n=== Cross Validation ROC AUC ===")
print("Logistic Regression:", lr_cv.mean(), "+/-", lr_cv.std())
print("Random Forest:", rf_cv.mean(), "+/-", rf_cv.std())
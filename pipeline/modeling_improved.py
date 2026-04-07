import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# =========================
# FILE PATHS
# =========================
FEATURES_CSV = r"C:\HSI_Data\DOI-10-13012-b2idb-8141497_v1\capstone\hyperspectral_features.csv"
GT_CSV = r"C:\HSI_Data\DOI-10-13012-b2idb-8141497_v1\spectra and reference parameters.csv"

# =========================
# LOAD + MERGE
# =========================
features_df = pd.read_csv(FEATURES_CSV)
gt_df = pd.read_csv(GT_CSV)

features_df["file_name"] = features_df["file_name"].str.replace(".mat", "", regex=False)
gt_df["Sample ID"] = gt_df["Sample ID"].str.replace("'", "", regex=False)

df = pd.merge(
    features_df,
    gt_df[["Sample ID", "Fertility status"]],
    left_on="file_name",
    right_on="Sample ID",
    how="inner"
)


df["label"] = df["Fertility status"].map({
    "Fertile": 1,
    "Infertile": 0
})

df = df.dropna(subset=["label"])

# =========================
# FEATURES
# =========================
X = df.drop(columns=[
    "file_name", "Sample ID", "Fertility status", "label"
])

# Rename wavelengths
wavelengths = np.linspace(374, 1015, 300)
X.columns = [
    f"{wavelengths[int(col.split('_')[1])]:.2f}nm"
    if col.startswith("band_") else col
    for col in X.columns
]

y = df["label"]

X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# STABILITY SELECTION (L1)
# =========================
n_runs = 50
selection_counts = np.zeros(X.shape[1])

for i in range(n_runs):
    # bootstrap sample
    idx = np.random.choice(len(X_train), len(X_train), replace=True)
    X_boot = X_train.iloc[idx]
    y_boot = y_train.iloc[idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_boot)

    model = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=0.05,   # stronger sparsity → fewer wavelengths
        max_iter=2000
    )

    model.fit(X_scaled, y_boot)

    selected = (model.coef_[0] != 0)
    selection_counts += selected

# =========================
# SELECT MOST STABLE WAVELENGTHS
# =========================
selection_freq = selection_counts / n_runs

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "frequency": selection_freq
}).sort_values(by="frequency", ascending=False)

print("\nTop stable wavelengths:")
print(feature_importance.head(20))

# Choose top N wavelengths
TOP_N = 8
selected_features = feature_importance.head(TOP_N)["feature"].values

print(f"\nSelected {TOP_N} wavelengths:")
print(selected_features)

# =========================
# TRAIN FINAL MODEL ON FEW WAVELENGTHS
# =========================
scaler = StandardScaler()
X_train_sel = scaler.fit_transform(X_train[selected_features])
X_test_sel = scaler.transform(X_test[selected_features])

final_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

final_model.fit(X_train_sel, y_train)

y_pred = final_model.predict(X_test_sel)
y_prob = final_model.predict_proba(X_test_sel)[:, 1]

print("\n=== FINAL MODEL (Few Wavelengths) ===")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# =========================
# SAVE SELECTED WAVELENGTHS
# =========================
feature_importance.to_csv("wavelength_stability.csv", index=False)

pd.DataFrame({
    "selected_wavelengths": selected_features
}).to_csv("selected_wavelengths.csv", index=False)

print("\nSaved wavelength selection results.")
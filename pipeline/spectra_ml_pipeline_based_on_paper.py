import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE

from scipy.signal import savgol_filter


# -----------------------------
# 1. Load data
# -----------------------------
FILE_PATH = r"C:\HSI_Data\DOI-10-13012-b2idb-8141497_v1\spectra and reference parameters.csv"

df = pd.read_csv(FILE_PATH)

# -----------------------------
# 2. Select ONLY spectral columns
# -----------------------------
# All columns after "Shell strength (N)" are wavelengths
start_col = "374.14"  # first wavelength column name (string)

spectral_cols = df.loc[:, start_col:].columns

X = df[spectral_cols].values
y = df["Fertility status"].values  # target


# Convert labels to binary if needed
# (adjust if already numeric)
if y.dtype == object:
    y = np.where(y == "Fertile", 1, 0)


# -----------------------------
# 3. Spectral preprocessing
# -----------------------------
def snv(input_data):
    """Standard Normal Variate"""
    return (input_data - np.mean(input_data, axis=1, keepdims=True)) / \
           np.std(input_data, axis=1, keepdims=True)


def first_derivative(data):
    """Savitzky-Golay first derivative"""
    return savgol_filter(data, window_length=11, polyorder=2, deriv=1, axis=1)


# Apply preprocessing (as in paper: FD performed best)
X_snv = snv(X)
X_fd = np.hstack((X_snv , first_derivative(X_snv)))
print(X_fd.shape)  # check SNV output


# -----------------------------
# 4. Train/Test split (73/27)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_fd, y, test_size=0.27, stratify=y, random_state=42
)


# -----------------------------
# 5. Handle imbalance (SMOTE)
# -----------------------------
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)


# -----------------------------
# 6. Define models
# -----------------------------
models = {
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss'
    ),

    "CatBoost": CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        verbose=0
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    ),

    "SVM": SVC(
        kernel='rbf',
        probability=True
    )
}


# -----------------------------
# 7. Train + Evaluate
# -----------------------------
results = []

for name, model in models.items():

    print(f"\n===== {name} =====")

    model.fit(X_train, y_train)

    # Test prediction
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Cross-validation (10-fold)
    cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')

    print(f"CV Accuracy: {cv_scores.mean():.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    results.append({
        "Model": name,
        "CV Accuracy": cv_scores.mean(),
        "Test Accuracy": acc,
        "Precision": prec,
        "F1 Score": f1
    })


# -----------------------------
# 8. Save results
# -----------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("model_results.csv", index=False)

print("\nSaved results to model_results.csv")
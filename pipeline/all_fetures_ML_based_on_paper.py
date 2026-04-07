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


# -----------------------------
# 1. Load data
# -----------------------------
FEATURE_FILE = r"C:\HSI_Data\DOI-10-13012-b2idb-8141497_v1\capstone\hyperspectral_features.csv"
LABEL_FILE = r"C:\HSI_Data\DOI-10-13012-b2idb-8141497_v1\spectra and reference parameters.csv"

df_features = pd.read_csv(FEATURE_FILE)
df_labels = pd.read_csv(LABEL_FILE)


# -----------------------------
# 2. Identify filename column
# -----------------------------
# 
filename_col = df_features.columns[-1]

df_features = df_features.rename(columns={filename_col: "Sample ID"})


# -----------------------------
# 3. Merge with labels
# -----------------------------
df_features["Sample ID"] = df_features["Sample ID"].str.replace(".mat", "", regex=False)
df_labels["Sample ID"] = df_labels["Sample ID"].str.replace("'", "", regex=False)

df = df_features.merge(
    df_labels,
    on="Sample ID",
    how="inner"
)
df
print(f"Merged dataset shape: {df.shape}")


# -----------------------------
# 4. Prepare X and y
# -----------------------------
X = df.drop(columns=["Sample ID", "Fertility status"]).values
y = df["Fertility status"].values

# Convert labels if needed
if y.dtype == object:
    y = np.where(y == "Fertile", 1, 0)


# -----------------------------
# 5. Train/Test split (73/27)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.27, stratify=y, random_state=42
)


# -----------------------------
# 6. SMOTE (handle imbalance)
# -----------------------------
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)


# -----------------------------
# 7. Define models
# -----------------------------
models = {
    "XGBoost": Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric='logloss'
        ))
    ]),

    "CatBoost": CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        verbose=0
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel='rbf', probability=True))
    ])
}


# -----------------------------
# 8. Train + Evaluate
# -----------------------------
results = []

for name, model in models.items():

    print(f"\n===== {name} =====")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

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
# 9. Save results
# -----------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("stat_features_model_results.csv", index=False)

print("\nSaved results to stat_features_model_results.csv")
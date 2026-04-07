import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, roc_auc_score

# =========================
# FILE PATHS
# =========================
FEATURES_CSV = r"C:\HSI_Data\DOI-10-13012-b2idb-8141497_v1\capstone\hyperspectral_features.csv"
GT_CSV = r"C:\HSI_Data\DOI-10-13012-b2idb-8141497_v1\spectra and reference parameters.csv"

# =========================
# LOAD DATA
# =========================
features_df = pd.read_csv(FEATURES_CSV)
gt_df = pd.read_csv(GT_CSV)

# =========================
# CLEAN + MERGE
# =========================
# Normalize filenames 
features_df["file_name"] = features_df["file_name"].str.replace(".mat", "", regex=False)
gt_df["Sample ID"] = gt_df["Sample ID"].str.replace("'", "", regex=False)

df = pd.merge(
    features_df,
    gt_df[["Sample ID", "Fertility status"]],
    left_on="file_name",
    right_on="Sample ID",
    how="inner"
)

print("Merged shape:", df.shape)

# =========================
# LABEL ENCODING
# =========================
df["label"] = df["Fertility status"].map({
    "Fertile": 1,
    "Infertile": 0
})

df = df.dropna(subset=["label"])

# =========================
# FEATURE / TARGET SPLIT
# =========================
X = df.drop(columns=[
    "file_name",
    "Sample ID",
    "Fertility status",
    "label"
])
X.columns = [f"{np.linspace(374, 1015, 300)[int(col.split('_')[1])]:.2f}nm_{'_'.join(col.split('_')[2:])}" if col.startswith("band_") else col for col in X.columns]

y = df["label"]

# Handle NaNs if any
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# Use ALL features for baseline model (no selection)
X_full = X.copy()
# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# FEATURE SELECTION (Top K)
# =========================
selector = SelectKBest(score_func=f_classif, k=50)
selector.fit(X_train, y_train)

selected_features = X.columns[selector.get_support()]
print("\nTop selected features:")
print(selected_features[:20])

# Save feature scores
feature_scores = pd.DataFrame({
    "feature": X.columns,
    "score": selector.scores_
}).sort_values(by="score", ascending=False)

feature_scores.to_csv("feature_importance_univariate.csv", index=False)

# Apply selection
X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)

# =========================
# MODEL 1: Random Forest
# =========================
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train_sel, y_train)

y_pred_rf = rf.predict(X_test_sel)
y_prob_rf = rf.predict_proba(X_test_sel)[:, 1]

print("\n=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, y_prob_rf))

# Feature importance (tree-based)
rf_importance = pd.DataFrame({
    "feature": selected_features,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)

rf_importance.to_csv("rf_feature_importance.csv", index=False)

# =========================
# MODEL 2: Logistic Regression
# =========================
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

pipe.fit(X_train_sel, y_train)

y_pred_lr = pipe.predict(X_test_sel)
y_prob_lr = pipe.predict_proba(X_test_sel)[:, 1]

print("\n=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr))
print("ROC AUC:", roc_auc_score(y_test, y_prob_lr))

# =========================
# CROSS VALIDATION
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_cv = cross_val_score(rf, selector.transform(X), y, cv=cv, scoring="roc_auc")
lr_cv = cross_val_score(pipe, selector.transform(X), y, cv=cv, scoring="roc_auc")

print("\n=== Cross Validation ROC AUC ===")
print("Random Forest:", rf_cv.mean(), "+/-", rf_cv.std())
print("Logistic Regression:", lr_cv.mean(), "+/-", lr_cv.std())


#baseline model using all features without selection
# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline (minimal preprocessing)
baseline_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

# Train
baseline_pipe.fit(X_train, y_train)

# Predict
y_pred_base = baseline_pipe.predict(X_test)
y_prob_base = baseline_pipe.predict_proba(X_test)[:, 1]

print("\n=== BASELINE (All Features - Logistic Regression) ===")
print(classification_report(y_test, y_pred_base))
print("ROC AUC:", roc_auc_score(y_test, y_prob_base))

# =========================
# FIND MISCLASSIFIED EGGS
# =========================
# Keep index alignment
X_test_df = X_test.copy()
X_test_df["true_label"] = y_test.values
X_test_df["pred_label"] = y_pred_base
X_test_df["confidence"] = y_prob_base

# Add file names back (important!)
# We must use original df indexing
test_indices = X_test.index
file_names = df.loc[test_indices, "file_name"].values
X_test_df["file_name"] = file_names

# Misclassified samples
misclassified = X_test_df[X_test_df["true_label"] != X_test_df["pred_label"]]

print("Number of misclassified eggs:", len(misclassified))

# Save only IDs + labels
misclassified_ids = misclassified[["file_name", "true_label", "pred_label", "confidence"]]
misclassified_ids.to_csv("misclassified_eggs_all_features.csv", index=False)

print("Saved misclassified eggs to misclassified_eggs_all_features.csv")


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
high_confidence_ids.to_csv("high_confidence_eggs_all_features_LR.csv", index=False)

print("Saved high confidence samples to high_confidence_eggs_all_features_LR.csv")

# =========================
# CROSS VALIDATION (baseline)
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baseline_cv = cross_val_score(
    baseline_pipe,
    X_full,
    y,
    cv=cv,
    scoring="roc_auc"
)

print("\nBaseline CV ROC AUC:", baseline_cv.mean(), "+/-", baseline_cv.std())

rf_base = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf_base.fit(X_train, y_train)

y_pred_rf_base = rf_base.predict(X_test)
y_prob_rf_base = rf_base.predict_proba(X_test)[:, 1]

print("\n=== BASELINE (All Features - Random Forest) ===")
print(classification_report(y_test, y_pred_rf_base))
print("ROC AUC:", roc_auc_score(y_test, y_prob_rf_base))
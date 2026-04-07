import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

# =========================
# FILES
# =========================
FEATURES_CSV = r"C:\HSI_Data\DOI-10-13012-b2idb-8141497_v1\capstone\hyperspectral_features.csv"
GT_CSV = r"C:\HSI_Data\DOI-10-13012-b2idb-8141497_v1\spectra and reference parameters.csv"
RF_IMPORTANCE_CSV = r"C:\HSI_Data\DOI-10-13012-b2idb-8141497_v1\capstone\rf_feature_importance.csv"

# =========================
# LOAD + MERGE (same logic)
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
# EXTRACT BAND FEATURES ONLY
# =========================
wavelengths = np.linspace(374, 1015, 300)
df.columns = [f"{np.linspace(374, 1015, 300)[int(col.split('_')[1])]:.2f}nm_{'_'.join(col.split('_')[2:])}" if col.startswith("band_") else col for col in df.columns]

band_cols = sorted([c for c in df.columns if "nm_mean" in c],
                   key=lambda x: float(x.split("nm")[0]))

X = df[band_cols]
print(X.columns[:5])  # check column names
y = df["label"]

# =========================
# 1. MEAN SPECTRUM
# =========================
mean_fertile = X[y == 1].mean()
mean_infertile = X[y == 0].mean()

plt.figure()
plt.plot(wavelengths, mean_fertile.values, label="Fertile")
plt.plot(wavelengths, mean_infertile.values, label="Infertile")
plt.title("Mean Spectrum Comparison")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.legend()
plt.savefig("mean_spectrum.png")
plt.close()

# =========================
# 2. DIFFERENCE SPECTRUM
# =========================
diff = mean_fertile - mean_infertile

plt.figure()
plt.plot(wavelengths, diff.values)
plt.title("Difference Spectrum (Fertile - Infertile)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Difference")
plt.axhline(0, linestyle="--")
plt.savefig("difference_spectrum.png")
plt.close()

# =========================
# 3. RANDOM FOREST IMPORTANCE
# =========================
rf_imp = pd.read_csv(RF_IMPORTANCE_CSV)

top_n = 30
top_features = rf_imp.head(top_n)
top_features["wavelength"] = top_features["feature"].str.extract(r'(\d+\.?\d*)nm')

plt.figure(figsize=(15, 12))
sns.barplot(
    x="importance",
    y="feature",
    data=top_features
)
plt.title("Top Feature Importance (Random Forest)")
plt.savefig("rf_top_features.png")
plt.close()


plt.figure(figsize=(10, 12))
sns.barplot(
    x="importance",
    y="wavelength",
    data=top_features
)
plt.title("Top wavelength Importance (Random Forest)")
plt.savefig("rf_top_wavelength.png")
plt.close()

# =========================
# 4. BAND IMPORTANCE MAP
# =========================
# Extract band index
rf_imp["wavelength"] = rf_imp["feature"].str.extract(r'(\d+\.?\d*)nm').astype(float)

band_importance = rf_imp.groupby("wavelength")["importance"].mean().sort_index()

plt.figure()
plt.plot(band_importance.index, band_importance.values)
plt.title("Importance Across Wavelengths")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Importance")
plt.savefig("band_importance.png")
plt.close()

# =========================
# 5. PCA VISUALIZATION
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure()
plt.scatter(
    X_pca[y == 0, 0],
    X_pca[y == 0, 1],
    label="Infertile",
    alpha=0.7
)
plt.scatter(
    X_pca[y == 1, 0],
    X_pca[y == 1, 1],
    label="Fertile",
    alpha=0.7
)

plt.title("PCA Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.savefig("pca_projection.png")
plt.close()

# =========================
# 6. CORRELATION HEATMAP (optional subset)
# =========================
subset_cols = band_cols[:50]  # avoid huge plot

corr = df[subset_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Feature Correlation Heatmap (subset)")
plt.savefig("correlation_heatmap.png")
plt.close()

# =========================
# 7. Most important wavelengths
# =========================

top_waves = band_importance.sort_values(ascending=False).head(10)

plt.figure()
plt.plot(band_importance.index, band_importance.values)
plt.scatter(top_waves.index, top_waves.values)
for w in top_waves.index:
    plt.text(w, band_importance.loc[w], f"{w:.0f}", fontsize=8)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Importance")
plt.title("Top Important Wavelengths")
plt.savefig("band_importance_annotated.png")
plt.close()


print("All visualizations saved.")
**Hyperspectral Imaging for Egg Fertility Detection**

This project develops and evaluates machine learning models for egg fertility classification using hyperspectral imaging (HSI) data. It explores both full-spectrum statistical features and low-cost LED-based sensing approaches to enable scalable real-world deployment.

├── clean_and_aggregate.py

├── all_fetures_ML_based_on_paper.py

├── optimize_led_wavelengths.py

├── final_led_model_evaluation.py

├── visualize_misclassified.py

├── selected_wavelengths.csv

├── wavelength_stability.csv

├── optimized_led_wavelengths*.csv

├── visualizations/

│   ├── *.png (model outputs, confusion matrices, grids, etc.)

├── outputs/

│   ├── generated images and results

**Pipeline Overview**
1. Feature Extraction from Hyperspectral Cubes

  Script: clean_and_aggregate.py

Loads .mat hyperspectral cubes
Extracts per-band statistical features:
mean, std, min, max, percentiles
skewness, kurtosis
Computes spectral-level features:
slope
total intensity
normalized spectrum stats

2. Full-Feature Machine Learning Models

Script: all_fetures_ML_based_on_paper.py

Merges features with fertility labels
Handles imbalance using SMOTE
Trains multiple models:
XGBoost
CatBoost
Random Forest
SVM
Performs:
cross-validation
test evaluation

3. LED-Based Feature Simulation (Low-Cost System)

Instead of using full spectra, this project simulates discrete LED measurements:

Gaussian weighting models LED bandwidth (FWHM)
Converts spectra → LED intensities
Adds ratio features between LEDs


4. LED Wavelength Optimization

Script: optimize_led_wavelengths.py

Searches for optimal LED combinations
Uses:
cross-validation
ROC AUC
balanced accuracy
Supports interruption + partial result saving


5. Final LED Model Evaluation

Script: compare_my_method_vs_paper.py

Uses selected wavelengths (e.g., [530, 610, 670, 950])
Performs:
train/validation/test split
threshold tuning
ROC + confusion matrix


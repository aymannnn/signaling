import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    RandomizedSearchCV,
    learning_curve,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, PoissonRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import train_test_split


OUTPUT_DIRECTORY = Path("machine_learning_model/")
df = pd.read_csv(OUTPUT_DIRECTORY / 'full_data_matrix.csv')

### PREPROCESSING ###

outcome_variable = 'reviews'

# note that the simulations are not independent, they're grouped by
# the result file prefix so we can do scenario-aware splitting
group_col = 'result_file_prefix'
groups = df[group_col]

# keep group column in dataframe for splitting but NOT as a feature
feature_df = df.drop(columns = [outcome_variable], errors = 'ignore').copy()

X = feature_df
y = df[outcome_variable].astype(float)

X = df.drop(columns = [outcome_variable])
y = df[outcome_variable]

# get train-test split

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

# 5-fold cross validation, grouped
cv = GroupKFold(n_splits=5)



fitted = estimator.fit(X_train, y_train)
y_pred = fitted.predict(X_test)

# metrics
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae = float(mean_absolute_error(y_test, y_pred))
r2 = float(r2_score(y_test, y_pred))

metrics = {"rmse": rmse, "mae": mae, "r2": r2, "best_cv_rmse": best_cv_rmse}

# ---- Save summary ----
summary_lines = [
    f"model_name: {model_name}",
    f"log_target: {log_target}",
    f"note: {spec.note}",
    f"n_train: {len(X_train)}",
    f"n_test:  {len(X_test)}",
    f"test_rmse: {rmse:.4f}",
    f"test_mae:  {mae:.4f}",
    f"test_r2:   {r2:.4f}",
]


from sklearn.linear_model import LinearRegression


# Regressor model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction result
y_pred_test = regressor.predict(X_test)     # predicted value of y_test
y_pred_train = regressor.predict(X_train)   # predicted value of y_train

regressor.score(X_train, y_train)  # R^2 on train set
regressor.score(X_test, y_test)  # R^2 on test set

# with an R^2 of 0.40 ...

ks = range(1, 41)
r2s, rmses = [], []

for k in ks:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=k))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2s.append(r2_score(y_test, pred))
    rmses.append(np.sqrt(mean_squared_error(y_test, pred)))

# Plot
plt.figure(figsize=(7, 5))
plt.plot(list(ks), r2s)
plt.xlabel("n_neighbors (k)")
plt.ylabel("R² on test set")
plt.title("Scaled KNN: test R² vs k")
plt.grid(alpha=0.2)
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(list(ks), rmses)
plt.xlabel("n_neighbors (k)")
plt.ylabel("RMSE on test set")
plt.title("Scaled KNN: test RMSE vs k")
plt.grid(alpha=0.2)
plt.show()

# Choose best k by RMSE
best_idx = int(np.argmin(rmses))
best_k = list(ks)[best_idx]
print("Best k (by test RMSE):", best_k, "RMSE:",
      rmses[best_idx], "R2:", r2s[best_idx])

# Fit final scaled model with best_k
knn_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=best_k))
])
knn_scaled.fit(X_train, y_train)
y_pred = knn_scaled.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("Final scaled KNN  RMSE:", rmse, " R2:", r2)

knn_scaled.fit(X_train, y_train)
y_pred = knn_scaled.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Scaled KNN  RMSE:", mse**0.5, " R2:", r2)

# Visualize

# --- 1) Predicted vs Actual (parity plot) ---
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
lims = [
    min(np.min(y_test), np.min(y_pred)),
    max(np.max(y_test), np.max(y_pred)),
]
plt.plot(lims, lims, 'k--', linewidth=1)
plt.xlim(lims)
plt.ylim(lims)
plt.xlabel("Actual reviews")
plt.ylabel("Predicted reviews")
plt.title(f"KNN parity plot (R²={r2:.3f}, RMSE={rmse:.3f})")
plt.grid(alpha=0.2)
plt.show()

# --- 2) Residuals vs Predicted ---
resid = y_test - y_pred
plt.figure(figsize=(7, 5))
plt.scatter(y_pred, resid, alpha=0.6)
plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.xlabel("Predicted reviews")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residuals vs Predicted")
plt.grid(alpha=0.2)
plt.show()

# --- 3) Residual distribution ---
plt.figure(figsize=(7, 5))
plt.hist(resid, bins=30)
plt.xlabel("Residual (actual - predicted)")
plt.ylabel("Count")
plt.title("Residual distribution")
plt.grid(alpha=0.2)
plt.show()

feature_names = X_test.columns if hasattr(X_test, "columns") else [
    f"x{i}" for i in range(X_test.shape[1])]

from sklearn.inspection import permutation_importance

# Use RMSE-based importance (drop in performance when feature is permuted)
perm = permutation_importance(
    knn_scaled,              
    X_test, y_test,
    n_repeats=30,
    random_state=0,
    scoring="neg_root_mean_squared_error",
)

imp = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)

print(imp)

# Plot features
plot_df = imp.head(len(imp)).iloc[::-1]  # reverse for barh
plt.figure(figsize=(8, 6))
plt.barh(plot_df["feature"], plot_df["importance_mean"],
         xerr=plot_df["importance_std"])
plt.xlabel("Importance (RMSE increase when permuted)")
plt.title(f"Permutation importance (top {len(imp)})")
plt.grid(alpha=0.2)
plt.show()

# save the model and use it for predictions later based on characteristics

import joblib

joblib.dump(knn_scaled, OUTPUT_DIRECTORY / 'knn_scaled_model.joblib')

# test the model on the NRMP data

nrmp_constants = pd.read_csv('NRMP/constants_nrmp.csv')
nrmp = {
    'n_programs': nrmp_constants['n_programs'],
    "simulated_positions": nrmp_constants['simulated_positions'],
    "n_applicants": nrmp_constants['n_applicants'],
    "interviews_per_spot": nrmp_constants['interviews_per_spot'],
    "max_applications": nrmp_constants['max_applications'],
    "program": nrmp_constants['result_file_prefix'],
    "signal_min": nrmp_constants['study_min_signal'],
    "signal_max": nrmp_constants['study_max_signal'], 
}

nrmp_df = pd.DataFrame(nrmp)

final_output = {
    # will be program : {signal: predicted reviews, ...}
}

final_output_dataframes = {}

for index, row in nrmp_df.iterrows():
    for signal in range(row['signal_min'], row['signal_max'] + 1):
        input_data = pd.DataFrame({
            'n_programs': row['n_programs'],
            "simulated_positions": row['simulated_positions'],
            "n_applicants": row['n_applicants'],
            "interviews_per_spot": row['interviews_per_spot'],
            "max_applications": row['max_applications'],
            "signal_value": [signal],
        })
        predicted_reviews = knn_scaled.predict(input_data)[0]
        if row['program'] not in final_output:
            final_output[row['program']] = {}
        final_output[row['program']][signal] = predicted_reviews    
    final_output_dataframes[row['program']] = pd.DataFrame.from_dict(final_output[row['program']], orient='index', columns=['predicted_reviews'])

optimal_signals = pd.DataFrame(columns = ['Program', 'Optimal Predicted Signal', 'Predicted Reviews at Optimal Signal'])

for program in final_output_dataframes.keys():
    df = final_output_dataframes[program]
    optimal_signal = df['predicted_reviews'].idxmax()
    optimal_reviews = df['predicted_reviews'].max()
    optimal_signals = pd.concat([optimal_signals, pd.DataFrame({
        'Program': [program],
        'Optimal Predicted Signal': [optimal_signal],
        'Predicted Reviews at Optimal Signal': [optimal_reviews]})], ignore_index=True)
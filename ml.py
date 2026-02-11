
# =============================================================================
# Interactive model evaluation (run cell-by-cell)
# Models:
#   - kNN regressor
#   - Random Forest regressor
#   - XGBoost regressor
#
# Focus:
#   - Group-aware CV (GroupKFold) + group-aware holdout split
#   - Grid search with multiple metrics (fit + decision metrics)
#   - Interpretable plots + export to disk (light theme)
#   - SHAPLEY plots for most predictive features
#       * XGBoost: SHAP contributions via Booster.predict(pred_contribs=True) (no shap dependency)
#       * RF / kNN: uses `shap` if installed (optional). Otherwise permutation importance is still exported.
#
# Notes:
#   - result_file_prefix is ONLY for grouping; it is not used as a feature.
#   - signal_value is used as a feature AND as the decision variable for argmin metrics.
# =============================================================================

# %% Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import xgboost as xgb

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.inspection import permutation_importance


# %% Config 
DATA_PATH = "machine_learning_model/full_data_matrix.csv"
OUT_DIR = "machine_learning_model/ml_outputs_interactive"

target_col = "reviews"
group_col = "result_file_prefix"
signal_col = "signal_value"

feature_cols = [
    "n_programs",
    "n_positions",
    "n_applicants",
    "interviews_per_spot",
    "max_applications",
    "signal_value",
]

# Tradeoff weight: overall fit (MAE) vs decision accuracy (min-signal error)
w_signal = 50.0

# CV settings
n_splits = 5
random_state = 42

os.makedirs(OUT_DIR, exist_ok=True)


# %% Light plot theme (cool, readable)
def set_light_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#CCCCCC",
        "grid.alpha": 0.25,
        "axes.titleweight": "semibold",
        "font.size": 11,
        "figure.dpi": 140,
    })

def savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# %% Load data
df = pd.read_csv(DATA_PATH)

# Quick sanity checks
required = set([target_col, group_col, signal_col] + feature_cols)
missing = sorted(list(required - set(df.columns)))
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

y = df[target_col]
X = df.drop(columns=[target_col])
groups = X[group_col]


# %% Group-aware holdout split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
groups_train, groups_test = groups.iloc[train_idx].copy(), groups.iloc[test_idx].copy()

print("Train rows:", len(X_train), "| Test rows:", len(X_test))
print("Train groups:", groups_train.nunique(), "| Test groups:", groups_test.nunique())


# %% Groupwise decision metrics (argmin over signal_value within each group)
def group_min_metrics(X_subset, y_true, y_pred, group_col=group_col, signal_col=signal_col):
    """
    Per-group metrics for "choose the signal_value that minimizes reviews".

      - abs error of argmin(signal_value): predicted vs true (within each group)
      - abs error of min reviews value: predicted min vs true min (within each group)
      - regret: true reviews at predicted-optimal signal - true minimum reviews

    Returns averages across groups (mean + median) + n_groups.
    """
    g = X_subset[group_col].to_numpy()
    s = X_subset[signal_col].to_numpy()
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    order = np.argsort(g, kind="mergesort")
    g, s, yt, yp = g[order], s[order], yt[order], yp[order]

    uniq, starts = np.unique(g, return_index=True)
    starts = list(starts) + [len(g)]

    signal_err = []
    min_reviews_err = []
    regret = []

    for i in range(len(uniq)):
        a, b = starts[i], starts[i + 1]
        yt_g, yp_g, s_g = yt[a:b], yp[a:b], s[a:b]

        true_min_pos = int(np.argmin(yt_g))
        pred_min_pos = int(np.argmin(yp_g))

        true_min_signal = float(s_g[true_min_pos])
        pred_min_signal = float(s_g[pred_min_pos])

        true_min_reviews = float(yt_g[true_min_pos])
        pred_min_reviews = float(yp_g[pred_min_pos])

        signal_err.append(abs(pred_min_signal - true_min_signal))
        min_reviews_err.append(abs(pred_min_reviews - true_min_reviews))
        regret.append(float(yt_g[pred_min_pos] - true_min_reviews))

    return {
        "mean_abs_error_min_signal": float(np.mean(signal_err)),
        "median_abs_error_min_signal": float(np.median(signal_err)),
        "mean_abs_error_min_reviews": float(np.mean(min_reviews_err)),
        "median_abs_error_min_reviews": float(np.median(min_reviews_err)),
        "mean_regret_reviews": float(np.mean(regret)),
        "median_regret_reviews": float(np.median(regret)),
        "n_groups": int(len(uniq)),
    }


# %% Scorers for GridSearchCV (higher is better)
def combined_objective(estimator, X_val, y_val, w_signal=w_signal):
    yp = estimator.predict(X_val)
    mae = mean_absolute_error(y_val, yp)
    ms = group_min_metrics(X_val, y_val, yp)["mean_abs_error_min_signal"]
    return -(mae + w_signal * ms)

def min_signal_scorer(estimator, X_val, y_val):
    yp = estimator.predict(X_val)
    ms = group_min_metrics(X_val, y_val, yp)["mean_abs_error_min_signal"]
    return -ms

def regret_scorer(estimator, X_val, y_val):
    yp = estimator.predict(X_val)
    rg = group_min_metrics(X_val, y_val, yp)["mean_regret_reviews"]
    return -rg


scoring = {
    "neg_mae": "neg_mean_absolute_error",
    "neg_rmse": make_scorer(lambda yt, yp: -np.sqrt(mean_squared_error(yt, yp))),
    "r2": "r2",
    "min_signal": min_signal_scorer,
    "neg_regret": regret_scorer,
    "combined": combined_objective,
}


# %% Preprocessor
def make_preprocessor(scale_numeric: bool):
    if scale_numeric:
        return ColumnTransformer([("num", StandardScaler(), feature_cols)], remainder="drop")
    return ColumnTransformer([("num", "passthrough", feature_cols)], remainder="drop")


# %% Model pipelines + grids
knn_pipe = Pipeline([
    ("prep", make_preprocessor(scale_numeric=True)),
    ("model", KNeighborsRegressor()),
])
knn_grid = {
    "model__n_neighbors": [3, 5, 9, 15, 25, 35],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2],
    "model__leaf_size": [15, 30, 60],
}

rf_pipe = Pipeline([
    ("prep", make_preprocessor(scale_numeric=False)),
    ("model", RandomForestRegressor(
        random_state=random_state,
        n_jobs=-1
    )),
])
rf_grid = {
    "model__n_estimators": [300, 600],
    "model__max_depth": [None, 6, 12, 20],
    "model__min_samples_leaf": [1, 2, 5, 10],
    "model__max_features": ["sqrt", 0.6, 1.0],
}

xgb_pipe = Pipeline([
    ("prep", make_preprocessor(scale_numeric=False)),
    ("model", XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )),
])
xgb_grid = {
    "model__n_estimators": [400, 900],
    "model__max_depth": [2, 3, 5, 7],
    "model__learning_rate": [0.03, 0.07, 0.12],
    "model__subsample": [0.7, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.9, 1.0],
    "model__reg_alpha": [0.0, 1e-3, 1e-2],
    "model__reg_lambda": [1.0, 5.0, 10.0],
}

models = {
    "knn": (knn_pipe, knn_grid),
    "random_forest": (rf_pipe, rf_grid),
    "xgboost": (xgb_pipe, xgb_grid),
}


# %% Helpers: evaluation + plotting
def evaluate_holdout(model, X_test, y_test):
    yp = model.predict(X_test)
    out = {
        "mae": float(mean_absolute_error(y_test, yp)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, yp))),
        "r2": float(r2_score(y_test, yp)),
    }
    out.update(group_min_metrics(X_test, y_test, yp))
    return out, yp


def plot_pred_vs_true(y_true, y_pred, title, out_path):
    set_light_plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(y_true, y_pred, s=10, alpha=0.35)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("True reviews")
    ax.set_ylabel("Predicted reviews")
    savefig(fig, out_path)


def plot_residuals(y_true, y_pred, title, out_path):
    set_light_plot_style()
    resid = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(y_pred, resid, s=10, alpha=0.35)
    ax.axhline(0.0, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Predicted reviews")
    ax.set_ylabel("Residual (pred - true)")
    savefig(fig, out_path)


def plot_group_curves_vs_signal(X_subset, y_true, y_pred, title, out_path, max_groups=12):
    """
    For a sample of groups, plot true vs predicted reviews across signal_value,
    and mark true argmin and predicted argmin (but plotted on true y for interpretability).
    """
    set_light_plot_style()
    rng = np.random.default_rng(random_state)

    dfp = X_subset[[group_col, signal_col]].copy()
    dfp["y_true"] = np.asarray(y_true)
    dfp["y_pred"] = np.asarray(y_pred)

    all_groups = dfp[group_col].unique()
    if len(all_groups) == 0:
        return
    chosen = all_groups if len(all_groups) <= max_groups else rng.choice(all_groups, size=max_groups, replace=False)

    n = len(chosen)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15.5, 4.0 * rows), squeeze=False)
    axes = axes.flatten()

    for i, g in enumerate(chosen):
        ax = axes[i]
        gdf = dfp[dfp[group_col] == g].sort_values(signal_col)

        ax.plot(gdf[signal_col], gdf["y_true"], marker="o", linewidth=1.6, markersize=3, label="true")
        ax.plot(gdf[signal_col], gdf["y_pred"], marker="o", linewidth=1.6, markersize=3, label="pred")

        true_idx = int(np.argmin(gdf["y_true"].to_numpy()))
        pred_idx = int(np.argmin(gdf["y_pred"].to_numpy()))

        ax.scatter([gdf[signal_col].iloc[true_idx]], [gdf["y_true"].iloc[true_idx]], s=80, marker="*", label="true argmin")
        ax.scatter([gdf[signal_col].iloc[pred_idx]], [gdf["y_true"].iloc[pred_idx]], s=80, marker="*", label="pred argmin (true y)")

        ax.set_title(str(g))
        ax.set_xlabel(signal_col)
        ax.set_ylabel("reviews")
        ax.legend(fontsize=9)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, y=1.02, fontsize=14, fontweight="semibold")
    savefig(fig, out_path)


def plot_cv_scores(gs, metric, title, out_path, top_k=25):
    set_light_plot_style()
    res = pd.DataFrame(gs.cv_results_).sort_values(f"mean_test_{metric}", ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(np.arange(len(res)), res[f"mean_test_{metric}"].to_numpy(), marker="o")
    ax.set_title(title)
    ax.set_xlabel(f"Top {top_k} configs (ranked)")
    ax.set_ylabel(f"mean_test_{metric} (higher is better)")
    savefig(fig, out_path)


def plot_permutation_importance(model, X_eval, y_eval, title, out_path, n_repeats=10):
    """
    IMPORTANT: permutation_importance permutes columns of X as *passed in*.
    Since our pipeline only uses `feature_cols`, pass ONLY those columns,
    otherwise you'll get a length mismatch (because X_eval also contains group ids).
    """
    set_light_plot_style()

    X_perm = X_eval[feature_cols].copy()

    r = permutation_importance(
        model, X_perm, y_eval,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )

    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(imp["feature"][::-1], imp["importance_mean"]
            [::-1], xerr=imp["importance_std"][::-1])
    ax.set_title(title)
    ax.set_xlabel("Permutation importance (Δ neg-MAE)")
    savefig(fig, out_path)


# %% SHAPLEY (XGBoost: no-dependency SHAP contributions)
def plot_xgb_shap_style(best_pipeline, X_eval, title_prefix, out_dir, max_rows=5000):
    """
    Computes SHAP-style feature contributions from XGBoost booster:
      booster.predict(DMatrix, pred_contribs=True)
    Exports:
      - bar of mean(|contribution|)
      - beeswarm-like plot of contribution distribution for top features
    """
    os.makedirs(out_dir, exist_ok=True)
    set_light_plot_style()

    rng = np.random.default_rng(random_state)
    if len(X_eval) > max_rows:
        idx = rng.choice(len(X_eval), size=max_rows, replace=False)
        Xs = X_eval.iloc[idx]
    else:
        Xs = X_eval

    prep = best_pipeline.named_steps["prep"]
    model = best_pipeline.named_steps["model"]

    X_trans = prep.transform(Xs)
    dm = xgb.DMatrix(X_trans, feature_names=feature_cols)
    booster = model.get_booster()

    contrib = np.asarray(booster.predict(dm, pred_contribs=True))
    shap_vals = contrib[:, :-1]  # last col is bias

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    imp = pd.DataFrame({"feature": feature_cols, "mean_abs_contrib": mean_abs}).sort_values("mean_abs_contrib", ascending=False)

    # Bar
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(imp["feature"][::-1], imp["mean_abs_contrib"][::-1])
    ax.set_title(f"{title_prefix}: mean(|contribution|)")
    ax.set_xlabel("Mean absolute contribution (SHAP-style)")
    savefig(fig, os.path.join(out_dir, "xgb_shap_bar.png"))

    # Beeswarm-ish for top 8
    top = imp.head(8)["feature"].tolist()
    top_idx = [feature_cols.index(f) for f in top]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    y_base = np.arange(len(top))

    # Use last scatter for colorbar handle
    sc = None
    for j, f_idx in enumerate(top_idx):
        x = shap_vals[:, f_idx]
        fv = np.asarray(Xs[top[j]].to_numpy())
        yj = y_base[j] + (rng.random(len(x)) - 0.5) * 0.35
        sc = ax.scatter(x, yj, s=10, alpha=0.35, c=fv)

    ax.axvline(0.0, linewidth=2)
    ax.set_yticks(y_base)
    ax.set_yticklabels(top)
    ax.set_title(f"{title_prefix}: top features (contribution)")
    ax.set_xlabel("Contribution (SHAP-style)")
    ax.set_ylabel("Feature")
    if sc is not None:
        fig.colorbar(sc, ax=ax, label="Feature value")
    savefig(fig, os.path.join(out_dir, "xgb_shap_beeswarm.png"))


# %% SHAP package (RF + kNN). Skips quietly if not installed.
def _try_import_shap():
    try:
        import shap  # type: ignore
        return shap
    except Exception:
        return None

def plot_shap_if_available(best_pipeline, X_eval, title_prefix, out_dir, max_rows=4000):
    """
    If shap is installed, export:
      - shap_bar.png
      - shap_beeswarm.png

    Note: Kernel SHAP for kNN may be slow; keep max_rows small.
    """
    shap = _try_import_shap()
    if shap is None:
        print("shap not installed -> skipping SHAP plots for this model.")
        return

    os.makedirs(out_dir, exist_ok=True)
    set_light_plot_style()

    rng = np.random.default_rng(random_state)
    if len(X_eval) > max_rows:
        idx = rng.choice(len(X_eval), size=max_rows, replace=False)
        Xs = X_eval.iloc[idx]
    else:
        Xs = X_eval

    prep = best_pipeline.named_steps["prep"]
    model = best_pipeline.named_steps["model"]
    X_trans = prep.transform(Xs)

    try:
        explainer = shap.Explainer(model, X_trans, feature_names=feature_cols)
        sv = explainer(X_trans)
    except Exception as e:
        print("SHAP explainer failed:", repr(e))
        return

    plt.figure(figsize=(8.2, 5.2))
    shap.plots.bar(sv, show=False, max_display=12)
    plt.title(f"{title_prefix} (SHAP bar)")
    savefig(plt.gcf(), os.path.join(out_dir, "shap_bar.png"))

    plt.figure(figsize=(8.2, 5.2))
    shap.plots.beeswarm(sv, show=False, max_display=12)
    plt.title(f"{title_prefix} (SHAP beeswarm)")
    savefig(plt.gcf(), os.path.join(out_dir, "shap_beeswarm.png"))


# %% Fit one model (interactive-friendly)
def run_one_model(model_name):
    pipe, grid = models[model_name]

    cv = GroupKFold(n_splits=n_splits)
    gs = GridSearchCV(
        pipe,
        param_grid=grid,
        scoring=scoring,
        refit="combined",   # refit best by combined objective
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    gs.fit(X_train, y_train, groups=groups_train)
    best_model = gs.best_estimator_

    # Save CV results
    model_out = os.path.join(OUT_DIR, model_name)
    os.makedirs(model_out, exist_ok=True)
    pd.DataFrame(gs.cv_results_).to_csv(os.path.join(model_out, "cv_results.csv"), index=False)

    # Holdout evaluation + plots
    metrics, y_pred = evaluate_holdout(best_model, X_test, y_test)

    plot_pred_vs_true(
        y_test.to_numpy(), y_pred,
        title=f"{model_name}: Predicted vs True (holdout)",
        out_path=os.path.join(model_out, "pred_vs_true.png"),
    )
    plot_residuals(
        y_test.to_numpy(), y_pred,
        title=f"{model_name}: Residuals (holdout)",
        out_path=os.path.join(model_out, "residuals.png"),
    )
    plot_group_curves_vs_signal(
        X_test, y_test, y_pred,
        title=f"{model_name}: True vs Pred curves by group (holdout sample)",
        out_path=os.path.join(model_out, "group_curves.png"),
    )
    plot_cv_scores(
        gs, metric="combined",
        title=f"{model_name}: Top CV scores (combined objective)",
        out_path=os.path.join(model_out, "cv_top_combined.png"),
    )

    # Permutation importance (works for all)
    plot_permutation_importance(
        best_model, X_test, y_test,
        title=f"{model_name}: Permutation importance on holdout",
        out_path=os.path.join(model_out, "perm_importance.png"),
    )

    # SHAPLEY plots
    if model_name == "xgboost":
        plot_xgb_shap_style(
            best_model, X_test,
            title_prefix="XGBoost feature contributions",
            out_dir=os.path.join(model_out, "shap"),
        )
    else:
        plot_shap_if_available(
            best_model, X_test,
            title_prefix=f"{model_name} feature attributions",
            out_dir=os.path.join(model_out, "shap"),
        )

    return gs, best_model, metrics


# %% Run models (uncomment and run line-by-line)
#gs_knn, best_knn, metrics_knn = run_one_model("knn")
#gs_rf, best_rf, metrics_rf = run_one_model("random_forest")
gs_xgb, best_xgb, metrics_xgb = run_one_model("xgboost")

# %% Collect / save summary (after running models above)
summary = pd.DataFrame([
    {"model": "knn", **metrics_knn},
    {"model": "random_forest", **metrics_rf},
    {"model": "xgboost", **metrics_xgb},
])
summary.to_csv(os.path.join(OUT_DIR, "model_summary.csv"), index=False)
summary

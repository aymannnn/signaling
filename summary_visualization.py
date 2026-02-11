from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="white")

# ----------------------------
# I/O + setup
# ----------------------------
out_dir = Path("figures")
out_dir.mkdir(parents=True, exist_ok=True)

base_case = pd.read_csv("results/base_case.csv")
nrmp_local = pd.read_csv("results/nrmp_local_analysis_results.csv")
df = pd.concat([base_case, nrmp_local], ignore_index=True)

UMAP_VARIABLES = {
    'n_neighbors': 50,
    'min_dist': 0.01,
    'metric': 'euclidean'
}

dependent_variables = [
    "n_programs",
    "n_positions",
    "n_applicants",
    "interviews_per_spot",
    "max_applications",
]

dependent_variables_names = [
    "# of Programs",
    "# of Positions",
    "# of Applicants",
    "Interviews per Spot",
    "# of Applications per Applicant",
]

# Graph these specific points on pairwise plots and (if supported) UMAPs.
specific_points = {
    "General Surgery (Categorical)": {
        "n_programs": 377,
        "n_positions": 1778,
        "n_applicants": 3305,
        "interviews_per_spot": 12,
        "max_applications": 40,
    },
    "Dermatology": {
        "n_programs": 14,
        "n_positions": 30,
        "n_applicants": 299,
        "interviews_per_spot": 12,
        "max_applications": 14,
    },
    "Internal Medicine (Categorical)": {
        "n_programs": 758,
        "n_positions": 10941,
        "n_applicants": 17131,
        "interviews_per_spot": 12,
        "max_applications": 40,
    },
}

pretty_map = dict(zip(dependent_variables, dependent_variables_names))

# Convert specific points to a DataFrame in "pretty" column-space
points_raw = pd.DataFrame.from_dict(specific_points, orient="index")
points_pretty = points_raw.rename(columns=pretty_map)

# Style for specialty points (small hollow circles with distinct edge colors)
point_style = {
    "General Surgery (Categorical)": dict(marker="o", s=28, facecolors="none", edgecolors="black", linewidths=0.9, zorder=10),
    "Dermatology": dict(marker="o", s=28, facecolors="none", edgecolors="#e69f00", linewidths=0.9, zorder=10),
    "Internal Medicine (Categorical)": dict(marker="o", s=28, facecolors="none", edgecolors="#009e73", linewidths=0.9, zorder=10),
}

point_short = {
    "General Surgery (Categorical)": "GS",
    "Dermatology": "D",
    "Internal Medicine (Categorical)": "IM",
}


def _specialty_handles():
    handles = []
    for name, style in point_style.items():
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor=style.get("edgecolors", "black"),
                markeredgewidth=style.get("linewidths", 1.0),
                markersize=6,
                label=name,
            )
        )
    return handles


def add_specific_points_to_pairgrid(pairgrid, points_df_pretty, annotate=True):
    """Overlay the specialty points on every off-diagonal axis in a seaborn PairGrid."""
    x_vars = list(pairgrid.x_vars)
    y_vars = list(pairgrid.y_vars)

    for i, yvar in enumerate(y_vars):
        for j, xvar in enumerate(x_vars):
            ax = pairgrid.axes[i, j]
            if i == j:
                continue

            for name, row in points_df_pretty.iterrows():
                if xvar not in row.index or yvar not in row.index:
                    continue
                if pd.isna(row[xvar]) or pd.isna(row[yvar]):
                    continue

                ax.scatter(row[xvar], row[yvar], **point_style.get(name, {}))

                if annotate:
                    ax.annotate(
                        point_short.get(name, name),
                        (row[xvar], row[yvar]),
                        textcoords="offset points",
                        xytext=(3, 3),
                        ha="left",
                        va="bottom",
                        fontsize=6,
                        zorder=11,
                    )


def project_points_to_umap(reducer, scaler, points_df_pretty):
    """Project specialty points into an existing UMAP embedding, if transform is available."""
    Xp = points_df_pretty[dependent_variables_names].astype(float).to_numpy()
    Xp_scaled = scaler.transform(Xp)

    if hasattr(reducer, "transform"):
        return reducer.transform(Xp_scaled)

    # Fallback: fit_transform on concatenated data (will alter the embedding); used only if transform is unavailable.
    return None


# ----------------------------
# CONTINUOUS: Optimal Signal
# ----------------------------
df["Optimal Signal"] = pd.to_numeric(df["best_signal_rpp"], errors="coerce")

X = df[dependent_variables].apply(pd.to_numeric, errors="coerce")
mask_cont = X.notna().all(axis=1) & df["Optimal Signal"].notna()
df_cont = df.loc[mask_cont].copy()
X_cont = X.loc[mask_cont].copy()
signal_cont = df_cont["Optimal Signal"]

# Apply pretty mapping for plotting (continuous too)
df_cont_plot = df_cont.rename(columns=pretty_map)
X_cont_plot = X_cont.rename(columns=pretty_map)


def offdiag_scatter_continuous(x, y, **kwargs):
    s = df_cont_plot.loc[x.index, "Optimal Signal"]
    sns.scatterplot(
        x=x,
        y=y,
        hue=s,
        palette="cool",
        legend=False,
        alpha=0.6,
        s=9,
        linewidth=0,
        edgecolor=None,
    )


# PairGrid (continuous)
g = sns.PairGrid(df_cont_plot, vars=dependent_variables_names, diag_sharey=False)
g.map_offdiag(offdiag_scatter_continuous)
g.map_diag(sns.kdeplot, fill=True, alpha=0.3)

# Overlay specialty points
add_specific_points_to_pairgrid(g, points_pretty, annotate=True)

norm = plt.Normalize(signal_cont.min(), signal_cont.max())
sm = plt.cm.ScalarMappable(cmap="cool", norm=norm)
sm.set_array([])
g.fig.colorbar(sm, ax=g.axes, label="Optimal Signal", shrink=0.8)

# Specialty legend
handles = _specialty_handles()
g.fig.legend(
    handles=handles,
    title="Specialty points",
    loc="upper right",
    bbox_to_anchor=(1.02, 1.0),
    frameon=True,
    fontsize=8,
    title_fontsize=9,
)

g.fig.suptitle("Pairplot of Model Variables Colored by Optimal Signal", y=1.02)
g.fig.savefig(out_dir / "pairplot_optimal_signal.png", dpi=300, bbox_inches="tight")
plt.close(g.fig)

# UMAP (continuous) — compute once and reuse in individual + panels
scaler_cont = StandardScaler().fit(X_cont_plot[dependent_variables_names])
Xc_scaled = scaler_cont.transform(X_cont_plot[dependent_variables_names])

# UMAP VARIABLES
reducer_cont = umap.UMAP(
    n_neighbors=UMAP_VARIABLES['n_neighbors'], 
    min_dist=UMAP_VARIABLES['min_dist'], 
    metric=UMAP_VARIABLES['metric'], 
    random_state=42)
emb_cont = reducer_cont.fit_transform(Xc_scaled)

emb_points_cont = project_points_to_umap(reducer_cont, scaler_cont, points_pretty)

# Individual continuous UMAP
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(
    x=emb_cont[:, 0],
    y=emb_cont[:, 1],
    hue=signal_cont,
    palette="cool",
    legend=False,
    s=18,
    alpha=0.6,
    linewidth=0,
    ax=ax,
)

# Overlay specialty points (if UMAP transform supported)
if emb_points_cont is not None:
    for idx, name in enumerate(points_pretty.index):
        ax.scatter(emb_points_cont[idx, 0], emb_points_cont[idx, 1], **point_style.get(name, {}))
        ax.annotate(
            point_short.get(name, name),
            (emb_points_cont[idx, 0], emb_points_cont[idx, 1]),
            textcoords="offset points",
            xytext=(4, 4),
            ha="left",
            va="bottom",
            fontsize=8,
            zorder=11,
        )

sm2 = plt.cm.ScalarMappable(cmap="cool", norm=plt.Normalize(signal_cont.min(), signal_cont.max()))
sm2.set_array([])
fig.colorbar(sm2, ax=ax, label="Optimal Signal", shrink=0.9)
ax.set_aspect("equal", "datalim")
ax.set_title("UMAP Projection (colored by Optimal Signal)")

# Add specialty legend (kept small)
ax.legend(handles=_specialty_handles(), title="Specialty points", loc="best", fontsize=8, title_fontsize=9)

fig.tight_layout()
fig.savefig(out_dir / "umap_optimal_signal.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ----------------------------
# BINARY: No Signal vs (+) Signal Benefit
# ----------------------------
df["best_signal_rpp_num"] = pd.to_numeric(df["best_signal_rpp"], errors="coerce")
df["Signal_bin"] = np.where(df["best_signal_rpp_num"] > 0, "(+) Signal Benefit", "No Signal Benefit")
df["Signal_bin"] = pd.Categorical(
    df["Signal_bin"],
    categories=["No Signal Benefit", "(+) Signal Benefit"],
    ordered=True,
)

X = df[dependent_variables].apply(pd.to_numeric, errors="coerce")
mask_bin = X.notna().all(axis=1) & df["Signal_bin"].notna() & df["best_signal_rpp_num"].notna()
df_bin = df.loc[mask_bin].copy()
X_bin = X.loc[mask_bin].copy()

# Apply pretty mapping for plotting (binary too)
df_bin_plot = df_bin.rename(columns=pretty_map)
X_bin_plot = X_bin.rename(columns=pretty_map)

# "cool" binary palette (discrete samples of same colormap)
cmap = plt.get_cmap("cool")
bin_palette = {
    "No Signal Benefit": cmap(0.15),
    "(+) Signal Benefit": cmap(0.85),
}

# PairGrid (binary)
g = sns.PairGrid(
    df_bin_plot,
    vars=dependent_variables_names,
    hue="Signal_bin",
    palette=bin_palette,
    diag_sharey=False,
)
g.map_offdiag(sns.scatterplot, alpha=0.6, s=9, linewidth=0, edgecolor=None)
g.map_diag(sns.kdeplot, fill=True, alpha=0.3)
g.add_legend(title="Signal")

# Overlay specialty points
add_specific_points_to_pairgrid(g, points_pretty, annotate=True)

# Specialty legend (separate from the hue legend)
g.fig.legend(
    handles=_specialty_handles(),
    title="Specialty points",
    loc="upper right",
    bbox_to_anchor=(1.02, 0.92),
    frameon=True,
    fontsize=8,
    title_fontsize=9,
)

g.fig.suptitle("Pairplot of Model Variables Colored by Signal (Binary)", y=1.02)
g.fig.savefig(out_dir / "pairplot_signal_binary.png", dpi=300, bbox_inches="tight")
plt.close(g.fig)

# UMAP (binary) — compute once and reuse in individual + panels
scaler_bin = StandardScaler().fit(X_bin_plot[dependent_variables_names])
Xb_scaled = scaler_bin.transform(X_bin_plot[dependent_variables_names])
reducer_bin = umap.UMAP(
    n_neighbors=UMAP_VARIABLES['n_neighbors'], 
    min_dist=UMAP_VARIABLES['min_dist'], 
    metric=UMAP_VARIABLES['metric'], 
    random_state=42)
emb_bin = reducer_bin.fit_transform(Xb_scaled)

emb_points_bin = project_points_to_umap(reducer_bin, scaler_bin, points_pretty)

# Individual binary UMAP
fig, ax = plt.subplots(figsize=(10, 8))
base = sns.scatterplot(
    x=emb_bin[:, 0],
    y=emb_bin[:, 1],
    hue=df_bin_plot["Signal_bin"],
    palette=bin_palette,
    s=18,
    alpha=0.6,
    linewidth=0,
    ax=ax,
)

# Overlay specialty points (if UMAP transform supported)
if emb_points_bin is not None:
    for idx, name in enumerate(points_pretty.index):
        ax.scatter(emb_points_bin[idx, 0], emb_points_bin[idx, 1], **point_style.get(name, {}))
        ax.annotate(
            point_short.get(name, name),
            (emb_points_bin[idx, 0], emb_points_bin[idx, 1]),
            textcoords="offset points",
            xytext=(4, 4),
            ha="left",
            va="bottom",
            fontsize=8,
            zorder=11,
        )

ax.set_aspect("equal", "datalim")
ax.set_title("UMAP Projection (Signal: Binary)")

# Keep the signal legend, then add a second legend for specialty points
signal_leg = ax.legend(title="Signal", loc="best")
ax.add_artist(signal_leg)
ax.legend(handles=_specialty_handles(), title="Specialty points", loc="lower right", fontsize=8, title_fontsize=9)

fig.tight_layout()
fig.savefig(out_dir / "umap_signal_binary.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ----------------------------
# TWO-PANEL FIGURES (original: UMAP + Pairplot)
# ----------------------------
# Helper to render a PairGrid onto a subplot axis by drawing it to an image first

def _pairgrid_to_image(pairgrid_fig):
    pairgrid_fig.canvas.draw()
    rgba = np.asarray(pairgrid_fig.canvas.buffer_rgba())  # (H, W, 4)
    rgb = rgba[..., :3].copy()  # (H, W, 3)
    return rgb


# --- Panel 1: Continuous (UMAP + Pairplot) ---
g_cont = sns.PairGrid(df_cont_plot, vars=dependent_variables_names, diag_sharey=False)
g_cont.map_offdiag(offdiag_scatter_continuous)
g_cont.map_diag(sns.kdeplot, fill=True, alpha=0.3)
add_specific_points_to_pairgrid(g_cont, points_pretty, annotate=True)

norm = plt.Normalize(signal_cont.min(), signal_cont.max())
sm = plt.cm.ScalarMappable(cmap="cool", norm=norm)
sm.set_array([])
g_cont.fig.colorbar(sm, ax=g_cont.axes, label="Optimal Signal", shrink=0.8)

g_cont.fig.legend(
    handles=_specialty_handles(),
    title="Specialty points",
    loc="upper right",
    bbox_to_anchor=(1.02, 1.0),
    frameon=True,
    fontsize=8,
    title_fontsize=9,
)

g_cont.fig.suptitle("Pairplot: Optimal Signal", y=1.02)

pair_img_cont = _pairgrid_to_image(g_cont.fig)
plt.close(g_cont.fig)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# left: UMAP
sns.scatterplot(
    x=emb_cont[:, 0],
    y=emb_cont[:, 1],
    hue=signal_cont,
    palette="cool",
    legend=False,
    s=18,
    alpha=0.6,
    linewidth=0,
    ax=axes[0],
)

if emb_points_cont is not None:
    for idx, name in enumerate(points_pretty.index):
        axes[0].scatter(emb_points_cont[idx, 0], emb_points_cont[idx, 1], **point_style.get(name, {}))
        axes[0].annotate(
            point_short.get(name, name),
            (emb_points_cont[idx, 0], emb_points_cont[idx, 1]),
            textcoords="offset points",
            xytext=(4, 4),
            ha="left",
            va="bottom",
            fontsize=8,
            zorder=11,
        )

smc = plt.cm.ScalarMappable(cmap="cool", norm=plt.Normalize(signal_cont.min(), signal_cont.max()))
smc.set_array([])
fig.colorbar(smc, ax=axes[0], label="Optimal Signal", shrink=0.9)
axes[0].set_aspect("equal", "datalim")
axes[0].set_title("UMAP: Optimal Signal")
axes[0].set_xlabel("UMAP 1")
axes[0].set_ylabel("UMAP 2")

# panel label A
axes[0].text(0.01, -0.05, "A", transform=axes[0].transAxes, va="top", ha="left", fontsize=14, fontweight="bold")

# right: PairGrid image
axes[1].imshow(pair_img_cont, aspect="auto")
axes[1].set_aspect("auto")
axes[1].axis("off")
axes[1].set_title("Pairplot: Optimal Signal")

# panel label B
axes[1].text(0.01, -0.05, "B", transform=axes[1].transAxes, va="top", ha="left", fontsize=14, fontweight="bold")

fig.tight_layout()
fig.savefig(out_dir / "panel_optimal_signal.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# --- Panel 2: Binary (UMAP + Pairplot) ---
g_bin = sns.PairGrid(
    df_bin_plot,
    vars=dependent_variables_names,
    hue="Signal_bin",
    palette=bin_palette,
    diag_sharey=False,
)
g_bin.map_offdiag(sns.scatterplot, alpha=0.6, s=9, linewidth=0, edgecolor=None)
g_bin.map_diag(sns.kdeplot, fill=True, alpha=0.3)
g_bin.add_legend(title="Signal")
add_specific_points_to_pairgrid(g_bin, points_pretty, annotate=True)

g_bin.fig.legend(
    handles=_specialty_handles(),
    title="Specialty points",
    loc="upper right",
    bbox_to_anchor=(1.02, 0.92),
    frameon=True,
    fontsize=8,
    title_fontsize=9,
)

g_bin.fig.suptitle("Pairplot: Signal (Binary)", y=1.02)

pair_img_bin = _pairgrid_to_image(g_bin.fig)
plt.close(g_bin.fig)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# left: UMAP
sns.scatterplot(
    x=emb_bin[:, 0],
    y=emb_bin[:, 1],
    hue=df_bin_plot["Signal_bin"],
    palette=bin_palette,
    s=18,
    alpha=0.6,
    linewidth=0,
    ax=axes[0],
)

if emb_points_bin is not None:
    for idx, name in enumerate(points_pretty.index):
        axes[0].scatter(emb_points_bin[idx, 0], emb_points_bin[idx, 1], **point_style.get(name, {}))
        axes[0].annotate(
            point_short.get(name, name),
            (emb_points_bin[idx, 0], emb_points_bin[idx, 1]),
            textcoords="offset points",
            xytext=(4, 4),
            ha="left",
            va="bottom",
            fontsize=8,
            zorder=11,
        )

axes[0].set_aspect("equal", "datalim")
axes[0].set_title("UMAP: Signal (Binary)")
axes[0].set_xlabel("UMAP 1")
axes[0].set_ylabel("UMAP 2")

# Two legends: keep Signal, add Specialty points
signal_leg = axes[0].legend(title="Signal", loc="best")
axes[0].add_artist(signal_leg)
axes[0].legend(handles=_specialty_handles(), title="Specialty points", loc="lower right", fontsize=8, title_fontsize=9)

# panel label A
axes[0].text(0.01, -0.05, "A", transform=axes[0].transAxes, va="top", ha="left", fontsize=14, fontweight="bold")

# right: PairGrid image
axes[1].imshow(pair_img_bin, aspect="auto")
axes[1].set_aspect("auto")
axes[1].axis("off")
axes[1].set_title("Pairplot: Signal (Binary)")

# panel label B
axes[1].text(0.01, -0.05, "B", transform=axes[1].transAxes, va="top", ha="left", fontsize=14, fontweight="bold")

fig.tight_layout()
fig.savefig(out_dir / "panel_signal_binary.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ----------------------------
# NEW PANELS (requested):
# 1) Both UMAPs together
# 2) Both Pairplots together
# ----------------------------

# Panel: UMAPs together
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Left: continuous
sns.scatterplot(
    x=emb_cont[:, 0],
    y=emb_cont[:, 1],
    hue=signal_cont,
    palette="cool",
    legend=False,
    s=18,
    alpha=0.6,
    linewidth=0,
    ax=axes[0],
)
if emb_points_cont is not None:
    for idx, name in enumerate(points_pretty.index):
        axes[0].scatter(emb_points_cont[idx, 0], emb_points_cont[idx, 1], **point_style.get(name, {}))
        axes[0].annotate(
            point_short.get(name, name),
            (emb_points_cont[idx, 0], emb_points_cont[idx, 1]),
            textcoords="offset points",
            xytext=(4, 4),
            ha="left",
            va="bottom",
            fontsize=8,
            zorder=11,
        )
smu = plt.cm.ScalarMappable(cmap="cool", norm=plt.Normalize(signal_cont.min(), signal_cont.max()))
smu.set_array([])
fig.colorbar(smu, ax=axes[0], label="Optimal Signal", shrink=0.9)
axes[0].set_aspect("equal", "datalim")
axes[0].set_title("UMAP: Optimal Signal")
axes[0].set_xlabel("UMAP 1")
axes[0].set_ylabel("UMAP 2")

# Right: binary
sns.scatterplot(
    x=emb_bin[:, 0],
    y=emb_bin[:, 1],
    hue=df_bin_plot["Signal_bin"],
    palette=bin_palette,
    s=18,
    alpha=0.6,
    linewidth=0,
    ax=axes[1],
)
if emb_points_bin is not None:
    for idx, name in enumerate(points_pretty.index):
        axes[1].scatter(emb_points_bin[idx, 0], emb_points_bin[idx, 1], **point_style.get(name, {}))
        axes[1].annotate(
            point_short.get(name, name),
            (emb_points_bin[idx, 0], emb_points_bin[idx, 1]),
            textcoords="offset points",
            xytext=(4, 4),
            ha="left",
            va="bottom",
            fontsize=8,
            zorder=11,
        )
axes[1].set_aspect("equal", "datalim")
axes[1].set_title("UMAP: Signal (Binary)")
axes[1].set_xlabel("UMAP 1")
axes[1].set_ylabel("UMAP 2")

# Legends on right: Signal + Specialty points
sig_leg = axes[1].legend(title="Signal", loc="best")
axes[1].add_artist(sig_leg)
axes[1].legend(handles=_specialty_handles(), title="Specialty points", loc="lower right", fontsize=8, title_fontsize=9)

fig.tight_layout()
fig.savefig(out_dir / "panel_umaps_together.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Panel: Pairplots together (render each PairGrid to image, then compose)

g_cont2 = sns.PairGrid(df_cont_plot, vars=dependent_variables_names, diag_sharey=False)
g_cont2.map_offdiag(offdiag_scatter_continuous)
g_cont2.map_diag(sns.kdeplot, fill=True, alpha=0.3)
add_specific_points_to_pairgrid(g_cont2, points_pretty, annotate=True)
norm = plt.Normalize(signal_cont.min(), signal_cont.max())
sm = plt.cm.ScalarMappable(cmap="cool", norm=norm)
sm.set_array([])
g_cont2.fig.colorbar(sm, ax=g_cont2.axes, label="Optimal Signal", shrink=0.8)
g_cont2.fig.legend(
    handles=_specialty_handles(),
    title="Specialty points",
    loc="upper right",
    bbox_to_anchor=(1.02, 1.0),
    frameon=True,
    fontsize=8,
    title_fontsize=9,
)
g_cont2.fig.suptitle("Pairplot: Optimal Signal", y=1.02)
pair_img_cont2 = _pairgrid_to_image(g_cont2.fig)
plt.close(g_cont2.fig)

g_bin2 = sns.PairGrid(
    df_bin_plot,
    vars=dependent_variables_names,
    hue="Signal_bin",
    palette=bin_palette,
    diag_sharey=False,
)
g_bin2.map_offdiag(sns.scatterplot, alpha=0.6, s=9, linewidth=0, edgecolor=None)
g_bin2.map_diag(sns.kdeplot, fill=True, alpha=0.3)
g_bin2.add_legend(title="Signal")
add_specific_points_to_pairgrid(g_bin2, points_pretty, annotate=True)

g_bin2.fig.legend(
    handles=_specialty_handles(),
    title="Specialty points",
    loc="upper right",
    bbox_to_anchor=(1.02, 0.92),
    frameon=True,
    fontsize=8,
    title_fontsize=9,
)

g_bin2.fig.suptitle("Pairplot: Signal (Binary)", y=1.02)
pair_img_bin2 = _pairgrid_to_image(g_bin2.fig)
plt.close(g_bin2.fig)

fig, axes = plt.subplots(1, 2, figsize=(22, 10))
axes[0].imshow(pair_img_cont2, aspect="auto")
axes[0].axis("off")
axes[0].set_title("Pairplot: Optimal Signal")

axes[1].imshow(pair_img_bin2, aspect="auto")
axes[1].axis("off")
axes[1].set_title("Pairplot: Signal (Binary)")

fig.tight_layout()
fig.savefig(out_dir / "panel_pairplots_together.png", dpi=300, bbox_inches="tight")
plt.close(fig)

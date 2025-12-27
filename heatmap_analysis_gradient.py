import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# defaults that are rarely changed
# things to change are at the bottom of script
HEATMAP_ORIGIN = "lower" # <--- makes y increase upward
SHOW_COLORBAR = True
VALUE_FMT = ".0f" # good for best_signal_rpp, using MEDIANS so shouldn't have decimals anyway

# in practice this is not used much since almost always have >12
AUTO_BIN_UNIQUE_GT = 12
DEFAULT_INTERVAL_FORMAT_DECIMALS = 2

# Binning helpers 
def _format_interval(iv: pd.Interval, decimals: int = DEFAULT_INTERVAL_FORMAT_DECIMALS) -> str:
    # formats an interval nicely for labeling
    # e.g., (0, 10] with decimals=0 -> "0–10"
    if decimals == 0:
        left = int(round(float(iv.left)))
        right = int(round(float(iv.right)))
        return f"{left}–{right}"
    left = round(float(iv.left), decimals)
    right = round(float(iv.right), decimals)
    return f"{left}–{right}"


def add_binned_column(
    data: pd.DataFrame,
    col: str,
    *,
    bins: int = 5,
    method: str = "qcut",
    decimals: int = 2,
    new_col: str | None = None,
) -> tuple[pd.DataFrame, str]:
    if new_col is None:
        new_col = f"{col}__bin"

    s = pd.to_numeric(data[col], errors="coerce")

    if method == "qcut":
        b = pd.qcut(s, q=bins, duplicates="drop")
    elif method == "cut":
        b = pd.cut(s, bins=bins)
    else:
        raise ValueError("method must be 'qcut' or 'cut'")

    labels = b.cat.categories
    label_map = {iv: _format_interval(iv, decimals=decimals) for iv in labels}

    out = data.copy()
    out[new_col] = b.map(label_map)
    out[new_col] = pd.Categorical(
        out[new_col],
        categories=[label_map[iv] for iv in labels],
        ordered=True,
    )
    return out, new_col


# Heatmap tables 
# ==============
def make_heatmap_tables(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    value: str,
    agg: str = "median",
    binning: dict | None = None,
    auto_bin_unique_gt: int = 12,
    sort_x_asc: bool = True,
    sort_y_asc: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if binning is None:
        binning = {}

    d = data.copy()

    def _maybe_bin(axis_name: str, col: str) -> tuple[str, pd.DataFrame]:
        spec = binning.get(axis_name, None)
        nunique = d[col].nunique(dropna=True)

        if spec is not None:
            d2, bcol = add_binned_column(
                d, col,
                bins=spec.get("bins", 6),
                method=spec.get("method", "qcut"),
                decimals=spec.get("decimals", DEFAULT_INTERVAL_FORMAT_DECIMALS),
                new_col=spec.get("new_col", None),
            )
            return bcol, d2

        if nunique > auto_bin_unique_gt:
            d2, bcol = add_binned_column(
                d, col, bins=6, method="qcut", decimals=DEFAULT_INTERVAL_FORMAT_DECIMALS)
            return bcol, d2

        return col, d

    x_col, d = _maybe_bin("x", x)
    y_col, d = _maybe_bin("y", y)

    # Value pivot
    pv = d.groupby([y_col, x_col], dropna=False)[value].agg(agg).unstack(x_col)

    # Count pivot (n per cell)
    pn = d.groupby([y_col, x_col], dropna=False)[value].count().unstack(x_col)

    # Sort in ascending order
    if sort_y_asc:
        pv = pv.sort_index(ascending=True)
        pn = pn.reindex(pv.index)
    else:
        pv = pv.sort_index(ascending=False)
        pn = pn.reindex(pv.index)

    if sort_x_asc:
        pv = pv.sort_index(axis=1, ascending=True)
        pn = pn.reindex(pv.columns, axis=1)
    else:
        pv = pv.sort_index(axis=1, ascending=False)
        pn = pn.reindex(pv.columns, axis=1)

    return pv, pn


# Plotting
# ======================
def plot_heatmap(
    pivot_values: pd.DataFrame,
    pivot_counts: pd.DataFrame,
    *,
    title: str,
    value_fmt: str = ".0f",
    origin: str = "lower",
    show_colorbar: bool = True,
    nan_text: str = "",
    save_path: str | None = None,
    x: str,
    y: str,
    labels: dict
):
    fig, ax = plt.subplots(
        figsize=(1.2 * pivot_values.shape[1] + 4,
                 0.8 * pivot_values.shape[0] + 3)
    )

    arr = pivot_values.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", origin=origin, cmap = 'plasma')

    ax.set_title(title)
    ax.set_xticks(np.arange(pivot_values.shape[1]))
    ax.set_yticks(np.arange(pivot_values.shape[0]))
    def _tick_str(v):
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return str(int(round(v)))
        return str(v)

    ax.set_xticklabels([_tick_str(c)
                    for c in pivot_values.columns], rotation=45, ha="right")
    ax.set_yticklabels([_tick_str(i) for i in pivot_values.index])

    for i in range(pivot_values.shape[0]):
        for j in range(pivot_values.shape[1]):
            v = pivot_values.iloc[i, j]
            n = pivot_counts.iloc[i, j]
            if pd.isna(v) or pd.isna(n) or n == 0:
                txt = nan_text
            else:
                txt = f"{int(round(v))}\n(n={int(n)})"
            ax.text(j, i, txt, ha="center", va="center")

    if show_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel(labels[x])
    ax.set_ylabel(labels[y])
    plt.tight_layout()
    plt.savefig(save_path) if save_path else None
    plt.show()


# Scatter-like "heatmaps" (continuous axes)
# =======================================
# These plots avoid coarse categorical bins by coloring points (scatter) or
# aggregating onto a fine 2D grid (hexbin) or interpolating a continuous field
# (tricontour). Pick a style via PLOT_STYLE below.

import matplotlib.tri as mtri

def plot_value_field(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    value: str,
    title: str,
    labels: dict,
    kind: str = "hexbin",
    agg: str = "median",
    gridsize: int = 60,
    mincnt: int = 1,
    levels: int = 60,
    show_colorbar: bool = True,
    save_path: str | None = None,
):
    """Scatter-like heatmap using continuous x/y.

    kind:
      - 'scatter': 1 point per row, colored by `value`
      - 'hexbin' : hexagonal binning with reduce_C_function (median/mean/sum)
      - 'tricontour': interpolated field over irregular grid (can look very smooth)
    """
    d = data[[x, y, value]].copy()
    for c in (x, y, value):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(title)

    if kind == "scatter":
        m = ax.scatter(
            d[x].to_numpy(),
            d[y].to_numpy(),
            c=d[value].to_numpy(),
            cmap="plasma",
            s=18,
            alpha=0.75,
            linewidths=0,
        )

    elif kind == "hexbin":
        reduce_func = {
            "median": np.median,
            "mean": np.mean,
            "sum": np.sum,
        }.get(agg, np.median)

        m = ax.hexbin(
            d[x].to_numpy(),
            d[y].to_numpy(),
            C=d[value].to_numpy(),
            reduce_C_function=reduce_func,
            gridsize=gridsize,
            mincnt=mincnt,
            cmap="plasma",
        )

    elif kind == "tricontour":
        # Triangulation can't handle duplicate (x, y) points well, so aggregate first
        d2 = d.groupby([x, y], as_index=False)[value].agg(agg)

        tri = mtri.Triangulation(d2[x].to_numpy(), d2[y].to_numpy())
        m = ax.tricontourf(
            tri,
            d2[value].to_numpy(),
            levels=levels,
            cmap="plasma",
        )
    else:
        raise ValueError("kind must be one of: 'scatter', 'hexbin', 'tricontour'")

    ax.set_xlabel(labels.get(x, x))
    ax.set_ylabel(labels.get(y, y))
    ax.grid(True, alpha=0.2)

    if show_colorbar:
        cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(labels.get(value, value))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()



# axis labels

LABELS = {
    # randomized constants
    "n_programs": "Number of Programs",
    "n_applicants": "Number of Applicants",
    "interviews_per_spot": "Interviews per Position",
    "n_positions": "Number of Positions (Pre-Rounding)",
    "max_applications": "Maximum Applications per Applicant",
    # calculated/derived constants
    "spots_per_program": "Spots per Program",
    "simulated_positions": "Number of Positions",
    "applicants_per_position": "Applicants per Position",
    # outcomes
    "best_signal_rpp": "Optimal Signal (Reviews per Program)",
}

# Imports + load
# ==============

INPUT_PATH = "heatmap_results/heatmap_results.csv"
OUTPUT_DIRECTORY = "heatmap_figures/"

df = pd.read_csv(INPUT_PATH)
print("Loaded:", df.shape)
df.head(5)


# Run heatmap settings:
# note that the save path will auto-prefix the OUTPUT_DIRECTORY
# binning method is best with quartile cutting, but you can do
# linear cuts if you want even-sized bins instead of even-populated bins
# do linear cuts, set method to "cut"

# if you leave out binning for an axis, it will auto-bin if there are
# more unique values than AUTO_BIN_UNIQUE_GT
# the decimals are for graphing

# file name is the key

PLOT_STYLE = "tricontour"   # one of: "binned", "scatter", "hexbin", "tricontour"
# higher -> finer "pixels" (try 40–120 depending on data size)
HEXBIN_GRIDSIZE = 40
# require at least this many points per hex (filters sparse areas)
HEXBIN_MINCNT = 1
TRICONTOUR_LEVELS = 60  # higher -> smoother contours (but can overfit)

graphs_to_create= {
    # graph heatmaps with variable being optimal signals 
    "app_vs_pos_signal": {
        "x": "n_applicants",
        "y": "simulated_positions",
        "heatmap_variable": "best_signal_rpp",
        "title": "Optimal Signal Heat-Map: Applicants vs. Number of Positions",
        "agg": "median",
        "binning": {
            "x": {"bins": 4, "method": "qcut", "decimals": 0},
            "y": {"bins": 4, "method": "qcut", "decimals": 0},
        }
    },
    "aperp_vs_interviews_per_spot_signal": {
        "x": "applicants_per_position",
        "y": "interviews_per_spot",
        "heatmap_variable": "best_signal_rpp",
        "title": "Optimal Signal Heat-Map: Applicants per Position vs. Interviews per Spot",
        "agg": "median",
        "binning": {
            "x": {"bins": 4, "method": "qcut", "decimals": 0},
            "y": {"bins": 4, "method": "qcut", "decimals": 0},
        }
    },
    "aprep_vs_max_applications_signal": {
        "x": "applicants_per_position",
        "y": "max_applications",
        "heatmap_variable": "best_signal_rpp",
        "title": "Optimal Signal Heat-Map: Applicants per Position vs. Maximum Applications",
        "agg": "median",
        "binning": {
            "x": {"bins": 4, "method": "qcut", "decimals": 0},
            "y": {"bins": 4, "method": "qcut", "decimals": 0},
        }
    },
    # graph heatmaps with variable being reviews per program at optimal signal

    "app_vs_pos_rpp": {
        "x": "n_applicants",
        "y": "simulated_positions",
        "heatmap_variable": "reviews_per_program_best_rpp",
        "title": "Reviews per Program Heat-Map: Applicants vs. Number of Positions",
        "agg": "median",
        "binning": {
            "x": {"bins": 4, "method": "qcut", "decimals": 0},
            "y": {"bins": 4, "method": "qcut", "decimals": 0},
        }
    },
    "aperp_vs_interviews_per_spot_rpp": {
        "x": "applicants_per_position",
        "y": "interviews_per_spot",
        "heatmap_variable": "reviews_per_program_best_rpp",
        "title": "Reviews per Program Heat-Map: Applicants per Position vs. Interviews per Spot",
        "agg": "median",
        "binning": {
            "x": {"bins": 4, "method": "qcut", "decimals": 0},
            "y": {"bins": 4, "method": "qcut", "decimals": 0},
        }
    },
    "aprep_vs_max_applications_rpp": {
        "x": "applicants_per_position",
        "y": "max_applications",
        "heatmap_variable": "reviews_per_program_best_rpp",
        "title": "Reviews per Program Heat-Map: Applicants per Position vs. Maximum Applications",
        "agg": "median",
        "binning": {
            "x": {"bins": 4, "method": "qcut", "decimals": 0},
            "y": {"bins": 4, "method": "qcut", "decimals": 0},
        }
    }
}

for graph_key, settings in graphs_to_create.items():
    save_path = OUTPUT_DIRECTORY + graph_key + ".png"

    if PLOT_STYLE == "binned":
        # Original behavior: categorical bins + annotated cells
        pivot_v, pivot_n = make_heatmap_tables(
            df,
            x=settings["x"],
            y=settings["y"],
            value=settings["heatmap_variable"],
            agg=settings.get("agg", "median"),
            binning=settings.get("binning", None),
            auto_bin_unique_gt=AUTO_BIN_UNIQUE_GT,
        )
        plot_heatmap(
            pivot_v,
            pivot_n,
            title=settings["title"],
            value_fmt=VALUE_FMT,
            origin=HEATMAP_ORIGIN,
            show_colorbar=SHOW_COLORBAR,
            save_path=save_path,
            x=settings["x"],
            y=settings["y"],
            labels=LABELS,
        )
    else:
        # New behavior: continuous x/y with gradient coloring
        plot_value_field(
            df,
            x=settings["x"],
            y=settings["y"],
            value=settings["heatmap_variable"],
            title=settings["title"],
            labels=LABELS,
            kind=PLOT_STYLE,
            agg=settings.get("agg", "median"),
            gridsize=HEXBIN_GRIDSIZE,
            mincnt=HEXBIN_MINCNT,
            levels=TRICONTOUR_LEVELS,
            show_colorbar=SHOW_COLORBAR,
            save_path=save_path,
        )

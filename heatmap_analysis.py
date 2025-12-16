#!/usr/bin/env python3
"""
Goal:
- Visualize the optimal number of signals. The heatmap color is ALWAYS:
    VALUE_COL = "best_signal_rpp"

Per your request:
1) Heatmap indicator: best_signal_rpp
2) Plot set A axes: Applicants/Positions ratio vs max_applications
   - Applicants/Positions ratio = n_applicants / (n_programs * spots_per_program)
3) Plot set B axes: n_applicants vs total positions
   - total positions = n_programs * spots_per_program

Outputs:
- A companion grid CSV per plot for manuscript reproducibility.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

RESULTS_PATH = Path("heatmap_results/heatmap_results.csv")
# Where to write figures/grids
OUTDIR = Path("heatmap_figures/")

# Heatmap value (will change to other values later)
VALUE_COL = "best_signal_rpp"

# Aggregation within each cell when multiple scenarios fall into the same bin
AGG: Literal["median", "mean"] = "mean"

# Mask cells with fewer than MIN_COUNT observations.
# - Use None to auto-pick a sensible value based on file size
MIN_COUNT: Optional[int] = 1

# Annotation behavior
ANNOTATE_IF_CELLS_LEQ = 100  # annotate only if grid is not too dense

# Tick label density (keeps axes readable)
MAX_TICKLABELS = 12

# For "discrete-looking" axes with too many unique values, we bin instead
DISCRETE_UNIQUE_LIMIT = 20

# Default binning controls (can be overridden per plot)
DEFAULT_BINS_X = 5
DEFAULT_BINS_Y = 5
DEFAULT_BIN_METHOD_X: Literal["linear", "quantile"] = "linear"
DEFAULT_BIN_METHOD_Y: Literal["linear", "quantile"] = "linear"

# Colormap for best_signal_rpp (discrete integer colorbar)
CMAP = "viridis"

# Plots to generate (edit/add to explore other X/Y combinations)
# Derived columns:
# total_positions = n_programs * spots_per_program
# applicants_per_position = n_applicants / total_positions

PLOTS = [
    # Set A: Applicants/Positions ratio vs max_applications
    # (max_applications often looks discrete; we allow discrete and will auto-bin if too many unique values)
    dict(
        name="optimal_rpp_applicants_vs_spots",
        x="n_applicants",
        y="total_positions",
        bins_x=5,  # slightly higher since x is fairly discrete
        bins_y=5,
        method_x="linear",
        method_y="linear",
        x_discrete_hint=True,
        y_discrete_hint=False,
        title="Optimal Signals by Applicants vs. Total Positions",
    ),
    # Set B: n_applicants vs total positions (n_programs * spots_per_program)
    # These can have many unique values -> bin both for readability.
    dict(
        name="optimal_rpp_app_position_ratio_vs_interviews_per_spot",
        x="applicants_per_position",
        y="interviews_per_spot",
        bins_x=5,
        bins_y=5,
        method_x="linear",
        method_y="linear",
        x_discrete_hint=False,
        y_discrete_hint=False,
        title="Optimal Signals by Applications/Position Ratio vs. Interviews per Spot",
    ),
]


# =========================
# Internal helpers
# =========================

REQUIRED_INPUT_COLS = [
    "n_programs",
    "n_applicants",
    "spots_per_program",
    "interviews_per_spot",
    "max_applications",
    VALUE_COL,
]

AXIS_LABELS = {
    "max_applications": "Application CAP per Applicant",
    "n_applicants": "Number of Applicants",
    "n_programs": "Number of Programs",
    "spots_per_program": "Spots per Program",
    "total_positions": "Total Residency Positions",
    "applicants_per_position": "Applicants per Position Ratio",
    "interviews_per_spot": "Interviews Offered per Position",
}


def _auto_min_count(n_rows: int, user_value: Optional[int]) -> int:
    """
    Pick a min_count that won't blank small datasets but still reduces noise on large ones.
    """
    if user_value is not None:
        return int(user_value)
    # 2% of rows, capped at 20, with a floor of 2.
    return max(2, min(20, int(round(0.02 * n_rows))))


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {missing}")

    # Coerce required cols and drop unusable rows
    for c in REQUIRED_INPUT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED_INPUT_COLS)

    # Stabilize integer-like fields
    int_like = [
        "n_programs", 
        "n_applicants", 
        "spots_per_program", 
        "interviews_per_spot", 
        "max_applications", 
        VALUE_COL]
    for c in int_like:
        df[c] = np.floor(df[c]).astype(int)

    return df


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_positions"] = df["n_programs"] * df["spots_per_program"]
    df["applicants_per_position"] = df["n_applicants"] / df["total_positions"]
    return df


def _make_bins(s: pd.Series, n_bins: int, method: Literal["linear", "quantile"]) -> np.ndarray:
    s = s.astype(float)

    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    if method == "quantile":
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(s, qs))
        # Discrete / low-variability data can collapse quantiles
        if len(edges) < 3:
            return _make_bins(s, n_bins=n_bins, method="linear")
        return edges

    # linear
    lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
    if lo == hi:
        return np.array([lo - 0.5, hi + 0.5])
    return np.linspace(lo, hi, n_bins + 1)


def _format_num(x: float) -> str:
    ax = abs(float(x))
    if ax >= 100:
        return f"{x:.0f}"
    if ax >= 10:
        return f"{x:.0f}"
    if ax >= 1:
        return f"{x:.1f}"
    return f"{x:.2f}"


def _interval_labels(intervals: List[pd.Interval], treat_as_int_values: bool) -> List[str]:
    labels: List[str] = []
    for itv in intervals:
        left = float(itv.left)
        right = float(itv.right)
        if treat_as_int_values:
            lo = int(np.ceil(left))
            hi = int(np.floor(right))
            labels.append(f"{lo}–{hi}")
        else:
            labels.append(f"{_format_num(left)}–{_format_num(right)}")
    return labels


def _sparsify_ticklabels(labels: List[str], max_labels: int) -> List[str]:
    if len(labels) <= max_labels:
        return labels
    k = int(np.ceil(len(labels) / max_labels))
    return [lab if (i % k == 0) else "" for i, lab in enumerate(labels)]


def _pivot_discrete(df: pd.DataFrame, x: str, y: str, value: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pivot on exact values (no binning). Use only when unique values are small enough.
    """
    grouped = df.groupby([y, x], observed=True)[value]
    if AGG == "median":
        pv = grouped.median().unstack(x)
    else:
        pv = grouped.mean().unstack(x)
    pc = grouped.size().unstack(x).fillna(0).astype(int)

    # Sort for readable axes
    pv = pv.sort_index().sort_index(axis=1)
    pc = pc.reindex(index=pv.index, columns=pv.columns)
    return pv, pc


def _pivot_binned(
    df: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    bins_x: int,
    bins_y: int,
    method_x: Literal["linear", "quantile"],
    method_y: Literal["linear", "quantile"],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bin x/y into intervals and aggregate into a 2D grid.
    """
    df = df.copy()
    x_edges = _make_bins(df[x], n_bins=bins_x, method=method_x)
    y_edges = _make_bins(df[y], n_bins=bins_y, method=method_y)

    df["_x_bin"] = pd.cut(df[x], bins=x_edges, include_lowest=True)
    df["_y_bin"] = pd.cut(df[y], bins=y_edges, include_lowest=True)

    grouped = df.groupby(["_y_bin", "_x_bin"], observed=True)[value]
    if AGG == "median":
        pv = grouped.median().unstack("_x_bin")
    else:
        pv = grouped.mean().unstack("_x_bin")
    pc = grouped.size().unstack("_x_bin").fillna(0).astype(int)

    pv = pv.sort_index().sort_index(axis=1)
    pc = pc.reindex(index=pv.index, columns=pv.columns)
    return pv, pc

def _pivot_mixed(
    df: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    x_mode: Literal["discrete", "binned"],
    y_mode: Literal["discrete", "binned"],
    bins_x: int,
    bins_y: int,
    method_x: Literal["linear", "quantile"],
    method_y: Literal["linear", "quantile"],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Support mixed discrete/binned axes.

    Examples:
      - x discrete, y binned: groupby(y_bin, x)
      - x binned, y discrete: groupby(y, x_bin)
    """
    df = df.copy()

    if x_mode == "binned":
        x_edges = _make_bins(df[x], n_bins=bins_x, method=method_x)
        df["_x_bin"] = pd.cut(df[x], bins=x_edges, include_lowest=True)
        x_key = "_x_bin"
    else:
        x_key = x

    if y_mode == "binned":
        y_edges = _make_bins(df[y], n_bins=bins_y, method=method_y)
        df["_y_bin"] = pd.cut(df[y], bins=y_edges, include_lowest=True)
        y_key = "_y_bin"
    else:
        y_key = y

    grouped = df.groupby([y_key, x_key], observed=True)[value]
    if AGG == "median":
        pv = grouped.median().unstack(x_key)
    else:
        pv = grouped.mean().unstack(x_key)
    pc = grouped.size().unstack(x_key).fillna(0).astype(int)

    pv = pv.sort_index().sort_index(axis=1)
    pc = pc.reindex(index=pv.index, columns=pv.columns)
    return pv, pc



def _choose_pivot(
    df: pd.DataFrame,
    x: str,
    y: str,
    bins_x: int,
    bins_y: int,
    method_x: Literal["linear", "quantile"],
    method_y: Literal["linear", "quantile"],
    x_discrete_hint: bool,
    y_discrete_hint: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Decide per-axis whether to treat it as discrete (exact values) or binned (pd.cut),
    then build the corresponding pivot grid.

    This makes it easy to mix:
      - x discrete + y binned (common: max_applications vs a ratio)
      - x binned + y discrete
      - both binned (common for large unique counts)
    """
    x_nu = int(df[x].nunique(dropna=True))
    y_nu = int(df[y].nunique(dropna=True))

    x_mode: Literal["discrete", "binned"] = "discrete" if (x_discrete_hint and x_nu <= DISCRETE_UNIQUE_LIMIT) else "binned"
    y_mode: Literal["discrete", "binned"] = "discrete" if (y_discrete_hint and y_nu <= DISCRETE_UNIQUE_LIMIT) else "binned"

    # If neither axis is flagged discrete, default to binning both.
    if not x_discrete_hint:
        x_mode = "binned"
    if not y_discrete_hint:
        y_mode = "binned"

    # Build pivot
    if x_mode == "discrete" and y_mode == "discrete":
        return _pivot_discrete(df, x=x, y=y, value=VALUE_COL)

    return _pivot_mixed(
        df=df,
        x=x,
        y=y,
        value=VALUE_COL,
        x_mode=x_mode,
        y_mode=y_mode,
        bins_x=bins_x,
        bins_y=bins_y,
        method_x=method_x,
        method_y=method_y,
    )



def _plot_heatmap(
    pivot_value: pd.DataFrame,
    pivot_count: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    out_png: Path,
    out_pdf: Path,
    min_count: int,
) -> None:
    # Publication-friendly defaults
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 450,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    vals = pivot_value.to_numpy(dtype=float)
    counts = pivot_count.to_numpy(dtype=int)

    # Mask low-support cells
    masked = np.where(counts >= min_count, vals, np.nan)

    # If everything masked, fall back to non-empty cells (prevents blank figures)
    if np.isnan(masked).all():
        masked = np.where(counts > 0, vals, np.nan)

    # Discrete integer colorbar for best_signal_rpp
    finite = masked[np.isfinite(masked)]
    if finite.size == 0:
        raise RuntimeError("All grid cells are empty after masking; try fewer bins or lower MIN_COUNT.")

    vmin = int(np.nanmin(finite))
    vmax = int(np.nanmax(finite))
    boundaries = np.arange(vmin - 0.5, vmax + 1.5, 1)
    norm = mcolors.BoundaryNorm(boundaries, ncolors=plt.get_cmap(CMAP).N)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.6), constrained_layout=True)
    im = ax.imshow(masked, origin="lower", aspect="auto", interpolation="nearest", cmap=CMAP, norm=norm)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Optimal # signals (best_signal_rpp)")
    cbar.set_ticks(list(range(vmin, vmax + 1)))

    # Axis labels
    ax.set_xlabel(AXIS_LABELS.get(x, x))
    ax.set_ylabel(AXIS_LABELS.get(y, y))

    # Tick labels
    x_labels: List[str]
    y_labels: List[str]

    if isinstance(pivot_value.columns[0], pd.Interval):
        # binned
        x_labels = _interval_labels(list(pivot_value.columns), treat_as_int_values=(x in {"max_applications", "n_applicants", "total_positions"}))
    else:
        # discrete values
        x_labels = [str(int(v)) for v in pivot_value.columns.to_list()]

    if isinstance(pivot_value.index[0], pd.Interval):
        y_labels = _interval_labels(list(pivot_value.index), treat_as_int_values=(y in {"max_applications", "n_applicants", "total_positions"}))
    else:
        y_labels = [str(int(v)) for v in pivot_value.index.to_list()]

    x_labels = _sparsify_ticklabels(x_labels, MAX_TICKLABELS)
    y_labels = _sparsify_ticklabels(y_labels, MAX_TICKLABELS)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)

    # Light gridlines
    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Title includes aggregation + mask setting (helps reviewers)
    ax.set_title(f"{title}\n({AGG}, cell mask: n<{min_count})")

    # Annotate if not too many cells
    if pivot_value.shape[0] * pivot_value.shape[1] <= ANNOTATE_IF_CELLS_LEQ:
        for i in range(masked.shape[0]):
            for j in range(masked.shape[1]):
                if not np.isfinite(masked[i, j]):
                    continue
                if counts[i, j] < min_count:
                    continue
                ax.text(
                    j,
                    i,
                    f"{int(round(masked[i, j]))}\n(n={counts[i, j]})",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)




def _write_grid_csv(grid: pd.DataFrame, path: Path) -> None:
    """
    Write a pivot grid to CSV with readable labels.

    If index/columns are Interval objects (from pd.cut), convert them to compact
    "lo–hi" strings without triggering pandas' dtype-cast warnings.
    """
    g = grid.copy()

    def _intervals_to_labels(itvs: pd.IntervalIndex) -> list[str]:
        out: list[str] = []
        for itv in itvs:
            left = float(itv.left)
            right = float(itv.right)
            out.append(f"{_format_num(left)}–{_format_num(right)}")
        return out

    if isinstance(g.index, pd.IntervalIndex):
        g.index = _intervals_to_labels(g.index)
    if isinstance(g.columns, pd.IntervalIndex):
        g.columns = _intervals_to_labels(g.columns)

    g = g.where(pd.notnull(g), "")
    g.to_csv(path, index=True)



def main() -> None:

    df = pd.read_csv(RESULTS_PATH)
    df = _coerce_numeric(df)
    df = _add_derived_columns(df)

    min_count = _auto_min_count(len(df), MIN_COUNT)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    for spec in PLOTS:
        x = spec["x"]
        y = spec["y"]

        if x not in df.columns:
            raise ValueError(f"Unknown X '{x}'. Available columns include: {sorted(df.columns)}")
        if y not in df.columns:
            raise ValueError(f"Unknown Y '{y}'. Available columns include: {sorted(df.columns)}")

        bins_x = int(spec.get("bins_x", DEFAULT_BINS_X))
        bins_y = int(spec.get("bins_y", DEFAULT_BINS_Y))
        method_x = spec.get("method_x", DEFAULT_BIN_METHOD_X)
        method_y = spec.get("method_y", DEFAULT_BIN_METHOD_Y)
        x_discrete_hint = bool(spec.get("x_discrete_hint", False))
        y_discrete_hint = bool(spec.get("y_discrete_hint", False))

        pv, pc = _choose_pivot(
            df=df,
            x=x,
            y=y,
            bins_x=bins_x,
            bins_y=bins_y,
            method_x=method_x,
            method_y=method_y,
            x_discrete_hint=x_discrete_hint,
            y_discrete_hint=y_discrete_hint,
        )

        base = OUTDIR / spec["name"]
        out_png = base.with_suffix(".png")
        out_pdf = base.with_suffix(".pdf")

        _plot_heatmap(
            pivot_value=pv,
            pivot_count=pc,
            x=x,
            y=y,
            title=spec.get("title", spec["name"]),
            out_png=out_png,
            out_pdf=out_pdf,
            min_count=min_count,
        )

        # Save companion grids
        # Save companion grids (stringify IntervalIndex to avoid pandas warnings and make CSVs readable)
        _write_grid_csv(pv, base.with_suffix(".grid_values.csv"))
        _write_grid_csv(pc, base.with_suffix(".grid_counts.csv"))

        nonempty = int((pc.to_numpy() > 0).sum())
        print(f"Wrote {out_png.name} / {out_pdf.name} | grid={pv.shape} | nonempty_cells={nonempty} | min_count={min_count}")


if __name__ == "__main__":
    main()

import warnings
import re
import difflib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# -------------------------
# CONFIG
# -------------------------

source_directories = {
    "results/all_data_nrmp/": "Base Case",
    "results/all_data_nrmp_gamma/": "Random # of Applications",
    "results/all_data_nrmp_no_quartile/": "Random Application Distribution",
    "results/all_data_nrmp_no_quartile_gamma/": "Random # and Distribution of Applications",
    "results/all_data_nrmp_random_program_rank_list/": "Random Program Rank List",
    "results/all_data_nrmp_random_applicant_rank_list/": "Random Applicant Rank List",
    "results/all_data_nrmp_random_applicant_and_program_rank_list/": "Random Applicant and Program Rank Lists",
}

# Output roots
INDIVIDUAL_ROOT = Path("figures/NRMP")
JOINT_OUTPUT_DIR = Path("figures/graphs_NRMP_joint")

PARAMETERS = [
    "unfilled_spots",
    "reviews_per_program",
    "pct_interview_given_signal",
    "pct_of_app_match_via_signal_given_matched",
]

PARAMETER_TITLES = [
    "Number of Unfilled Spots",
    "Number of Reviews Per Program",
    "Percent Interview Given Signal",
    "Percent of Matches from Signaled Applicants",
]

# Programs to include in joint overlay panels (expect 5)
MULTI_GRAPH_PROGRAMS = [
    "Anesthesiology",
    "Dermatology",
    "Surgery(Categorical)",
    "Internal Medicine(Categorical)",
    "Vascular Surgery",
]

ALL_PROGRAMS_FALLBACK = [
    "Anesthesiology",
    "Child Neurology",
    "Dermatology",
    "Emergency Medicine",
    "Family Medicine",
    "Internal Medicine(Categorical)",
    "Medicine-Emergency Med",
    "Medicine-Pediatrics",
    "Medicine-Preliminary(PGY-1 Only)",
    "Medicine-Primary",
    "Interventional Radiology(Integrated)",
    "Neurological Surgery",
    "Neurology",
    "Obstetrics-Gynecology",
    "Orthopaedic Surgery",
    "Otolaryngology",
    "Pathology",
    "Pediatrics(Categorical)",
    "Pediatrics-Primary",
    "Physical Medicine & Rehab",
    "Plastic Surgery(Integrated)",
    "Psychiatry",
    "Radiology-Diagnostic",
    "Surgery(Categorical)",
    "Surgery-Preliminary(PGY-1 Only)",
    "Thoracic Surgery",
    "Transitional(PGY-1 Only)",
    "Vascular Surgery",
]

# Joint plot behavior
EXCLUDE_ZERO_IN_JOINT = False
CONFIDENCE = 0.95

# Joint plot styling
JOINT_LINEWIDTH = 1.2
JOINT_LINE_ALPHA = 0.95
JOINT_MARKERSIZE = 3
JOINT_CI_ALPHA = 0.08

LEGEND_FONT_SIZE = 8
LEGEND_FRAME_ALPHA = 0.95

# Heatmap styling
HEATMAP_CMAP = "cool"

# Joint line x-axis: fixed tick spec requested
JOINT_XTICKS = list(range(0, 41, 5))
JOINT_XLIM = (0, 40)


# -------------------------
# PLOTTING STYLE
# -------------------------
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

_SIGNAL_RE = re.compile(r"^-?\d+$")


def reviews_reduction_table(
    scenarios: list[dict],
    programs: list[str],
    optimal_sigs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculates the percent reduction in reviews per program:
    Reduction = 100 * (Reviews_at_0 - Reviews_at_Optimal) / Reviews_at_0
    """
    cols = [sc["label"] for sc in scenarios]
    mat = pd.DataFrame(index=programs, columns=cols, dtype=float)

    for sc in scenarios:
        sc_label = sc["label"]
        for program in programs:
            prog_summary = sc["program_summaries"].get(program)
            if prog_summary is None:
                continue

            stats_param = prog_summary.get("reviews_per_program")
            if stats_param is None:
                continue

            sigs = stats_param["signals"]
            means = stats_param["mean"]

            # 1. Get value at Signal 0
            try:
                idx_zero = int(np.where(sigs == 0)[0][0])
                val_zero = means[idx_zero]
            except (IndexError, ValueError):
                val_zero = np.nan

            # 2. Get value at Optimal Signal
            sig_opt = (
                optimal_sigs.loc[program, sc_label]
                if (program in optimal_sigs.index and sc_label in optimal_sigs.columns)
                else np.nan
            )

            try:
                idx_opt = int(np.where(sigs == int(sig_opt))[0][0])
                val_opt = means[idx_opt]
            except (IndexError, ValueError, TypeError):
                val_opt = np.nan

            # 3. Calculate Percent Reduction
            if np.isfinite(val_zero) and np.isfinite(val_opt) and val_zero > 0:
                reduction = 100.0 * (val_zero - val_opt) / val_zero
                mat.loc[program, sc_label] = reduction

    return mat

def _normalize_program_name(name: str) -> str:
    """Normalize program names to make joint-plot program selection resilient to small filename/stem differences."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def resolve_programs_for_joint(scenarios: list[dict], desired_programs: list[str]) -> list[str]:
    """Resolve MULTI_GRAPH_PROGRAMS against discovered program stems.

    Part A iterates over discovered program stems, but Part B uses MULTI_GRAPH_PROGRAMS.
    If stems differ by punctuation/spacing/case, programs can plot individually but be skipped in joint plots.
    This resolver first tries exact matches, then a normalized match, and warns about anything still missing.
    """
    # Programs available anywhere across scenarios
    available = sorted({p for sc in scenarios for p in sc.get("program_summaries", {}).keys()})

    # Map normalized -> canonical available name (first seen)
    norm_to_available: dict[str, str] = {}
    for p in available:
        n = _normalize_program_name(p)
        if n not in norm_to_available:
            norm_to_available[n] = p

    resolved: list[str] = []
    missing: list[str] = []

    def _present(name: str) -> bool:
        return any(name in sc.get("program_summaries", {}) for sc in scenarios)

    for p in desired_programs:
        # 1) exact
        if _present(p):
            candidate = p
        else:
            # 2) normalized match
            candidate = norm_to_available.get(_normalize_program_name(p))

        if candidate and _present(candidate):
            if candidate not in resolved:  # de-dupe while preserving order
                resolved.append(candidate)
        else:
            missing.append(p)

    if missing:
        lines = [
            "Joint plot: the following MULTI_GRAPH_PROGRAMS were not found in loaded data (even after normalization):"
        ]
        for p in missing:
            matches = difflib.get_close_matches(p, available, n=5, cutoff=0.55)
            if matches:
                lines.append(f"  - {p!r} (close matches: {matches})")
            else:
                lines.append(f"  - {p!r}")
        warnings.warn("\n".join(lines), RuntimeWarning)

    return resolved


def _tick_step(n: int, max_ticks: int = 18) -> int:
    if n <= 0:
        return 1
    return max(1, int(np.ceil(n / max_ticks)))


def mean_and_ci(values, confidence: float = CONFIDENCE):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return np.nan, np.nan, np.nan

    m = float(arr.mean())
    if n == 1:
        return m, m, m

    sem = float(stats.sem(arr, ddof=1))
    if (not np.isfinite(sem)) or sem <= 0.0:
        return m, m, m

    df = n - 1
    tcrit = float(stats.t.ppf((1.0 + confidence) / 2.0, df))
    if not np.isfinite(tcrit):
        return m, m, m

    margin = tcrit * sem
    return m, m - margin, m + margin


def _signal_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if str(c).strip().lower() == "parameter":
            continue
        name = str(c).strip()
        if _SIGNAL_RE.match(name):
            cols.append(name)
    return cols


def summarize_file(csv_path: Path):
    df = pd.read_csv(csv_path)

    # normalize Parameter col
    if "Parameter" not in df.columns:
        lower_map = {c.lower(): c for c in df.columns}
        if "parameter" in lower_map:
            df = df.rename(columns={lower_map["parameter"]: "Parameter"})
        else:
            raise ValueError(
                f"{csv_path.name} missing 'Parameter' column; found: {list(df.columns)}"
            )

    signal_cols = _signal_columns(df)
    if not signal_cols:
        raise ValueError(
            f"{csv_path.name} has no integer-like signal columns (e.g. '0','1','2',...). "
            f"Found: {list(df.columns)}"
        )

    signal_values = sorted({int(c) for c in signal_cols})
    signal_strs = [str(s) for s in signal_values]

    df[signal_strs] = df[signal_strs].apply(pd.to_numeric, errors="coerce")

    summary = {}
    for param in PARAMETERS:
        param_df = df[df["Parameter"] == param]

        means, lows, highs = [], [], []
        if param_df.empty:
            warnings.warn(
                f"{csv_path.name}: parameter '{param}' not found; plots will be NaN.",
                RuntimeWarning,
            )
            means = [np.nan] * len(signal_strs)
            lows = [np.nan] * len(signal_strs)
            highs = [np.nan] * len(signal_strs)
        else:
            for col in signal_strs:
                m, lo, hi = mean_and_ci(param_df[col].values)
                means.append(m)
                lows.append(lo)
                highs.append(hi)

        summary[param] = {
            "signals": np.array(signal_values, dtype=int),
            "mean": np.array(means, dtype=float),
            "ci_low": np.array(lows, dtype=float),
            "ci_high": np.array(highs, dtype=float),
        }

    return summary


def discover_programs_union(scenario_dirs: list[Path]) -> list[str]:
    """
    Union program CSV stems across ALL scenario directories.
    This prevents missing programs in joint plots when some scenarios have extra CSVs.
    """
    stems = set()
    for d in scenario_dirs:
        if not d.exists():
            continue
        for p in d.glob("*.csv"):
            stems.add(p.stem)

    if stems:
        return sorted(stems, key=lambda s: s.lower())

    return ALL_PROGRAMS_FALLBACK


def sanitize_folder(name: str) -> str:
    return name.replace("/", "-").strip()


def plot_individual_program(
    scenario_label: str,
    program: str,
    summary: dict,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    for param, title in zip(PARAMETERS, PARAMETER_TITLES):
        stats_param = summary[param]
        sigs = stats_param["signals"]
        mean = stats_param["mean"]
        lo = stats_param["ci_low"]
        hi = stats_param["ci_high"]

        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            is_zero = sigs == 0
            is_nonzero = ~is_zero

            # main line (nonzero)
            if np.isfinite(mean[is_nonzero]).any():
                ax.plot(
                    sigs[is_nonzero],
                    mean[is_nonzero],
                    linewidth=2,
                    marker="o",
                )

            # CI band (omit 0 CI for clarity)
            lo_plot = lo.copy()
            hi_plot = hi.copy()
            lo_plot[is_zero] = np.nan
            hi_plot[is_zero] = np.nan
            finite_ci = np.isfinite(lo_plot) & np.isfinite(hi_plot)
            if finite_ci.any():
                ax.fill_between(sigs, lo_plot, hi_plot, where=finite_ci, alpha=0.15)

            # 0-signal highlight
            if is_zero.any() and np.isfinite(mean[is_zero]).any():
                ax.scatter(
                    sigs[is_zero],
                    mean[is_zero],
                    s=90,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=2,
                    zorder=6,
                    label="No Signaling",
                )
                ax.legend(frameon=False, loc="best")

            ax.set_xlabel("Number of Signals")
            ax.set_ylabel(title)
            ax.set_title(f"{scenario_label} — {program}\n{title} vs Number of Signals")

            step = _tick_step(len(sigs))
            if len(sigs) > 0:
                ax.set_xticks(sigs[::step])

            ax.grid(True, alpha=0.3, linestyle="--")
            fig.tight_layout()
            fig.savefig(out_dir / f"{param}.png", bbox_inches="tight")
        finally:
            plt.close(fig)


# -------------------------
# JOINT LINE PLOTS (5 program panels, legend outside; wide figure)
# -------------------------
def _prepare_joint_axes_5_wide():
    # 2x3 grid gives us space for 5 panels; 6th is unused (turned off)
    # Make it wide and reserve right margin for the legend.
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13.8, 5.2), squeeze=False)
    return fig, axes.ravel()


def plot_joint_by_program(
    param: str,
    title: str,
    programs: list[str],
    scenarios: list[dict],
    out_png: Path,
):
    # Keep order, but only plot the first 5 programs (expected)
    programs = list(programs)[:5]

    fig, axes_flat = _prepare_joint_axes_5_wide()
    program_axes = axes_flat[:5]
    unused_ax = axes_flat[5]
    unused_ax.axis("off")

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(scenarios))]

    # In case fewer than 5 programs are available, hide extras
    for ax in program_axes[len(programs):]:
        ax.axis("off")

    for ax, program in zip(program_axes, programs):
        # build unified signal grid across scenarios for this program
        all_sigs = set()
        for sc in scenarios:
            prog_summary = sc["program_summaries"].get(program)
            if prog_summary is None:
                continue
            stats_param = prog_summary.get(param)
            if stats_param is None:
                continue
            all_sigs.update([int(x) for x in stats_param["signals"]])

        unified = np.array(sorted(all_sigs), dtype=int)
        if EXCLUDE_ZERO_IN_JOINT:
            unified = unified[unified != 0]

        if unified.size == 0:
            ax.set_title(program)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
            ax.axis("off")
            continue

        # plot each scenario line
        for i, sc in enumerate(scenarios):
            prog_summary = sc["program_summaries"].get(program)
            if prog_summary is None:
                continue

            stats_param = prog_summary.get(param)
            if stats_param is None:
                continue

            sigs = stats_param["signals"]
            mean = stats_param["mean"]
            lo = stats_param["ci_low"]
            hi = stats_param["ci_high"]

            # reindex onto unified
            idx = {int(s): k for k, s in enumerate(unified)}
            full_mean = np.full(unified.shape, np.nan, dtype=float)
            full_lo = np.full(unified.shape, np.nan, dtype=float)
            full_hi = np.full(unified.shape, np.nan, dtype=float)

            for s, m, l, h in zip(sigs, mean, lo, hi):
                s = int(s)
                if s in idx:
                    j = idx[s]
                    full_mean[j] = m
                    full_lo[j] = l
                    full_hi[j] = h

            if np.isfinite(full_mean).any():
                ax.plot(
                    unified,
                    full_mean,
                    linewidth=JOINT_LINEWIDTH,
                    marker="o",
                    markersize=JOINT_MARKERSIZE,
                    label=sc["label"],
                    color=colors[i],
                    alpha=JOINT_LINE_ALPHA,
                )

            finite_ci = np.isfinite(full_lo) & np.isfinite(full_hi)
            if finite_ci.any():
                ax.fill_between(
                    unified,
                    full_lo,
                    full_hi,
                    where=finite_ci,
                    alpha=JOINT_CI_ALPHA,
                    color=colors[i],
                )

        ax.set_title(program)
        ax.set_xlabel("Signals")
        ax.set_ylabel(title)

        # x ticks/range: keep the requested 0–40 view when data fits, but expand if needed
        xmax_data = int(np.nanmax(unified)) if unified.size else JOINT_XLIM[1]
        if xmax_data <= JOINT_XLIM[1]:
            ax.set_xticks(JOINT_XTICKS)
            ax.set_xlim(*JOINT_XLIM)
        else:
            xmax_ceil = int(((xmax_data + 4) // 5) * 5)  # round up to nearest 5
            ax.set_xlim(JOINT_XLIM[0], xmax_ceil)
            ax.set_xticks(list(range(JOINT_XLIM[0], xmax_ceil + 1, 5)))

        ax.grid(True, alpha=0.25, linestyle="--")

    # Legend outside panels, right side
    # Get handles from the first axis with any plotted line
    handle_src = None
    for ax in program_axes:
        h, _ = ax.get_legend_handles_labels()
        if h:
            handle_src = ax
            break

    if handle_src is not None:
        handles, labels = handle_src.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.995, 0.5),
            frameon=True,
            fancybox=True,
            framealpha=LEGEND_FRAME_ALPHA,
            fontsize=LEGEND_FONT_SIZE,
            borderpad=0.6,
            labelspacing=0.5,
            handlelength=1.6,
            handletextpad=0.6,
            markerscale=0.9,
        )
        # Leave space on the right for legend
        fig.tight_layout(rect=[0, 0, 0.90, 1])
    else:
        fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# HEATMAPS (optimal signal + parameter values at optimal)
# -------------------------
def _param_title(param: str) -> str:
    m = dict(zip(PARAMETERS, PARAMETER_TITLES))
    return m.get(param, param)


def optimal_signal_table(
    scenarios: list[dict],
    programs: list[str],
    objective_param: str = "reviews_per_program",
) -> pd.DataFrame:
    cols = [sc["label"] for sc in scenarios]
    mat = pd.DataFrame(index=programs, columns=cols, dtype=float)

    for sc in scenarios:
        for program in programs:
            prog_summary = sc["program_summaries"].get(program)
            if prog_summary is None:
                continue
            stats_param = prog_summary.get(objective_param)
            if stats_param is None:
                continue
            sigs = stats_param["signals"]
            mean = stats_param["mean"]
            if mean.size == 0 or np.all(np.isnan(mean)):
                continue
            j = int(np.nanargmin(mean))
            mat.loc[program, sc["label"]] = float(sigs[j])

    return mat


def value_at_optimal_signal_table(
    scenarios: list[dict],
    programs: list[str],
    optimal_sigs: pd.DataFrame,
    param: str,
) -> pd.DataFrame:
    cols = [sc["label"] for sc in scenarios]
    mat = pd.DataFrame(index=programs, columns=cols, dtype=float)

    for sc in scenarios:
        sc_label = sc["label"]
        for program in programs:
            sig = (
                optimal_sigs.loc[program, sc_label]
                if (program in optimal_sigs.index and sc_label in optimal_sigs.columns)
                else np.nan
            )
            if not np.isfinite(sig):
                continue
            sig = int(sig)

            prog_summary = sc["program_summaries"].get(program)
            if prog_summary is None:
                continue

            stats_param = prog_summary.get(param)
            if stats_param is None:
                continue

            sigs = stats_param["signals"]
            mean = stats_param["mean"]

            # exact match only (signals are discrete)
            try:
                k = int(np.where(sigs == sig)[0][0])
            except Exception:
                continue

            v = float(mean[k]) if k < mean.size else np.nan
            mat.loc[program, sc_label] = v

    return mat


def save_heatmap_table(
    table: pd.DataFrame,
    out_csv: Path,
    out_png: Path,
    title: str,
    cmap: str = HEATMAP_CMAP,
    annotate_integers: bool = False,
):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=True)

    programs = list(table.index)
    cols = list(table.columns)
    data = table.values.astype(float)

    fig_w = 1.2 + 0.6 * max(1, len(cols))
    fig_h = 1.2 + 0.30 * max(1, len(programs))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(programs)))
    ax.set_yticklabels(programs)
    ax.set_title(title)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Program")

    # annotate cells
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = data[r, c]
            if np.isfinite(v):
                txt = f"{int(round(v))}" if annotate_integers else f"{v:.2f}"
                ax.text(c, r, txt, ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def save_joint_4panel_heatmap(
    tables: list[tuple[str, pd.DataFrame]],
    out_png: Path,
    cmap: str = HEATMAP_CMAP,
):
    if len(tables) != 4:
        raise ValueError("save_joint_4panel_heatmap expects exactly 4 tables.")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24.0, 18.0), squeeze=False)
    axes_flat = axes.ravel()

    for ax, (panel_title, table) in zip(axes_flat, tables):
        programs = list(table.index)
        cols = list(table.columns)
        data = table.values.astype(float)

        im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_title(panel_title)
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(cols, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(programs)))
        ax.set_yticklabels(programs)
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Program")

        annotate_integers = "Optimal # Signals" in panel_title
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                v = data[r, c]
                if np.isfinite(v):
                    txt = f"{int(round(v))}" if annotate_integers else f"{v:.2f}"
                    ax.text(c, r, txt, ha="center", va="center", fontsize=6)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    JOINT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_ROOT.mkdir(parents=True, exist_ok=True)

    # Validate scenario dirs
    scenario_items = []
    for d, label in source_directories.items():
        p = Path(d)
        if not p.exists():
            warnings.warn(
                f"Scenario directory not found (skipping): {p.resolve()}",
                RuntimeWarning,
            )
            continue
        scenario_items.append({"dir": p, "label": label})

    if not scenario_items:
        raise FileNotFoundError(
            "None of the scenario directories exist. Check `source_directories` paths."
        )

    # IMPORTANT: union program list across all scenario dirs (fixes missing programs in joint plots)
    scenario_dirs = [s["dir"] for s in scenario_items]
    all_programs = discover_programs_union(scenario_dirs)

    # Load all scenarios, all programs
    scenarios = []
    for sc in scenario_items:
        program_summaries = {}
        for program in all_programs:
            csv_path = sc["dir"] / f"{program}.csv"
            if not csv_path.exists():
                continue
            try:
                program_summaries[program] = summarize_file(csv_path)
            except Exception as e:
                warnings.warn(f"Failed to summarize {csv_path}: {e}", RuntimeWarning)

        scenarios.append(
            {"label": sc["label"], "dir": sc["dir"], "program_summaries": program_summaries}
        )

    # -------------------------
    # A) Individual graphs
    # -------------------------
    for sc in scenarios:
        scenario_label = sanitize_folder(sc["label"])
        for program, summary in sc["program_summaries"].items():
            out_dir = INDIVIDUAL_ROOT / scenario_label / sanitize_folder(program)
            plot_individual_program(
                scenario_label=sc["label"],
                program=program,
                summary=summary,
                out_dir=out_dir,
            )

    # -------------------------
    # B) Joint line overlays (5 programs; legend outside; wide)
    # -------------------------
    programs_joint = resolve_programs_for_joint(scenarios, MULTI_GRAPH_PROGRAMS)
    if not programs_joint:
        warnings.warn(
            "No MULTI_GRAPH_PROGRAMS found in loaded data; joint plots skipped.",
            RuntimeWarning,
        )
    else:
        for param, title in zip(PARAMETERS, PARAMETER_TITLES):
            out_png = JOINT_OUTPUT_DIR / f"{param}_by_program.png"
            plot_joint_by_program(param, title, programs_joint, scenarios, out_png)

    # -------------------------
    # C) Heatmaps (unchanged)
    # -------------------------
    programs_all = sorted({p for sc in scenarios for p in sc["program_summaries"].keys()})

    opt = optimal_signal_table(
        scenarios=scenarios,
        programs=programs_all,
        objective_param="reviews_per_program",
    )

    save_heatmap_table(
        table=opt,
        out_csv=JOINT_OUTPUT_DIR / "heatmap_optimal_signals_min_reviews_per_program.csv",
        out_png=JOINT_OUTPUT_DIR / "heatmap_optimal_signals_min_reviews_per_program.png",
        title="Optimal # Signals by Program (Minimizing Reviews Per Program)",
        cmap=HEATMAP_CMAP,
        annotate_integers=True,
    )

    include_params = [
        "unfilled_spots",
        "reviews_per_program",
        "pct_interview_given_signal",
    ]

    param_tables = []
    for p in include_params:
        t = value_at_optimal_signal_table(
            scenarios=scenarios,
            programs=programs_all,
            optimal_sigs=opt,
            param=p,
        )
        param_tables.append((p, t))

        save_heatmap_table(
            table=t,
            out_csv=JOINT_OUTPUT_DIR / f"heatmap_{p}_at_optimal_signals.csv",
            out_png=JOINT_OUTPUT_DIR / f"heatmap_{p}_at_optimal_signals.png",
            title=f"{_param_title(p)} (at Optimal # Signals)",
            cmap=HEATMAP_CMAP,
            annotate_integers=False,
        )
        
    reduction_table = reviews_reduction_table(
        scenarios=scenarios,
        programs=programs_all,
        optimal_sigs=opt
    )

    save_heatmap_table(
        table=reduction_table,
        out_csv=JOINT_OUTPUT_DIR / "heatmap_reviews_percent_reduction.csv",
        out_png=JOINT_OUTPUT_DIR / "heatmap_reviews_percent_reduction.png",
        title="Percent Reduction in Reviews/Program (Signal 0 vs Optimal)",
        cmap=HEATMAP_CMAP,  # Using a green map to highlight positive reduction
        annotate_integers=False,
    )

    joint_panels = [
        ("Optimal # Signals (min Reviews/Program)", opt),
        (
            f"{_param_title('unfilled_spots')} (at Optimal # Signals)",
            dict(param_tables)["unfilled_spots"],
        ),
        (
            f"{_param_title('reviews_per_program')} (at Optimal # Signals)",
            dict(param_tables)["reviews_per_program"],
        ),
        (
            f"{_param_title('pct_interview_given_signal')} (at Optimal # Signals)",
            dict(param_tables)["pct_interview_given_signal"],
        ),
    ]

    save_joint_4panel_heatmap(
        tables=joint_panels,
        out_png=JOINT_OUTPUT_DIR / "heatmaps_joint_4panel.png",
        cmap=HEATMAP_CMAP,
    )


if __name__ == "__main__":
    main()

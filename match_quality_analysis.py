import warnings
import re
import json
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""""

This script generates and saves match-composition heatmaps at the program-specific
optimal signal for every (scenario, program) pair that has:

1) An optimal signal in that scenario's summary-results CSV, and
2) Parquet data available for that optimal signal.

For MULTI_GRAPH_PROGRAMS, the script also creates a multi-panel figure per program
that shows one heatmap per scenario (by default, the six non-base scenarios), so each
program gets a single "6-heatmap" figure.

The heatmap bins are configurable via N_CUTS (e.g., 10=deciles, 4=quartiles,
100=percentiles). Heatmap matrices are always saved as CSV at the optimal signal.

The simulation run remains the unit of replication: we compute the per-simulation
row-normalized composition matrix and then average matrices across simulations.
"""

# -------------------------
# CONFIG
# -------------------------

HEATMAP_STYLE = 'viridis'

# Where to write figures and tables
OUTPUT_ROOT = Path("figures/match_quality_analysis/")
N_CUTS = 10  # default number of cuts for heatmaps

# Output sub-roots
BY_SCENARIO_ROOT = OUTPUT_ROOT / "by_scenario"          # scenario/constant_set outputs
MULTI_PROGRAM_ROOT = OUTPUT_ROOT / "multi_program"      # multi-panel per program

# Sensitivity analyses and their STORE_ALL_DATA_PATH roots
# NOTE: these may be absolute (Windows) or relative paths. The script uses pathlib.
analyses_and_path = {
    "nrmp_only": "C:/Users/ayman/Desktop/simulation_results/all_data_nrmp/",
    "nrmp_with_gamma": "C:/Users/ayman/Desktop/simulation_results/all_data_nrmp_gamma/",
    "nrmp_no_quartile": "C:/Users/ayman/Desktop/simulation_results/all_data_nrmp_no_quartile/",
    "nrmp_no_quartile_gamma": "C:/Users/ayman/Desktop/simulation_results/all_data_nrmp_no_quartile_gamma/",
    "nrmp_random_program_rank_list": "C:/Users/ayman/Desktop/simulation_results/all_data_nrmp_random_program_rank_list/",
    "nrmp_random_applicant_rank_list": "C:/Users/ayman/Desktop/simulation_results/all_data_nrmp_random_applicant_rank_list/",
    "nrmp_random_applicant_and_program_rank_list": "C:/Users/ayman/Desktop/simulation_results/all_data_nrmp_random_applicant_and_program_rank_list/",
}

# Summary-result CSVs that contain the optimal signal per program (from prior analysis)
summary_result_paths = {
    "nrmp_only": "results/nrmp_results.csv",
    "nrmp_with_gamma": "results/nrmp_results_gamma.csv",
    "nrmp_no_quartile": "results/nrmp_results_no_quartile.csv",
    "nrmp_no_quartile_gamma": "results/nrmp_results_no_quartile_gamma.csv",
    "nrmp_random_program_rank_list": "results/nrmp_results_random_program_rank_list.csv",
    "nrmp_random_applicant_rank_list": "results/nrmp_results_random_applicant_rank_list.csv",
    "nrmp_random_applicant_and_program_rank_list": "results/nrmp_results_random_applicant_and_program_rank_list.csv",
}

# Scenario labels (to match data_analysis_NRMP naming)
SCENARIO_DISPLAY_NAMES = {
    "nrmp_only": "Base Case",
    "nrmp_with_gamma": "Random # of Applications",
    "nrmp_no_quartile": "Random Application Distribution",
    "nrmp_no_quartile_gamma": "Random # and Distribution of Applications",
    "nrmp_random_program_rank_list": "Random Program Rank List",
    "nrmp_random_applicant_rank_list": "Random Applicant Rank List",
    "nrmp_random_applicant_and_program_rank_list": "Random Applicant and Program Rank Lists",
}

# Programs to include in multi-panel heatmap figures
MULTI_GRAPH_PROGRAMS = [
    "Anesthesiology",
    "Dermatology",
    "Surgery (Categorical)",
    "Internal Medicine (Categorical)",
    "Vascular Surgery",
]

# Multi-panel scenario selection:
# Include all scenarios (including the Base Case) in the multi-panel figures.
# Set this to a non-empty set of display names to exclude specific scenarios.
MULTI_PANEL_EXCLUDE_DISPLAY_NAMES: set[str] = set()

# Column names in the summary-results CSV
PROGRAM_COL = "result_file_prefix"
BEST_SIGNALS_COL = "best_signal_rpp"

# Filename patterns written by simulation.py
_APPLICANTS_RE = re.compile(r"signal_(?P<signal>-?\d+)_sim_(?P<sim>-?\d+)_applicants\.parquet$")
_PROGRAMS_RE = re.compile(r"signal_(?P<signal>-?\d+)_sim_(?P<sim>-?\d+)_programs\.parquet$")

# -------------------------
# PLOTTING STYLE
# -------------------------
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# -------------------------
# DATA STRUCTURES
# -------------------------

@dataclass(frozen=True)
class SimKey:
    scenario: str
    constant_set: str
    signal: int
    sim: int


@dataclass
class SimHeatmap:
    key: SimKey
    matrix: np.ndarray


# -------------------------
# DISCOVERY / IO HELPERS
# -------------------------

def discover_constant_sets(store_root: Path) -> list[Path]:
    """Immediate subdirectories correspond to constant sets (programs)."""
    if not store_root.exists():
        return []
    return sorted([p for p in store_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def discover_signal_sim_pairs(constant_dir: Path) -> list[tuple[int, int]]:
    """Return (signal, sim) pairs with BOTH applicants and programs parquet present."""
    pairs = set()
    if not constant_dir.exists():
        return []

    for p in constant_dir.glob("*_applicants.parquet"):
        m = _APPLICANTS_RE.search(p.name)
        if not m:
            continue
        pairs.add((int(m.group("signal")), int(m.group("sim"))))

    valid = []
    for signal, sim in sorted(pairs):
        apps = constant_dir / f"signal_{signal}_sim_{sim}_applicants.parquet"
        progs = constant_dir / f"signal_{signal}_sim_{sim}_programs.parquet"
        if apps.exists() and progs.exists():
            valid.append((signal, sim))
    return valid


def _read_parquet_pair(constant_dir: Path, signal: int, sim: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    apps_path = constant_dir / f"signal_{signal}_sim_{sim}_applicants.parquet"
    progs_path = constant_dir / f"signal_{signal}_sim_{sim}_programs.parquet"
    applicants = pd.read_parquet(apps_path)
    programs = pd.read_parquet(progs_path)
    return applicants, programs


def _safe_json_loads(x):
    if pd.isna(x):
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    s = str(x)
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        # Be forgiving; sometimes brackets/quotes can be slightly off.
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            return []


# -------------------------
# OPTIMAL SIGNAL LOOKUP
# -------------------------

def load_best_signal_map(summary_csv: Path) -> dict[str, int]:
    """Return {PROGRAM_COL -> BEST_SIGNALS_COL} from the summary-results CSV."""
    if not summary_csv.exists():
        return {}

    df = pd.read_csv(summary_csv)
    if PROGRAM_COL not in df.columns or BEST_SIGNALS_COL not in df.columns:
        warnings.warn(
            f"Summary results CSV {summary_csv} missing required columns: {PROGRAM_COL!r}, {BEST_SIGNALS_COL!r}.",
            RuntimeWarning,
        )
        return {}

    out: dict[str, int] = {}
    prog = df[PROGRAM_COL].astype(str)
    best = pd.to_numeric(df[BEST_SIGNALS_COL], errors="coerce")

    for p, s in zip(prog, best):
        if pd.isna(s):
            continue
        try:
            out[str(p)] = int(round(float(s)))
        except Exception:
            continue

    return out


def select_optimal_signal(
    signals_present: list[int],
    best_signal_map: dict[str, int],
    constant_set: str,
) -> Optional[int]:
    """Return the optimal signal if present; otherwise None (heatmap is skipped)."""
    if not signals_present:
        return None

    best = best_signal_map.get(constant_set)
    if best is None:
        warnings.warn(
            f"{constant_set}: no optimal signal found in summary CSV; skipping heatmap.",
            RuntimeWarning,
        )
        return None

    best_i = int(best)
    if best_i not in set(signals_present):
        warnings.warn(
            f"{constant_set}: optimal signal={best_i} (from summary CSV) not in stored signals; skipping heatmap.",
            RuntimeWarning,
        )
        return None

    return best_i


# -------------------------
# HEATMAP COMPUTATION
# -------------------------

def _cut_from_index(idx: int, n: int, n_cuts: int) -> int:
    """Return cut in [1..n_cuts] based on index in [0..n-1]."""
    if n_cuts <= 1:
        return 1
    if n <= 0:
        return 1
    idx = int(max(0, min(idx, n - 1)))
    c = int(np.floor((idx / n) * n_cuts)) + 1
    return int(min(max(c, 1), n_cuts))


def _compute_sim_heatmap(
    scenario: str,
    constant_set: str,
    signal: int,
    sim: int,
    applicants: pd.DataFrame,
    programs: pd.DataFrame,
    n_cuts: int,
) -> SimHeatmap:
    expected_app_cols = {"applicant_id", "matched_program", "final_rank_list"}
    expected_prog_cols = {"program_id", "n_positions", "final_rank_list", "tentative_matches"}
    missing_a = expected_app_cols - set(applicants.columns)
    missing_p = expected_prog_cols - set(programs.columns)
    if missing_a:
        raise ValueError(f"Applicants parquet missing columns: {sorted(missing_a)}")
    if missing_p:
        raise ValueError(f"Programs parquet missing columns: {sorted(missing_p)}")

    n_applicants = int(applicants["applicant_id"].max()) + 1
    n_programs = int(programs["program_id"].max()) + 1

    programs = programs.copy()
    programs["tentative_matches_parsed"] = programs["tentative_matches"].map(_safe_json_loads)

    match_rows = []
    for _, row in programs.iterrows():
        pid = int(row["program_id"])
        matches = [int(x) for x in row["tentative_matches_parsed"]]
        for aid in matches:
            match_rows.append((pid, aid))

    mat = np.zeros((n_cuts, n_cuts), dtype=float)

    if match_rows:
        match_df = pd.DataFrame(match_rows, columns=["program_id", "applicant_id"])
        match_df["prog_cut"] = match_df["program_id"].map(lambda x: _cut_from_index(int(x), n_programs, n_cuts))
        match_df["app_cut"] = match_df["applicant_id"].map(lambda x: _cut_from_index(int(x), n_applicants, n_cuts))

        for (pc, ac), g in match_df.groupby(["prog_cut", "app_cut"]):
            mat[int(pc) - 1, int(ac) - 1] = float(len(g))

        row_sums = mat.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mat = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)

    key = SimKey(scenario=scenario, constant_set=constant_set, signal=int(signal), sim=int(sim))
    return SimHeatmap(key=key, matrix=mat)


def _aggregate_heatmaps(heats: list[SimHeatmap], signal: int) -> np.ndarray:
    mats = [h.matrix for h in heats if h.key.signal == signal]
    if not mats:
        return np.zeros((1, 1), dtype=float)
    return np.nanmean(np.stack(mats, axis=0), axis=0)


# -------------------------
# UNMATCHED-APPLICANT HISTOGRAM (BY DECILE/CUT)
# -------------------------

def _compute_sim_unmatched_props(applicants: pd.DataFrame, n_cuts: int) -> np.ndarray:
    """Return length-n_cuts array: proportion unmatched within each applicant cut.

    Unmatched is defined as matched_program being NA or an empty/whitespace string.
    Applicant cuts are computed using the same index-based cut logic as the heatmaps.
    """
    if 'applicant_id' not in applicants.columns or 'matched_program' not in applicants.columns:
        raise ValueError('Applicants parquet must include applicant_id and matched_program to compute unmatched histogram.')

    a = applicants[['applicant_id', 'matched_program']].copy()
    a['applicant_id'] = pd.to_numeric(a['applicant_id'], errors='coerce').astype('Int64')
    a = a.dropna(subset=['applicant_id']).copy()
    a['applicant_id'] = a['applicant_id'].astype(int)

    # Determine N from the max applicant_id, consistent with other parts of the script.
    n_applicants = int(a['applicant_id'].max()) + 1 if len(a) else 0

    # Cut assignment
    a['app_cut'] = a['applicant_id'].map(lambda x: _cut_from_index(int(x), n_applicants, n_cuts))

    # Unmatched: matched_program is NA or empty/whitespace after string conversion
    mp = a['matched_program']
    unmatched = mp.isna() | (mp.astype(str).str.strip() == '')
    a['unmatched'] = unmatched.astype(int)

    denom = a.groupby('app_cut', observed=True).size().reindex(range(1, n_cuts + 1), fill_value=0).astype(float)
    numer = a.groupby('app_cut', observed=True)['unmatched'].sum().reindex(range(1, n_cuts + 1), fill_value=0).astype(float)

    with np.errstate(divide='ignore', invalid='ignore'):
        props = np.divide(numer.values, denom.values, out=np.zeros(n_cuts, dtype=float), where=denom.values > 0)

    return props


def _plot_unmatched_histogram(props: np.ndarray, title: str, out_png: Path, n_cuts: int):
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    try:
        x = np.arange(1, n_cuts + 1)
        ax.bar(x, props)
        ax.set_title(title)
        ax.set_xlabel('Applicant decile (cut)')
        ax.set_ylabel('Proportion unmatched within cut')
        ax.set_xticks(x)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis='y', alpha=0.25)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches='tight')
    finally:
        plt.close(fig)


def _save_unmatched_hist_csv(props: np.ndarray, out_csv: Path, n_cuts: int):
    df = pd.DataFrame({
        'app_cut': list(range(1, n_cuts + 1)),
        'prop_unmatched': props.astype(float),
    })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


# -------------------------
# PLOTTING / SAVING HELPERS
# -------------------------

def _normalize_program_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def resolve_programs_for_multi(available_programs: list[str], desired_programs: list[str]) -> list[str]:
    """Resolve MULTI_GRAPH_PROGRAMS against discovered program names (normalization + close matches)."""
    norm_to_available: dict[str, str] = {}
    for p in available_programs:
        n = _normalize_program_name(p)
        if n not in norm_to_available:
            norm_to_available[n] = p

    resolved: list[str] = []
    missing: list[str] = []

    for p in desired_programs:
        if p in available_programs:
            candidate = p
        else:
            candidate = norm_to_available.get(_normalize_program_name(p))

        if candidate and candidate in available_programs:
            if candidate not in resolved:
                resolved.append(candidate)
        else:
            missing.append(p)

    if missing:
        lines = [
            "Multi-panel heatmaps: the following MULTI_GRAPH_PROGRAMS were not found in loaded data (even after normalization):"
        ]
        for p in missing:
            matches = difflib.get_close_matches(p, available_programs, n=5, cutoff=0.55)
            if matches:
                lines.append(f"  - {p!r} (close matches: {matches})")
            else:
                lines.append(f"  - {p!r}")
        warnings.warn("\n".join(lines), RuntimeWarning)

    return resolved


def _plot_heatmap_ax(
    ax,
    mat: np.ndarray,
    title: str,
    n_cuts: int,
    vmin: float,
    vmax: float,
    cmap: str = HEATMAP_STYLE,
):
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Applicant Decile")
    ax.set_ylabel("Program Decile")
    ax.set_xticks(np.arange(n_cuts))
    ax.set_yticks(np.arange(n_cuts))
    ax.set_xticklabels([str(i) for i in range(1, n_cuts + 1)])
    ax.set_yticklabels([str(i) for i in range(1, n_cuts + 1)])
    return im


def _plot_heatmap_single(mat: np.ndarray, title: str, out_png: Path, n_cuts: int):
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    try:
        vmax = max(1e-12, float(np.nanmax(mat)))
        im = _plot_heatmap_ax(ax, mat, title=title, n_cuts=n_cuts, vmin=0.0, vmax=vmax, cmap=HEATMAP_STYLE)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Proportion Gradient")
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
    finally:
        plt.close(fig)


def _save_heatmap_csv(mat: np.ndarray, out_csv: Path, n_cuts: int):
    df = pd.DataFrame(mat, index=[f"prog_cut_{i}" for i in range(1, n_cuts + 1)])
    df.columns = [f"app_cut_{i}" for i in range(1, n_cuts + 1)]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=True)


def _prepare_multi_axes(n_panels: int):
    if n_panels <= 1:
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        return fig, np.array([ax], dtype=object), (1, 1)

    # Aim for up to 3 columns (keeps 6 panels as a 2x3 grid by default).
    ncols = int(min(3, n_panels))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False)
    return fig, axes, (nrows, ncols)


def _prepare_fixed_grid(nrows: int, ncols: int):
    """Prepare a fixed grid of axes (used to reserve empty cells for legends)."""
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.6 * ncols, 4.2 * nrows),
        squeeze=False,
    )
    return fig, axes, (nrows, ncols)


def _plot_multi_program_heatmaps(
    program: str,
    scenario_to_mat: dict[str, tuple[np.ndarray, int]],
    out_png: Path,
    n_cuts: int,
):
    scenario_labels = list(scenario_to_mat.keys())
    mats = [scenario_to_mat[s][0] for s in scenario_labels]
    vmax = max(1e-12, float(np.nanmax([np.nanmax(m) for m in mats]))) if mats else 1.0

    # Use a fixed 3x3 grid when we have <= 9 panels so we can place the colorbar
    # into an unused grid cell (avoids overlaying any heatmap panels).
    if len(scenario_labels) <= 9:
        fig, axes, _shape = _prepare_fixed_grid(3, 3)
    else:
        fig, axes, _shape = _prepare_multi_axes(len(scenario_labels))
    try:
        axes_flat = axes.ravel()
        last_im = None

        for i, sc in enumerate(scenario_labels):
            mat, sig = scenario_to_mat[sc]
            ax = axes_flat[i]
            last_im = _plot_heatmap_ax(
                ax,
                mat,
                title=f"{sc}\n(signal={sig})",
                n_cuts=n_cuts,
                vmin=0.0,
                vmax=vmax,
                cmap=HEATMAP_STYLE,
            )

        # Place the shared colorbar in the first unused axis, if available.
        cax = None
        if len(scenario_labels) < axes_flat.size:
            cax = axes_flat[len(scenario_labels)]
            cax.clear()

        # Turn off any remaining unused axes (excluding the colorbar axis if used).
        for j in range(len(scenario_labels), axes_flat.size):
            if cax is not None and axes_flat[j] is cax:
                continue
            axes_flat[j].axis("off")

        if last_im is not None:
            if cax is not None:
                fig.colorbar(last_im, cax=cax, label="Proportion within Program Decile")
            else:
                # Fallback: place colorbar beside the grid.
                fig.colorbar(
                    last_im,
                    ax=axes_flat.tolist(),
                    fraction=0.020,
                    pad=0.02,
                    label="Proportion within Program Decile",
                )

        fig.suptitle(program, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
    finally:
        plt.close(fig)


def _plot_unmatched_histogram_ax(ax, props: np.ndarray, title: str, n_cuts: int):
    """Plot a single unmatched-applicant histogram into an existing axis."""
    x = np.arange(1, n_cuts + 1)
    ax.bar(x, props)
    ax.set_title(title)
    ax.set_xlabel("Applicant Decile")
    ax.set_ylabel("Proportion Unmatched")
    ax.set_xticks(x)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)


def _plot_multi_program_unmatched_histograms(
    program: str,
    scenario_to_props: dict[str, tuple[np.ndarray, int]],
    out_png: Path,
    n_cuts: int,
):
    """Multi-panel unmatched-applicant histograms (one per scenario) for a program."""
    scenario_labels = list(scenario_to_props.keys())

    fig, axes, _shape = _prepare_multi_axes(len(scenario_labels))
    try:
        axes_flat = axes.ravel()

        for i, sc in enumerate(scenario_labels):
            props, sig = scenario_to_props[sc]
            ax = axes_flat[i]
            _plot_unmatched_histogram_ax(
                ax,
                props,
                title=f"{sc}\n(signal={sig})",
                n_cuts=n_cuts,
            )

        for j in range(len(scenario_labels), axes_flat.size):
            axes_flat[j].axis("off")

        fig.suptitle(f"{program} — Unmatched Applicants", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
    finally:
        plt.close(fig)


# -------------------------
# MAIN
# -------------------------

def main():
    if N_CUTS < 1:
        raise ValueError("N_CUTS must be >= 1")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    BY_SCENARIO_ROOT.mkdir(parents=True, exist_ok=True)
    MULTI_PROGRAM_ROOT.mkdir(parents=True, exist_ok=True)

    # Cache for multi-panel plots:
    #   heatmaps:   program -> scenario_display -> (matrix, optimal_signal)
    #   unmatched:  program -> scenario_display -> (props_by_cut, optimal_signal)
    multi_cache: dict[str, dict[str, tuple[np.ndarray, int]]] = {}
    multi_cache_unmatched: dict[str, dict[str, tuple[np.ndarray, int]]] = {}

    for scenario_key, store_path in analyses_and_path.items():
        scenario_display = SCENARIO_DISPLAY_NAMES.get(scenario_key, scenario_key)
        store_root = Path(str(store_path))

        constant_dirs = discover_constant_sets(store_root)
        if not constant_dirs:
            warnings.warn(
                f"{scenario_display}: no constant-set directories found under {store_root}. "
                "Confirm STORE_ALL_DATA_PATH points to the folder containing {result_file_prefix}/ subfolders.",
                RuntimeWarning,
            )
            continue

        summary_csv = Path(summary_result_paths.get(scenario_key, ""))
        best_signal_map = load_best_signal_map(summary_csv) if summary_csv else {}

        for constant_dir in constant_dirs:
            constant_set = constant_dir.name
            pairs = discover_signal_sim_pairs(constant_dir)
            if not pairs:
                continue

            signals_present = sorted({s for s, _ in pairs})
            heat_sig = select_optimal_signal(signals_present, best_signal_map, constant_set)
            if heat_sig is None:
                continue

            pairs_at_opt = [(s, i) for s, i in pairs if int(s) == int(heat_sig)]
            if not pairs_at_opt:
                warnings.warn(
                    f"{scenario_display}/{constant_set}: no parquet pairs found at optimal signal={heat_sig}; skipping heatmap.",
                    RuntimeWarning,
                )
                continue

            print(
                f"Scenario={scenario_display} | constant_set={constant_set} | "
                f"optimal_signal={heat_sig} | sims={len(pairs_at_opt)} | n_cuts={N_CUTS}"
            )

            sim_heats: list[SimHeatmap] = []
            sim_unmatched_props: list[np.ndarray] = []
            for signal, sim in pairs_at_opt:
                try:
                    applicants, programs = _read_parquet_pair(constant_dir, signal, sim)
                    sim_unmatched_props.append(_compute_sim_unmatched_props(applicants, n_cuts=N_CUTS))
                    sim_heats.append(
                        _compute_sim_heatmap(
                            scenario=scenario_display,
                            constant_set=constant_set,
                            signal=signal,
                            sim=sim,
                            applicants=applicants,
                            programs=programs,
                            n_cuts=N_CUTS,
                        )
                    )
                except Exception as e:
                    warnings.warn(
                        f"Failed to process {scenario_display}/{constant_set} (signal={signal}, sim={sim}): {e}",
                        RuntimeWarning,
                    )

            if not sim_heats:
                warnings.warn(
                    f"{scenario_display}/{constant_set}: no heatmaps computed at optimal signal={heat_sig}; skipping.",
                    RuntimeWarning,
                )
                continue

            mat = _aggregate_heatmaps(sim_heats, heat_sig)

            out_dir = BY_SCENARIO_ROOT / scenario_display.replace("/", "-") / constant_set
            out_dir.mkdir(parents=True, exist_ok=True)

            _plot_heatmap_single(
                mat,
                title=(
                    f"{scenario_display} — {constant_set}\n"
                    f"Program-cut → Applicant-cut composition (signal={heat_sig}, cuts={N_CUTS})"
                ),
                out_png=out_dir /
                f"heatmap_program_cut_to_applicant_cut_signal_{heat_sig}_cuts_{N_CUTS}.png",
                n_cuts=N_CUTS,
            )

            _save_heatmap_csv(
                mat,
                out_csv=out_dir /
                f"heatmap_program_cut_to_applicant_cut_signal_{heat_sig}_cuts_{N_CUTS}.csv",
                n_cuts=N_CUTS,
            )

            # Unmatched-applicant histogram (by applicant decile/cut)
            if sim_unmatched_props:
                props = np.nanmean(np.stack(sim_unmatched_props, axis=0), axis=0)
                _plot_unmatched_histogram(
                    props,
                    title=(
                        f"{scenario_display} — {constant_set}\n"
                        f"Unmatched applicants by decile (signal={heat_sig}, cuts={N_CUTS})"
                    ),
                    out_png=out_dir / f"unmatched_applicants_hist_signal_{heat_sig}_cuts_{N_CUTS}.png",
                    n_cuts=N_CUTS,
                )
                _save_unmatched_hist_csv(
                    props,
                    out_csv=out_dir / f"unmatched_applicants_hist_signal_{heat_sig}_cuts_{N_CUTS}.csv",
                    n_cuts=N_CUTS,
                )

                # Cache for multi-panel unmatched plots
                multi_cache_unmatched.setdefault(constant_set, {})[scenario_display] = (props, int(heat_sig))

            # Cache for multi-panel plots
            multi_cache.setdefault(constant_set, {})[scenario_display] = (mat, int(heat_sig))

    # -------------------------
    # MULTI-GRAPH PROGRAMS: one figure per program, one heatmap per scenario
    # -------------------------
    if not multi_cache:
        raise FileNotFoundError(
            "No heatmaps were created. Confirm that STORE_ALL_DATA_PATH roots and summary_result_paths are correct."
        )

    available_programs = sorted(multi_cache.keys(), key=lambda s: s.lower())
    programs_multi = resolve_programs_for_multi(available_programs, MULTI_GRAPH_PROGRAMS)

    # Select scenarios for the multi-panel figure.
    all_scenarios = sorted({sc for prog in multi_cache for sc in multi_cache[prog].keys()}, key=lambda s: s.lower())
    multi_scenarios = [sc for sc in all_scenarios if sc not in MULTI_PANEL_EXCLUDE_DISPLAY_NAMES]
    if not multi_scenarios:
        multi_scenarios = list(all_scenarios)

    # Do not truncate: show all scenarios available for the multi-panel figures.

    for program in programs_multi:
        scenario_map = multi_cache.get(program, {})
        if not scenario_map:
            continue

        # Preserve the scenario order chosen above, but only include scenarios we actually have for this program.
        scenario_to_mat: dict[str, tuple[np.ndarray, int]] = {}
        for sc in multi_scenarios:
            v = scenario_map.get(sc)
            if v is not None:
                scenario_to_mat[sc] = v

        if not scenario_to_mat:
            continue

        _plot_multi_program_heatmaps(
            program=program,
            scenario_to_mat=scenario_to_mat,
            out_png=MULTI_PROGRAM_ROOT / program.replace("/", "-") / f"heatmaps_cuts_{N_CUTS}.png",
            n_cuts=N_CUTS,
        )

        # Multi-panel unmatched-applicant histograms (same scenario selection as heatmaps)
        unmatched_map = multi_cache_unmatched.get(program, {})
        if unmatched_map:
            scenario_to_props: dict[str, tuple[np.ndarray, int]] = {}
            for sc in multi_scenarios:
                v = unmatched_map.get(sc)
                if v is not None:
                    scenario_to_props[sc] = v

            if scenario_to_props:
                _plot_multi_program_unmatched_histograms(
                    program=program,
                    scenario_to_props=scenario_to_props,
                    out_png=MULTI_PROGRAM_ROOT / program.replace("/", "-") / f"unmatched_histograms_cuts_{N_CUTS}.png",
                    n_cuts=N_CUTS,
                )

    print(f"Done. Outputs written under: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()

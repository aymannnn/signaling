import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import re
import warnings

# for the joint graph with everything together
GRAPH_OUTPUT_DIR = Path("NRMP/graphs/")
INPUT_FILES = Path("NRMP/all_data_nrmp/")

# Parameters and display names
PARAMETERS = [
    "unfilled_spots",
    "reviews_per_program",
    "pct_interview_given_signal",
    "pct_of_app_match_via_signal_given_matched",
]

PARAMETER_TITLES = [
    "Average Number of Unfilled Spots",
    "Average Number of Reviews Per Program",
    "Average Percent Interview Given Signal",
    "Average Percent of Matches from Signaled Applicants",
]

# TODO: UPDATE TO SPECIFIC PROGRAMS FOR THE HUGE GRAPH

# -------------------------------------------------------------------
# Plotting style
# -------------------------------------------------------------------

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def load_data():
    """
    Returns: list[(Path, str)] of (csv_path, label)
    """
    GRAPH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILES.exists():
        raise FileNotFoundError(
            f"INPUT_FILES directory not found: {INPUT_FILES.resolve()}\n"
            f"Either create it, or update INPUT_FILES to point at your CSV folder."
        )

    # Stable ordering so colors/legends don't randomly change between runs
    scenarios = sorted(
        [(p, p.stem) for p in INPUT_FILES.glob("*.csv")],
        key=lambda x: x[1].lower(),
    )

    if not scenarios:
        raise FileNotFoundError(
            f"No .csv files found in {INPUT_FILES.resolve()} (INPUT_FILES)."
        )

    return scenarios


def mean_and_ci(values, confidence: float = 0.95):
    """
    Compute mean and (lower, upper) confidence interval for a 1D array.

    Uses a t-distribution with df = n - 1 when n > 1.
    For n <= 1, non-finite SEM, or SEM==0, returns a degenerate CI at the mean.

    NOTE: scipy.stats.t.interval(...) can return (nan, nan) with warnings when scale=0.
    We avoid that by computing the margin explicitly.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size

    if n == 0:
        return np.nan, np.nan, np.nan

    mean = float(arr.mean())

    if n == 1:
        return mean, mean, mean

    # SEM of finite values
    sem = float(stats.sem(arr, ddof=1))
    if (not np.isfinite(sem)) or sem <= 0.0:
        return mean, mean, mean

    df = n - 1
    tcrit = float(stats.t.ppf((1.0 + confidence) / 2.0, df))
    if not np.isfinite(tcrit):
        return mean, mean, mean

    margin = tcrit * sem
    return mean, mean - margin, mean + margin


_SIGNAL_RE = re.compile(r"^-?\d+$")


def _signal_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify signal columns robustly.

    Old behavior assumed every non-'Parameter' column name is an int string.
    This version keeps only columns whose names look like integers (e.g. "0", "10"),
    and ignores common junk columns like "Unnamed: 0".
    """
    cols = []
    for c in df.columns:
        if c == "Parameter":
            continue
        name = str(c).strip()
        if _SIGNAL_RE.match(name):
            cols.append(name)
    return cols


def summarize_file(path: str):
    """
    Load a simulation_results CSV and compute mean + CI across simulations
    for each parameter and signal value.

    Returns:
        signal_values (list[int]),
        summary (dict[param] -> dict with keys: signals, mean, ci_low, ci_high)
    """
    df = pd.read_csv(path)

    # Be forgiving about column casing
    if "Parameter" not in df.columns:
        lower_map = {c.lower(): c for c in df.columns}
        if "parameter" in lower_map:
            df = df.rename(columns={lower_map["parameter"]: "Parameter"})
        else:
            raise ValueError(
                f"{Path(path).name} is missing a 'Parameter' column. "
                f"Columns found: {list(df.columns)}"
            )

    # Signal columns
    signal_cols = _signal_columns(df)
    if not signal_cols:
        raise ValueError(
            f"{Path(path).name} has no signal columns that look like integers. "
            f"(Expected columns like '0', '1', '2', ...)"
        )

    signal_values = sorted({int(c) for c in signal_cols})
    signal_strs = [str(s) for s in signal_values]

    # Ensure signal cols are numeric (coerce weird strings to NaN)
    df[signal_strs] = df[signal_strs].apply(pd.to_numeric, errors="coerce")

    summary = {}
    for param in PARAMETERS:
        param_df = df[df["Parameter"] == param]

        if param_df.empty:
            warnings.warn(
                f"{Path(path).name}: parameter '{param}' not found. "
                "Plots for this parameter will be all-NaN.",
                RuntimeWarning,
            )
            means = [np.nan] * len(signal_strs)
            ci_low = [np.nan] * len(signal_strs)
            ci_high = [np.nan] * len(signal_strs)
        else:
            means = []
            ci_low = []
            ci_high = []
            for col in signal_strs:
                m, lo, hi = mean_and_ci(param_df[col].values)
                means.append(m)
                ci_low.append(lo)
                ci_high.append(hi)

        summary[param] = {
            "signals": np.array(signal_values, dtype=int),
            "mean": np.array(means, dtype=float),
            "ci_low": np.array(ci_low, dtype=float),
            "ci_high": np.array(ci_high, dtype=float),
        }

    return signal_values, summary


def _tick_step(n: int, max_ticks: int = 20) -> int:
    """Pick an x-tick step so we show <= max_ticks ticks."""
    if n <= 0:
        return 1
    return max(1, int(np.ceil(n / max_ticks)))


# -------------------------------------------------------------------
# Main analysis for multiple files
# -------------------------------------------------------------------


def main():
    # Load and summarize each scenario
    scenario_summaries = []
    all_signal_values = set()
    scenarios = load_data()

    for csv_path, label in scenarios:
        signals, summary = summarize_file(csv_path)
        all_signal_values.update(signals)

        scenario_dir = GRAPH_OUTPUT_DIR / label
        scenario_dir.mkdir(parents=True, exist_ok=True)

        scenario_summaries.append(
            {
                "path": csv_path,
                "label": label,
                "signals": signals,
                "summary": summary,
                "directory": scenario_dir,
            }
        )

    # Build a unified sorted signal grid across all scenarios
    unified_signals = np.array(sorted(all_signal_values), dtype=int)
    sig_to_idx = {s: i for i, s in enumerate(unified_signals)}

    # Color cycle for scenarios
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(scenario_summaries))]

    # ---------------------------------------------------------------
    # 1) Per-parameter overlay plots across scenarios
    # ---------------------------------------------------------------
    for param, title in zip(PARAMETERS, PARAMETER_TITLES):
        fig, ax = plt.subplots(figsize=(6, 4))

        for idx, scenario in enumerate(scenario_summaries):
            stats_param = scenario["summary"][param]
            sigs = stats_param["signals"]
            mean = stats_param["mean"]
            lo = stats_param["ci_low"]
            hi = stats_param["ci_high"]

            # Reindex scenario data onto the unified signal grid
            full_mean = np.full_like(unified_signals, np.nan, dtype=float)
            full_lo = np.full_like(unified_signals, np.nan, dtype=float)
            full_hi = np.full_like(unified_signals, np.nan, dtype=float)

            for s, m, l, h in zip(sigs, mean, lo, hi):
                j = sig_to_idx[int(s)]
                full_mean[j] = m
                full_lo[j] = l
                full_hi[j] = h

            color = colors[idx]
            label = scenario["label"]

            ax.plot(
                unified_signals,
                full_mean,
                color=color,
                linewidth=2,
                marker="o",
                label=label,
            )
            finite_ci = np.isfinite(full_lo) & np.isfinite(full_hi)
            ax.fill_between(
                unified_signals,
                full_lo,
                full_hi,
                where=finite_ci,
                color=color,
                alpha=0.15,
            )

        ax.set_xlabel("Number of Signals")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs Number of Signals")

        step = _tick_step(len(unified_signals), max_ticks=18)
        ax.set_xticks(unified_signals[::step])

        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),   # just outside the right edge
            borderaxespad=0.0,
        )
        fig.tight_layout(rect=[0, 0, 0.78, 1])  # leave space on the right for legend
        out_png = GRAPH_OUTPUT_DIR / f"multi_{param.lower()}_overlay.png"
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

    # ---------------------------------------------------------------
    # 1b) Per-scenario plots (saved per scenario)
    # ---------------------------------------------------------------
    for scenario in scenario_summaries:
        out_dir = scenario["directory"]
        out_dir.mkdir(parents=True, exist_ok=True)

        for param, title in zip(PARAMETERS, PARAMETER_TITLES):
            stats_param = scenario["summary"][param]
            sigs = stats_param["signals"]
            mean = stats_param["mean"]
            lo = stats_param["ci_low"]
            hi = stats_param["ci_high"]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(sigs, mean, linewidth=2, marker="o")
            finite_ci = np.isfinite(lo) & np.isfinite(hi)
            ax.fill_between(sigs, lo, hi, where=finite_ci, alpha=0.15)

            ax.set_xlabel("Number of Signals")
            ax.set_ylabel(title)
            ax.set_title(f"{scenario['label']}: {title} vs Number of Signals")

            step = _tick_step(len(sigs), max_ticks=18)
            if len(sigs) > 0:
                ax.set_xticks(sigs[::step])

            ax.grid(True, alpha=0.3, linestyle="--")

            fig.tight_layout()
            out_png = out_dir / f"{param.lower()}.png"
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)

    # ---------------------------------------------------------------
    # 2) Summary bar charts: optimal signal per scenario
    #    (minimizing mean number of reviews)
    # ---------------------------------------------------------------
    scenario_labels = [s["label"] for s in scenario_summaries]

    best_signal_num_reviews = []
    for s in scenario_summaries:
        stats_num_reviews = s["summary"]["reviews_per_program"]
        sigs = stats_num_reviews["signals"]
        mean = stats_num_reviews["mean"]

        if mean.size == 0 or np.all(np.isnan(mean)):
            warnings.warn(
                f"{s['label']}: reviews_per_program is all-NaN; cannot compute optimum.",
                RuntimeWarning,
            )
            best_signal_num_reviews.append(np.nan)
            continue

        idx_min = int(np.nanargmin(mean))
        best_signal_num_reviews.append(float(sigs[idx_min]))

    best_signal_num_reviews = np.array(best_signal_num_reviews, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(scenario_labels))

    valid = np.isfinite(best_signal_num_reviews)
    if valid.any():
        ax.bar(x[valid], best_signal_num_reviews[valid], color=np.array(colors, dtype=object)[valid])
    # Mark missing values (if any)
    for xi, ok in zip(x, valid):
        if not ok:
            ax.text(xi, 0.5, "NA", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=30, ha="right")
    ax.set_ylabel("Signal Value Minimizing Number of Reviews")
    ax.set_title("Optimal Number of Signals by Scenario (Minimizing Reviews Per Program)")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    if valid.any():
        ymax = float(np.nanmax(best_signal_num_reviews)) + 2.0
        ax.set_ylim(0, ymax)

    fig.tight_layout()
    out_png = GRAPH_OUTPUT_DIR / "multi_optimal_signals_num_reviews.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

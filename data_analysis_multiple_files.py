import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# List of (CSV path, label) pairs for each simulation scenario
# Adjust paths/labels as needed.
SCENARIOS = [
    ("simulation_results/base_10_cap/base_case_10_simulation_results.csv", "AC 10"),
    ("simulation_results/base_20_cap/base_case_20_simulation_results.csv", "AC 20"),
    ("simulation_results/base_30_cap/base_case_30_simulation_results.csv", "AC 30"),
    ("simulation_results/base_50_cap/base_case_50_simulation_results.csv", "AC 50"),
    ("simulation_results/base_case/base_case_simulation_results.csv", "Base case - AC 40"),
    ("simulation_results/base_case_randomized/base_case_randomized_simulation_results.csv",
     "BC - RP/AC 40"),
]

OUTPUT_DIR = Path("simulation_results/joint_graph/")

# Parameters and display names
PARAMETERS = ["Unmatched_Applicants", "Unfilled_Spots", "Reviews_Per_Program"]
PARAMETER_TITLES = [
    "Number of Unmatched Applicants",
    "Number of Unfilled Spots",
    "Average Number of Reviews Per Program",
]

# -------------------------------------------------------------------
# Plotting style (publication-friendly)
# -------------------------------------------------------------------

plt.rcParams.update({
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
})


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def mean_and_ci(values, alpha: float = 0.95):
    """
    Compute mean and (lower, upper) confidence interval for a 1D array.

    Uses a t-distribution with df = n - 1 when n > 1.
    For n <= 1 or non-finite SEM, returns a degenerate CI at the mean.
    """
    arr = np.asarray(values, dtype=float)
    mask = ~np.isnan(arr)
    n = mask.sum()

    if n == 0:
        return np.nan, np.nan, np.nan

    mean = arr[mask].mean()

    if n == 1:
        return mean, mean, mean

    sem = stats.sem(arr[mask], ddof=1)
    if not np.isfinite(sem) or sem < 0:
        return mean, mean, mean

    ci_low, ci_high = stats.t.interval(alpha, n - 1, loc=mean, scale=sem)
    return mean, ci_low, ci_high


def summarize_file(path: str):
    """
    Load a simulation_results CSV and compute mean + CI across simulations
    for each parameter and signal value.

    Returns:
        signal_values (list[int]),
        summary (dict[param] -> dict with keys: signals, mean, ci_low, ci_high)
    """
    df = pd.read_csv(path)

    # Signal columns are everything except "Parameter"
    signal_cols = [c for c in df.columns if c != "Parameter"]
    signal_values = sorted(int(c) for c in signal_cols)
    signal_strs = [str(s) for s in signal_values]

    summary = {}
    for param in PARAMETERS:
        param_df = df[df["Parameter"] == param]

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


# -------------------------------------------------------------------
# Main analysis for multiple files
# -------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and summarize each scenario
    scenario_summaries = []
    all_signal_values = set()

    for csv_path, label in SCENARIOS:
        signals, summary = summarize_file(csv_path)
        all_signal_values.update(signals)

        scenario_summaries.append(
            {
                "path": csv_path,
                "label": label,
                "signals": signals,
                "summary": summary,
            }
        )

    # Build a unified sorted signal grid across all scenarios
    unified_signals = sorted(all_signal_values)
    unified_signals = np.array(unified_signals, dtype=int)
    sig_to_idx = {s: i for i, s in enumerate(unified_signals)}

    # Color cycle for scenarios
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(scenario_summaries))]

    # ---------------------------------------------------------------
    # 1) Per-parameter overlay plots across scenarios
    # ---------------------------------------------------------------
    # One figure per parameter, with multiple scenarios overlaid.
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
                j = sig_to_idx[s]
                full_mean[j] = m
                full_lo[j] = l
                full_hi[j] = h

            color = colors[idx]
            label = scenario["label"]

            # Matplotlib will break the line where there are NaNs,
            # so each scenario's line appears only where it has data.
            ax.plot(
                unified_signals,
                full_mean,
                color=color,
                linewidth=2,
                marker="o",
                label=label,
            )
            ax.fill_between(
                unified_signals,
                full_lo,
                full_hi,
                where=~np.isnan(full_lo) & ~np.isnan(full_hi),
                color=color,
                alpha=0.15,
            )

        ax.set_xlabel("Number of Signals")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs Number of Signals")

        # NEW: show every other signal label to reduce crowding
        ax.set_xticks(unified_signals[::2])

        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(frameon=False, ncol=2)

        fig.tight_layout()
        out_png = OUTPUT_DIR / f"multi_{param.lower()}_overlay.png"
        out_pdf = OUTPUT_DIR / f"multi_{param.lower()}_overlay.pdf"
        fig.savefig(out_png, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

    # ---------------------------------------------------------------
    # 2) Summary bar charts: optimal signal per scenario
    #    (minimizing mean number of reviews)
    # ---------------------------------------------------------------
    scenario_labels = [s["label"] for s in scenario_summaries]

    # Best signal by number of reviews
    best_signal_num_reviews = []
    for s in scenario_summaries:
        stats_num_reviews = s["summary"]["Reviews_Per_Program"]
        sigs = stats_num_reviews["signals"]
        mean = stats_num_reviews["mean"]
        idx_min = int(np.nanargmin(mean))
        best_signal_num_reviews.append(int(sigs[idx_min]))

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(scenario_labels))
    ax.bar(x, best_signal_num_reviews, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=30, ha="right")
    ax.set_ylabel("Signal Value Minimizing Number of Reviews")
    ax.set_title(
        "Optimal Number of Signals by Scenario (Minimizizing Reviews Per Program)"
    )
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    # NEW: add headroom on y-axis: max(min reviews) + 2
    if best_signal_num_reviews:
        ymax = max(best_signal_num_reviews) + 2
        ax.set_ylim(0, ymax)

    fig.tight_layout()
    out_png = OUTPUT_DIR / "multi_optimal_signals_num_reviews.png"
    out_pdf = OUTPUT_DIR / "multi_optimal_signals_num_reviews.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

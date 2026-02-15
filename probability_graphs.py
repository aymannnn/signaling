import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
RESULTS_PATH = "results/probabilistic/results.csv"
OUTPUT_DIR = "figures/program/"
# The primary scalar metrics calculated in calculate_results
METRICS_TO_PLOT = [
    'p_int_given_signal',
    'p_int_given_nosignal',
    'pct_matches_from_signal',
    'pct_match_from_nosignal',
    'reviews_per_program',
    'unfilled_positions'
]

# Readable titles for the plots
METRIC_TITLES = {
    'p_int_given_signal': 'Prob. Interview given Signal',
    'p_int_given_nosignal': 'Prob. Interview given No Signal',
    'pct_matches_from_signal': '% Class Filled by Signals',
    'pct_match_from_nosignal': '% Class Filled by Non-Signals',
    'reviews_per_program': 'Applications Reviewed per Program',
    'unfilled_positions': 'Unfilled Positions'
}


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find results file at {path}")
    return pd.read_csv(path)


def calculate_stats(df, group_cols, metric):
    """
    Calculates mean and 95% Confidence Interval for a specific metric.
    """
    # Group by program and signals
    stats = df.groupby(group_cols)[metric].agg(
        ['mean', 'count', 'std']).reset_index()

    # Calculate 95% CI
    # CI = 1.96 * (std / sqrt(n))
    stats['ci'] = 1.96 * (stats['std'] / np.sqrt(stats['count']))
    stats['lower'] = stats['mean'] - stats['ci']
    stats['upper'] = stats['mean'] + stats['ci']

    return stats


def plot_program_metrics(program_name, program_data):
    """
    Generates a 2x3 panel of plots for a single program.
    """
    # Create figure and axes
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Simulation Results: {program_name}", fontsize=16)

    axes = axes.flatten()

    # Get sorted signal values for X-axis consistency
    signal_values = sorted(program_data['signals'].unique())

    for idx, metric in enumerate(METRICS_TO_PLOT):
        ax = axes[idx]

        # Calculate stats for this specific metric
        stats = calculate_stats(program_data, ['signals'], metric)
        stats = stats.sort_values('signals')

        # Plot Mean Line
        ax.plot(stats['signals'], stats['mean'],
                marker='o', linewidth=2, label='Mean')

        # Plot 95% Confidence Interval
        ax.fill_between(
            stats['signals'],
            stats['lower'],
            stats['upper'],
            alpha=0.2,
            color='blue',
            label='95% CI'
        )

        # Formatting
        ax.set_title(METRIC_TITLES.get(metric, metric))
        ax.set_xlabel('Number of Signals Sent')
        ax.set_ylabel('Magnitude')
        ax.grid(True, linestyle='--', alpha=0.7)

        # Handle y-axis limits for percentages to keep them readable
        if 'p_int' in metric or 'pct' in metric:
            ax.set_ylim(-0.05, 1.05)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle

    # Save file
    filename = f"{program_name.replace(' ', '_').lower()}_metrics.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved figure: {save_path}")


def main():
    # 1. Setup Directories
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # 2. Load Data
    try:
        df = load_data(RESULTS_PATH)
        print(f"Loaded data with {len(df)} rows.")
    except Exception as e:
        print(e)
        return

    # 3. Process each program
    programs = df['program'].unique()

    print(f"Found {len(programs)} unique programs: {programs}")

    for program in programs:
        print(f"Processing program: {program}")
        program_data = df[df['program'] == program]
        plot_program_metrics(program, program_data)

    print("All plots generated successfully.")


if __name__ == "__main__":
    # Set style for better looking plots
    sns.set_theme(style="whitegrid")
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
import os

# --- Configuration ---

INPUT_DIRECTORY = "results/calculated/gamma_30/"
OUTPUT_DIRECTORY = "figures/gamma_30/"

# The 4 analyses you want to graph. These should match the CSV filenames (without .csv)
ANALYSES_TO_GRAPH = [
    "base",
    "random_distribution",
    "random_applicant_rank_list",
    "random_program_rank_list"
]

PROGRAMS_TO_GRAPH = [
    ("Anesthesiology", 10),
    ("Surgery (Categorical)", 12),
    ("Interventional Radiology (Integrated)", 8)
]

METRIC_NAMES = {
    'p_int_given_signal_mean': 'Interview | Signal',
    'p_int_given_nosignal_mean': 'Interview | No Signal',
    'pct_matches_from_signal_mean': 'Match from Signal',
    'pct_match_from_nosignal_mean': 'Match from No Signal',
    'unfilled_positions_mean': 'Unfilled Positions',
    'reviews_per_program_mean': 'Application Reviews',
    'expect_int_per_signal_mean': 'Expected Interviews / Signal'
}

PANEL_TITLES = [
    "Panel A: Interview & Match Rates (%)",
    "Panel B: Unfilled Positions",
    "Panel C: Workload vs. Expected Interviews",
    "Panel D: Decile Match Heatmap (Base Analysis) at Signal = {}"
]

# Style dictionaries to keep graphs readable
ANALYSIS_COLORS = sns.color_palette("tab10", len(ANALYSES_TO_GRAPH))

METRIC_STYLES_A = {
    'p_int_given_signal_mean': {'ls': '-', 'marker': 'o'},
    'p_int_given_nosignal_mean': {'ls': '--', 'marker': 'x'},
    'pct_matches_from_signal_mean': {'ls': ':', 'marker': 's'},
    'pct_match_from_nosignal_mean': {'ls': '-.', 'marker': 'd'}
}


def create_4_panel_graph(data_dict, program_name, decile_signal):
    """
    Generates and saves a 4-panel figure comparing multiple analyses for a specific program.
    Panel D exclusively shows the 'base' analysis.
    """
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"{program_name} - Cross-Analysis Comparison",
                 fontsize=22, fontweight='bold')

    # Use GridSpec for clean layout control
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c1 = fig.add_subplot(gs[1, 0])
    ax_c2 = ax_c1.twinx()
    # Reverted to standard subplot for Panel D
    ax_d = fig.add_subplot(gs[1, 1])

    # Track handles for custom legends
    analysis_legend_elements = [Line2D([0], [0], color=ANALYSIS_COLORS[i], lw=3, label=ana)
                                for i, ana in enumerate(ANALYSES_TO_GRAPH)]

    # ---------------------------------------------------------
    # Panel A: Probabilities/Percentages (Interview & Match)
    # ---------------------------------------------------------
    for i, analysis in enumerate(ANALYSES_TO_GRAPH):
        if analysis not in data_dict:
            continue
        prog_df = data_dict[analysis][data_dict[analysis]
                                      ['program'] == program_name].sort_values('signals')
        if prog_df.empty:
            continue

        color = ANALYSIS_COLORS[i]
        for metric, style in METRIC_STYLES_A.items():
            if metric in prog_df.columns:
                ax_a.plot(prog_df['signals'], prog_df[metric] * 100, color=color,
                          linestyle=style['ls'], marker=style['marker'], markersize=5, alpha=0.8)

    ax_a.set_title(PANEL_TITLES[0], fontsize=14)
    ax_a.set_xlabel("Signals")
    ax_a.set_ylabel("Percentage (%)")
    ax_a.grid(True, linestyle='--', alpha=0.6)

    # Custom Legend for Panel A Metrics
    metric_legend_elements = [Line2D([0], [0], color='black', lw=2, linestyle=style['ls'],
                                     marker=style['marker'], label=METRIC_NAMES[metric])
                              for metric, style in METRIC_STYLES_A.items()]

    leg1 = ax_a.legend(handles=analysis_legend_elements,
                       loc='upper left', title="Analyses", fontsize=9)
    ax_a.add_artist(leg1)
    ax_a.legend(handles=metric_legend_elements,
                loc='upper right', title="Metrics", fontsize=9)

    # ---------------------------------------------------------
    # Panel B: Unfilled Positions
    # ---------------------------------------------------------
    metric_b = 'unfilled_positions_mean'
    for i, analysis in enumerate(ANALYSES_TO_GRAPH):
        if analysis not in data_dict:
            continue
        prog_df = data_dict[analysis][data_dict[analysis]
                                      ['program'] == program_name].sort_values('signals')
        if prog_df.empty:
            continue

        color = ANALYSIS_COLORS[i]
        if metric_b in prog_df.columns:
            ax_b.plot(prog_df['signals'], prog_df[metric_b],
                      color=color, marker='s', markersize=5)

            # Shaded confidence interval if available
            if 'unfilled_positions_lower' in prog_df.columns:
                ax_b.fill_between(prog_df['signals'], prog_df['unfilled_positions_lower'],
                                  prog_df['unfilled_positions_upper'], color=color, alpha=0.15)

    ax_b.set_title(PANEL_TITLES[1], fontsize=14)
    ax_b.set_xlabel("Signals")
    ax_b.set_ylabel("Count")
    ax_b.grid(True, linestyle='--', alpha=0.6)
    ax_b.legend(handles=analysis_legend_elements, loc='best')

    # ---------------------------------------------------------
    # Panel C: Reviews vs. Expected Interviews (Dual Axis)
    # ---------------------------------------------------------
    metric_c1 = 'reviews_per_program_mean'
    metric_c2 = 'expect_int_per_signal_mean'

    for i, analysis in enumerate(ANALYSES_TO_GRAPH):
        if analysis not in data_dict:
            continue
        prog_df = data_dict[analysis][data_dict[analysis]
                                      ['program'] == program_name].sort_values('signals')
        if prog_df.empty:
            continue

        color = ANALYSIS_COLORS[i]

        if metric_c1 in prog_df.columns:
            ax_c1.plot(prog_df['signals'], prog_df[metric_c1],
                       color=color, linestyle='-', marker='^', alpha=0.8)
        if metric_c2 in prog_df.columns:
            ax_c2.plot(prog_df['signals'], prog_df[metric_c2],
                       color=color, linestyle='--', marker='v', alpha=0.8)

    ax_c1.set_ylabel(METRIC_NAMES[metric_c1])
    ax_c2.set_ylabel(METRIC_NAMES[metric_c2])
    ax_c1.set_title(PANEL_TITLES[2], fontsize=14)
    ax_c1.set_xlabel("Signals")
    ax_c1.grid(True, linestyle='--', alpha=0.6)

    c_style_elements = [
        Line2D([0], [0], color='black', linestyle='-',
               marker='^', label=METRIC_NAMES[metric_c1]),
        Line2D([0], [0], color='black', linestyle='--',
               marker='v', label=METRIC_NAMES[metric_c2])
    ]
    ax_c1.legend(handles=c_style_elements, loc='upper left', title="Metrics")
    ax_c2.legend(handles=analysis_legend_elements,
                 loc='upper right', title="Analyses")

    # ---------------------------------------------------------
    # Panel D: Decile Match Matrix (Base Analysis Only)
    # ---------------------------------------------------------
    ax_d.set_title(PANEL_TITLES[3].format(decile_signal), fontsize=14)

    if 'base' in data_dict:
        prog_df = data_dict['base'][data_dict['base']
                                    ['program'] == program_name]
        point_data = prog_df[prog_df['signals'] == decile_signal]

        if point_data.empty:
            ax_d.text(
                0.5, 0.5, f"No data found for signals = {decile_signal}", ha='center', va='center')
        else:
            decile_matrix = np.zeros((10, 10))
            for p in range(1, 11):
                for a in range(1, 11):
                    col_name = f'p{p}_a{a}'
                    if col_name in point_data.columns:
                        decile_matrix[p-1, a-1] = point_data.iloc[0][col_name]

            sns.heatmap(decile_matrix, ax=ax_d, cmap="YlGnBu", annot=False,
                        vmin=0, cbar_kws={'label': 'Match Probability'})

            # Format heatmap axes to show 1-10 cleanly
            ax_d.set_xticks(np.arange(10) + 0.5)
            ax_d.set_xticklabels(range(1, 11))
            ax_d.set_yticks(np.arange(10) + 0.5)
            ax_d.set_yticklabels(range(1, 11))

            ax_d.set_xlabel("Applicant Decile", fontsize=12)
            ax_d.set_ylabel("Program Decile", fontsize=12)
    else:
        ax_d.text(0.5, 0.5, "'base' analysis not found in data",
                  ha='center', va='center')

    # ---------------------------------------------------------
    # Save & Cleanup
    # ---------------------------------------------------------
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    clean_prog_name = program_name.replace(' ', '_')
    filename = f"{clean_prog_name}_{decile_signal}.png"
    save_path = os.path.join(OUTPUT_DIRECTORY, filename)

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved cross-analysis panel: {save_path}")
    plt.close(fig)


def main():
    # Load all requested analyses into a dictionary of DataFrames
    data_dict = {}
    for analysis in ANALYSES_TO_GRAPH:
        file_path = os.path.join(INPUT_DIRECTORY, f"{analysis}.csv")
        if os.path.exists(file_path):
            data_dict[analysis] = pd.read_csv(file_path)
            print(f"Loaded: {analysis}.csv")
        else:
            print(f"Warning: Could not find {file_path}")

    if not data_dict:
        print(
            "Error: No data files found. Check your INPUT_DIRECTORY and ANALYSES_TO_GRAPH.")
        return

    # Generate and save graphs for each program specified
    for program_name, decile_signal in PROGRAMS_TO_GRAPH:
        print(
            f"Generating panels for: {program_name} (Decile Signal: {decile_signal})")
        create_4_panel_graph(data_dict, program_name, decile_signal)

    print("All multi-analysis graphs successfully processed and saved!")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from individual_graphs import METRICS_TO_PLOT, METRIC_TITLES, calculate_stats, load_data

ANALYSES = [
    "base",
    "random_distribution",
    "random_applicant_rank_list",
    "random_program_rank_list",
    "random_all"
]

def plot_combined_scenario_graphs(program_name: str, output_base_dir: str):
    """
    Generates a 2x3 panel of plots for a single program, with each plot
    containing the results from all 5 analysis scenarios.
    """
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle(f"Combined Analysis Results: {program_name}", fontsize=20)
    axes = axes.flatten()

    for idx, metric in enumerate(METRICS_TO_PLOT):
        ax = axes[idx]
        
        for analysis_name in ANALYSES:
            try:
                results_filepath = os.path.join("results", f"results_{analysis_name}.csv")
                df = load_data(results_filepath)
                program_data = df[df['program'] == program_name]

                if program_data.empty:
                    continue

                stats = calculate_stats(program_data, ['signals'], metric)
                stats = stats.sort_values('signals')

                ax.plot(stats['signals'], stats['mean'], marker='o', linewidth=2, label=analysis_name)
                ax.fill_between(
                    stats['signals'],
                    stats['lower'],
                    stats['upper'],
                    alpha=0.2,
                    label=f'{analysis_name} 95% CI'
                )
            except FileNotFoundError:
                print(f"Could not find results for {analysis_name}")
                continue

        ax.set_title(METRIC_TITLES.get(metric, metric))
        ax.set_xlabel('Number of Signals Sent')
        ax.set_ylabel('Magnitude')
        ax.grid(True, linestyle='--', alpha=0.7)
        if 'p_int' in metric or 'pct' in metric:
            ax.set_ylim(-0.05, 1.05)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = f"{program_name.replace(' ', '_').lower()}.png"
    save_path = os.path.join(output_base_dir, filename)
    os.makedirs(output_base_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved combined figure: {save_path}")

def plot_dual_axis_program_graphs(program_names: list, output_base_dir: str):
    """
    Generates a figure with one subplot per program, each showing two metrics
    on a dual-Y axis, for all 5 analysis scenarios.
    """
    fig, axes = plt.subplots(len(program_names), 1, figsize=(15, 8 * len(program_names)), sharex=True)
    if len(program_names) == 1:
        axes = [axes]
    
    fig.suptitle("Specific Program Analysis: Prob. Interview vs. Reviews", fontsize=20)

    for i, program_name in enumerate(program_names):
        ax1 = axes[i]
        ax2 = ax1.twinx()

        for analysis_name in ANALYSES:
            try:
                results_filepath = os.path.join("results", f"results_{analysis_name}.csv")
                df = load_data(results_filepath)
                program_data = df[df['program'] == program_name]

                if program_data.empty:
                    continue

                # Plot for p_int_given_signal on ax1
                stats1 = calculate_stats(program_data, ['signals'], 'p_int_given_signal')
                stats1 = stats1.sort_values('signals')
                line1 = ax1.plot(stats1['signals'], stats1['mean'], marker='o', label=f"{analysis_name} (Prob. Int.)")
                ax1.fill_between(stats1['signals'], stats1['lower'], stats1['upper'], color=line1[0].get_color(), alpha=0.2)

                # Plot for reviews_per_program on ax2
                stats2 = calculate_stats(program_data, ['signals'], 'reviews_per_program')
                stats2 = stats2.sort_values('signals')
                line2 = ax2.plot(stats2['signals'], stats2['mean'], marker='x', linestyle='--', label=f"{analysis_name} (Reviews)")
                ax2.fill_between(stats2['signals'], stats2['lower'], stats2['upper'], color=line2[0].get_color(), alpha=0.2)

            except FileNotFoundError:
                print(f"Could not find results for {analysis_name}")
                continue
        
        ax1.set_title(program_name)
        ax1.set_xlabel('Number of Signals Sent')
        ax1.set_ylabel('Prob. Interview given Signal', color='blue')
        ax2.set_ylabel('Applications Reviewed per Program', color='red')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax1.grid(True, linestyle='--', alpha=0.7)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = "specific_programs_dual_axis.png"
    save_path = os.path.join(output_base_dir, filename)
    os.makedirs(output_base_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved dual axis figure: {save_path}")

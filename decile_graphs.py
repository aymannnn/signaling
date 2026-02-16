import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from individual_graphs import load_data

ANALYSES = [
    "base",
    "random_distribution",
    "random_applicant_rank_list",
    "random_program_rank_list",
    "random_all"
]

def plot_decile_heatmap_for_program(analysis_name: str, program_name: str, output_base_dir: str):
    """
    Generates a decile heatmap for a single program at the signal value
    that minimizes the reviews per program.
    """
    try:
        results_filepath = os.path.join("results", f"results_{analysis_name}.csv")
        df = load_data(results_filepath)
    except FileNotFoundError:
        print(f"Could not find results for {analysis_name}, skipping decile plot for {program_name}.")
        return

    program_data = df[df['program'] == program_name]
    if program_data.empty:
        print(f"No data for program {program_name} in {analysis_name}, skipping decile plot.")
        return

    # Find the signal value that minimizes reviews_per_program
    reviews_by_signal = program_data.groupby('signals')['reviews_per_program'].mean()
    if reviews_by_signal.empty:
        print(f"No review data for program {program_name} in {analysis_name}, skipping.")
        return
        
    optimal_signals = reviews_by_signal.idxmin()

    # Filter data for the optimal signal value
    optimal_data = program_data[program_data['signals'] == optimal_signals]
    if optimal_data.empty:
        print(f"No data at optimal signal count for {program_name} in {analysis_name}, skipping.")
        return

    # Create a 10x10 matrix for the heatmap
    decile_matrix = pd.DataFrame(np.zeros((10, 10)),
                                 index=[f'P{i}' for i in range(1, 11)],
                                 columns=[f'A{i}' for i in range(1, 11)])

    for i in range(1, 11):  # Program decile
        for j in range(1, 11):  # Applicant decile
            col_name = f'p{i}_a{j}'
            if col_name in optimal_data:
                decile_matrix.loc[f'P{i}', f'A{j}'] = optimal_data[col_name].mean()

    # Plotting
    plt.figure(figsize=(12, 10))
    sns.heatmap(decile_matrix, annot=True, fmt=".2f", cmap="viridis", linewidths=.5)
    
    plt.title(f'Match Distribution: Program vs. Applicant Decile for {program_name}\nAnalysis: {analysis_name}, Signals: {optimal_signals}')
    plt.xlabel('Applicant Decile')
    plt.ylabel('Program Decile')
    
    # Save file
    filename = f"{program_name.replace(' ', '_').lower()}_decile_heatmap.png"
    save_path = os.path.join(output_base_dir, filename)
    os.makedirs(output_base_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved decile heatmap: {save_path}")

def plot_combined_decile_heatmap(program_name: str, output_base_dir: str):
    """
    Generates a 2x3 panel of decile heatmaps for a single program,
    comparing the 5 analysis scenarios.
    """
    fig, axes = plt.subplots(2, 3, figsize=(21, 14), sharey=True, sharex=True)
    fig.suptitle(f'Match Distribution Comparison for {program_name} at Min Reviews', fontsize=20)
    axes = axes.flatten()
    
    # Hide the 6th unused subplot
    fig.delaxes(axes[5])

    for i, analysis_name in enumerate(ANALYSES):
        ax = axes[i]
        try:
            results_filepath = os.path.join("results", f"results_{analysis_name}.csv")
            df = load_data(results_filepath)
        except FileNotFoundError:
            ax.set_title(f"{analysis_name}\n(No data)")
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        program_data = df[df['program'] == program_name]
        if program_data.empty:
            ax.set_title(f"{analysis_name}\n(No data for program)")
            ax.text(0.5, 0.5, 'No data for program', ha='center', va='center', transform=ax.transAxes)
            continue

        reviews_by_signal = program_data.groupby('signals')['reviews_per_program'].mean()
        if reviews_by_signal.empty:
            ax.set_title(f"{analysis_name}\n(No review data)")
            continue

        optimal_signals = reviews_by_signal.idxmin()
        optimal_data = program_data[program_data['signals'] == optimal_signals]

        decile_matrix = pd.DataFrame(np.zeros((10, 10)),
                                     index=[f'P{i}' for i in range(1, 11)],
                                     columns=[f'A{i}' for i in range(1, 11)])

        for r in range(1, 11):
            for c in range(1, 11):
                col_name = f'p{r}_a{c}'
                if col_name in optimal_data:
                    decile_matrix.loc[f'P{r}', f'A{c}'] = optimal_data[col_name].mean()

        sns.heatmap(decile_matrix, ax=ax, annot=True, fmt=".2f", cmap="viridis", cbar= (i==2 or i==4))
        ax.set_title(f'{analysis_name}\n(Signals: {optimal_signals})')

    fig.text(0.5, 0.04, 'Applicant Decile', ha='center', va='center', fontsize=14)
    fig.text(0.07, 0.5, 'Program Decile', ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(rect=[0.08, 0.05, 1, 0.95])
    
    # Save file
    filename = f"{program_name.replace(' ', '_').lower()}_decile_heatmap_comparison.png"
    save_path = os.path.join(output_base_dir, filename)
    os.makedirs(output_base_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved combined decile heatmap for {program_name}: {save_path}")

def generate_all_decile_graphs():
    """
    Main function to generate all decile graphs.
    """
    sns.set_theme()
    # Individual decile plots for all programs in each analysis
    for analysis in ANALYSES:
        print(f"Generating individual decile heatmaps for '{analysis}' analysis...")
        try:
            results = pd.read_csv(f"results/results_{analysis}.csv")
            all_programs = results['program'].unique()
            output_dir = f"figures/{analysis}"
            for program_name in all_programs:
                plot_decile_heatmap_for_program(analysis, program_name, output_dir)
        except FileNotFoundError:
            print(f"Could not find results for {analysis}, skipping.")
        print("-" * 50)

    # Combined 5-panel decile plot for a specific program
    print(f"Generating combined decile heatmaps for all programs...")
    output_dir_combined = "figures/joint/combined"
    try:
        base_results = pd.read_csv("results/results_base.csv")
        all_programs = base_results['program'].unique()
        for program_name in all_programs:
            plot_combined_decile_heatmap(program_name, output_dir_combined)
    except FileNotFoundError:
        print("Could not find results/results_base.csv, skipping combined heatmaps.")
    print("-" * 50)
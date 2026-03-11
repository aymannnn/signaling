import pandas as pd
import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
# The subdirectory for the specific model output we want to process
INPUT_DIRECTORY = f"results/model_output/gamma_72/"
OUTPUT_DIRECTORY = f"results/calculated/gamma_72/"

# Scenarios defined in the project configuration
ANALYSES = [
    "base",
    "random_distribution",
    "random_applicant_rank_list",
    "random_program_rank_list",
    "random_all"
]

# Core metrics used in individual, combined, and dual-axis graphs
CORE_METRICS = [
    'p_int_given_signal',
    'p_int_given_nosignal',
    'pct_matches_from_signal',
    'pct_match_from_nosignal',
    'reviews_per_program',
    'unfilled_positions'
]

# Generate list of all decile match columns (p1_a1 ... p10_a10)
# These represent the match probability between program decile (P) and applicant decile (A)
DECILE_METRICS = [f'p{p}_a{a}' for p in range(1, 11) for a in range(1, 11)]


def process_analysis_file(analysis_name):
    """
    Reads a raw simulation result file and calculates summary statistics.
    """
    file_path = os.path.join(INPUT_DIRECTORY, f"results_{analysis_name}.csv")

    if not os.path.exists(file_path):
        print(f"Skipping {analysis_name}: File not found at {file_path}")
        return None

    print(f"Processing {analysis_name}...")

    # Load raw data
    df = pd.read_csv(file_path)

    # 1. Define aggregation map
    agg_map = {metric: ['mean', 'std', 'count'] for metric in CORE_METRICS}
    for d_metric in DECILE_METRICS:
        if d_metric in df.columns:
            agg_map[d_metric] = ['mean']

    # 2. Execute GroupBy
    stats = df.groupby(['program', 'signals']).agg(agg_map)

    # 3. Flatten MultiIndex columns and clean up names
    new_cols = []
    for col, stat in stats.columns:
        if col in CORE_METRICS:
            new_cols.append(f"{col}_{stat}")
        else:
            new_cols.append(col)

    stats.columns = new_cols

    # FIX: Create a clean, contiguous memory block BEFORE resetting the index.
    # This gives reset_index() a de-fragmented frame to work with.
    stats = stats.copy()
    stats = stats.reset_index()

    # 4. Calculate 95% Confidence Intervals for core metrics
    # Collect new columns in a dictionary to prevent fragmentation
    new_columns = {}

    for metric in CORE_METRICS:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        count_col = f"{metric}_count"

        # Standard Error of the Mean (SEM) * 1.96 for 95% CI
        ci_val = 1.96 * (stats[std_col] / np.sqrt(stats[count_col]))

        new_columns[f"{metric}_ci"] = ci_val
        new_columns[f"{metric}_lower"] = stats[mean_col] - ci_val
        new_columns[f"{metric}_upper"] = stats[mean_col] + ci_val

    # 5. Derived Metric: Expected Interviews from Signals
    # Calculation: number of signals * probability of interview given signal
    new_columns['expect_int_per_signal_mean'] = stats['signals'] * stats['p_int_given_signal_mean']
    new_columns['expect_int_per_signal_ci'] = stats['signals'] * new_columns['p_int_given_signal_ci']
    new_columns['expect_int_per_signal_lower'] = stats['signals'] * new_columns['p_int_given_signal_lower']
    new_columns['expect_int_per_signal_upper'] = stats['signals'] * new_columns['p_int_given_signal_upper']

    # Concatenate the original stats DataFrame with the new columns all at once
    stats = pd.concat([stats, pd.DataFrame(new_columns)], axis=1)

    return stats

def create_max_min_report(all_summaries: dict):
    master_df = None

    for analysis_name, df in all_summaries.items():
        # 1. Minimum reviews per program (find the signal count)
        min_idx = df.groupby('program')['reviews_per_program_mean'].idxmin()
        min_signal_df = df.loc[min_idx, ['program', 'signals']].copy()
        min_signal_df.columns = ['program', f'Optimal Signals (Min Reviews) - {analysis_name}']

        # 2. Maximum expected interviews (find the signal count)
        max_idx = df.groupby('program')['expect_int_per_signal_mean'].idxmax()
        max_signal_df = df.loc[max_idx, ['program', 'signals']].copy()
        max_signal_df.columns = ['program', f'Optimal Signals (Max Exp Int) - {analysis_name}']

        # Combine analysis-specific results
        analysis_summary = pd.merge(max_signal_df, min_signal_df, on='program')

        # Merge into the master report
        if master_df is None:
            master_df = analysis_summary
        else:
            master_df = pd.merge(master_df, analysis_summary, on='program')

    return master_df

def main():
    """
    Main execution loop: creates the output directory and processes all analysis scenarios.
    """
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
        print(f"Created output directory: {OUTPUT_DIRECTORY}")

    all_summaries = {}

    for analysis in ANALYSES:
        summary_df = process_analysis_file(analysis)
        
        if summary_df is not None:
            # Export individual summarized data to CSV
            output_path = os.path.join(OUTPUT_DIRECTORY, f"{analysis}.csv")
            summary_df.to_csv(output_path, index=False)
            print(f"Saved individual summary to {output_path}")
            
            # Store for the master report
            all_summaries[analysis] = summary_df
            print("-" * 30)

    # Generate and save the master summary report
    if all_summaries:
        master_report = create_max_min_report(all_summaries)
        master_output_path = os.path.join(OUTPUT_DIRECTORY, "max_min_signals.csv")
        master_report.to_csv(master_output_path, index=False)
        print(f"Successfully saved master summary report to {master_output_path}")

if __name__ == "__main__":
    main()

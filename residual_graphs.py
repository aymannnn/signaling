import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---

# File paths as requested
IMPORT_PATH = "results/calculated/gamma_30/"
EXPORT_PATH = "figures/gamma_30/"

# Sensitivity analyses to include
ANALYSES = [
    "base",
    "random_distribution",
    "random_applicant_rank_list",
    "random_program_rank_list",
    "random_all"
]


def calculate_residuals(df, analysis_name):
    """
    Calculates the required differences (residuals) for each program in the given dataframe.
    """
    results = []

    for program, group in df.groupby('program'):
        # 1. Find the signal value where reviews_per_program_mean is minimized
        min_rev_idx = group['reviews_per_program_mean'].idxmin()
        s_min_rev = group.loc[min_rev_idx, 'signals']
        min_reviews = group.loc[min_rev_idx, 'reviews_per_program_mean']

        # 2. Find the signal value where expect_int_per_signal_mean is maximized
        max_exp_idx = group['expect_int_per_signal_mean'].idxmax()
        s_max_exp = group.loc[max_exp_idx, 'signals']
        max_expected_interviews = group.loc[max_exp_idx,
                                            'expect_int_per_signal_mean']

        # 3. Find reviews per program AT the signal value of maximum expected interviews
        rev_at_s_max_exp = group[group['signals'] ==
                                 s_max_exp]['reviews_per_program_mean'].values[0]

        # 4. Find expected interviews AT the signal value of minimum reviews
        exp_int_at_s_min_rev = group[group['signals'] ==
                                     s_min_rev]['expect_int_per_signal_mean'].values[0]

        # 5. Calculate metrics
        signal_distance = s_max_exp - s_min_rev

        # Middle Panel (B): Relative percent INCREASE in reviews if maximizing interviews
        if min_reviews > 0:
            rel_increase_rev = (
                (rev_at_s_max_exp - min_reviews) / min_reviews) * 100
        else:
            rel_increase_rev = 0

        # Right Panel (C): Relative percent LOSS in interviews if minimizing reviews
        if max_expected_interviews > 0:
            loss_at_min_rev = max_expected_interviews - exp_int_at_s_min_rev
            rel_loss_min_rev = (
                loss_at_min_rev / max_expected_interviews) * 100
        else:
            rel_loss_min_rev = 0

        results.append({
            'program': program,
            'analysis': analysis_name,
            'signal_distance': signal_distance,
            'rel_increase_rev': rel_increase_rev,
            'rel_loss_min_rev': rel_loss_min_rev
        })

    return pd.DataFrame(results)


def main():
    all_data = []

    for analysis in ANALYSES:
        file_path = IMPORT_PATH + f"{analysis}.csv"

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            processed_df = calculate_residuals(df, analysis)
            all_data.append(processed_df)
            print(f"Processed: {analysis}.csv")
        else:
            print(f"Warning: Could not find data at {file_path}")

    if not all_data:
        print("Error: No data files found. Check your IMPORT_PATH and available CSVs.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Sort the dataframe alphabetically by program so the Y-axis is consistently ordered
    combined_df = combined_df.sort_values(by='program', ascending=False)

    # --- Plotting ---
    # Setup tall/vertical 3-panel figure with shared Y-axis
    fig, axes = plt.subplots(1, 3, figsize=(20, 18), sharey=True)

    # Shifted 'y' higher to add white space between the suptitle and the subplots
    fig.suptitle("Trade-offs: Minimizing Application Reviews vs. Maximizing Expected Interviews",
                 fontsize=22, fontweight='bold', y=0.96)

    palette = sns.color_palette("tab10", len(ANALYSES))

    # --- Panel 1 (Left Panel) ---
    ax1 = axes[0]
    sns.scatterplot(
        data=combined_df,
        x='signal_distance',
        y='program',
        hue='analysis',
        palette=palette,
        s=90,
        alpha=0.8,
        ax=ax1,
        zorder=3
    )

    ax1.set_title("A. Signal Distance:\nMin Review vs. Max Expected Int.",
                  fontsize=15, fontweight='bold', pad=20)
    ax1.set_xlabel("Difference (Signals)", fontsize=12)
    ax1.set_ylabel("Program", fontsize=14)
    ax1.axvline(0, color='black', linestyle='--', linewidth=1.5, zorder=2)
    ax1.grid(True, axis='x', linestyle=':', alpha=0.6)

    # --- Panel 2 (Middle Panel) ---
    ax2 = axes[1]
    sns.scatterplot(
        data=combined_df,
        x='rel_increase_rev',
        y='program',
        hue='analysis',
        palette=palette,
        s=90,
        alpha=0.8,
        ax=ax2,
        zorder=3
    )

    ax2.set_title("B. Relative Percent Increase in\nReviews Given Maximized Interviews",
                  fontsize=15, fontweight='bold', pad=20)
    ax2.set_xlabel("Percent Increase (%)", fontsize=12)
    ax2.set_ylabel("")
    ax2.grid(True, axis='x', linestyle=':', alpha=0.6)

    # --- Panel 3 (Right Panel) ---
    ax3 = axes[2]
    sns.scatterplot(
        data=combined_df,
        x='rel_loss_min_rev',
        y='program',
        hue='analysis',
        palette=palette,
        s=90,
        alpha=0.8,
        ax=ax3,
        zorder=3
    )

    ax3.set_title("C. Relative Percent Loss of Expected\nInterviews Given Minimized Reviews",
                  fontsize=15, fontweight='bold', pad=20)
    ax3.set_xlabel("Percent Loss (%)", fontsize=12)
    ax3.set_ylabel("")
    ax3.grid(True, axis='x', linestyle=':', alpha=0.6)

    # --- Alternating Background Shading ---
    unique_programs = combined_df['program'].unique()

    # Apply shading row by row across all three axes
    for i, prog in enumerate(unique_programs):
        if i % 2 == 0:
            ax1.axhspan(i - 0.5, i + 0.5, color='lightgray',
                        alpha=0.25, zorder=1)
            ax2.axhspan(i - 0.5, i + 0.5, color='lightgray',
                        alpha=0.25, zorder=1)
            ax3.axhspan(i - 0.5, i + 0.5, color='lightgray',
                        alpha=0.25, zorder=1)

    # Cap the y-limits so the shading doesn't spill past the top/bottom programs
    ax1.set_ylim(-0.5, len(unique_programs) - 0.5)

    # Adjust legends to be clean and non-overlapping
    if ax1.get_legend():
        ax1.get_legend().remove()
    if ax2.get_legend():
        ax2.get_legend().remove()

    ax3.legend(title="Sensitivity Analysis", fontsize=11,
               loc='center left', bbox_to_anchor=(1.02, 0.5))

    # Adjust subplot tops and minimize horizontal space to make it look clean
    plt.subplots_adjust(top=0.88, wspace=0.05)

    # Save Figure
    save_path = EXPORT_PATH + "residual_analysis_vertical.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved graph successfully to: {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()

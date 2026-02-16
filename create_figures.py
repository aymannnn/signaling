import os
import pandas as pd
from probability_graphs import create_program_graphs_for_analysis
from combined_graphs import plot_combined_scenario_graphs, plot_dual_axis_program_graphs

GRAPH_INDIVIDUAL_ANALYSES = False
GRAPH_COMBINED_SCENARIOS = False
GRAPH_DUAL_AXIS_PROGRAMS = True

DUAL_AXIS_PROGRAMS = [
    "Surgery (Categorical)",
    "Internal Medicine (Categorical)",
    "Vascular Surgery"
]

def main():
    analyses = [
        "base",
        "random_distribution",
        "random_applicant_rank_list",
        "random_program_rank_list",
        "random_all"
    ]

    if GRAPH_INDIVIDUAL_ANALYSES:
        
        # Original individual analysis graphs
        for analysis_name in analyses:
            results_filepath = os.path.join("results", f"results_{analysis_name}.csv")
            output_base_dir = os.path.join("figures", analysis_name)
            
            print(f"Generating graphs for analysis: {analysis_name}")
            create_program_graphs_for_analysis(results_filepath, output_base_dir)
            print("-" * 50)

    if GRAPH_COMBINED_SCENARIOS:
        # New combined scenario graphs for all programs
        print("Generating combined scenario graphs for all programs...")
        try:
            base_results = pd.read_csv("results/results_base.csv")
            all_programs = base_results['program'].unique()
            
            output_dir_combined = "figures/joint/combined"
            for program_name in all_programs:
                plot_combined_scenario_graphs(program_name, output_dir_combined)

        except FileNotFoundError:
            print("Could not find results/results_base.csv, skipping combined scenario graphs.")
        print("-" * 50)

    if GRAPH_DUAL_AXIS_PROGRAMS:
        # New dual-axis graphs for specific programs
        print("Generating dual-axis graphs for specific programs...")
        specific_programs = DUAL_AXIS_PROGRAMS
        output_dir_dual_axis = "figures/joint/"
        plot_dual_axis_program_graphs(specific_programs, output_dir_dual_axis)
        print("-" * 50)


if __name__ == "__main__":
    main()

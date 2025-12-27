import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


SUMMARY_DATA_PATH = Path("heatmap_results/heatmap_results.csv")
OUTPUT_DIRECTORY = Path("machine_learning_model/")
ALL_DATA_DIRECTORY = Path("heatmap_results/all_data/")

def generate_whole_matrix(input_path = SUMMARY_DATA_PATH, 
                          export = True):
    summary_df = pd.read_csv(input_path)

    df_characteristics = [
        'n_programs',
        # 'n_positions', # should probably  use the simulated positions
        'simulated_positions',
        'n_applicants',
        'interviews_per_spot',
        'max_applications',
        'result_file_prefix'  # for joining later in our big dataframe
    ]
    summary_df = summary_df[df_characteristics]

    final_columns = df_characteristics + [
        'signal_value',  # dependent
        'reviews'  # outcome
    ]

    final_dataframe = pd.DataFrame(columns=final_columns)

    # construct data matrix, will have to expand the dataframe to include all data ...

    for file in ALL_DATA_DIRECTORY.iterdir():
        if not file.name.endswith('.csv'):
            continue
        scenario = file.stem
        scenario_characteristics = summary_df[
            summary_df['result_file_prefix'] == scenario]
        scenario_data = pd.read_csv(file)
        # scenario data is in format parameter, signal values
        scenario_data = scenario_data[scenario_data['Parameter']
                                    == 'reviews_per_program']
        signal_cols = scenario_data.columns.tolist()
        signal_cols.remove('Parameter')
        for signal in signal_cols:
            dataframe_row = scenario_characteristics.copy()
            dataframe_row['signal_value'] = int(signal)
            dataframe_row['reviews'] = scenario_data[signal].mean()
            final_dataframe = pd.concat(
                [final_dataframe, dataframe_row], ignore_index=True)
            
    if export:
        final_dataframe.to_csv(
            OUTPUT_DIRECTORY / 'full_data_matrix.csv', index=False)
    else:
        return final_dataframe
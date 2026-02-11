import pandas as pd
from pathlib import Path

# the purpose of this file is to generate the full data matrix for machine
# learning, which looks different then the summary data frame
# specifically, summary doesn't have info for all signal values

ALL_DATA_BASE_CASE = 'results/all_data_base_case/'
BASE_CASE_SUMMARY = 'results/base_case.csv'

ALL_DATA_LOCAL = 'results/all_data_local_analysis/'
LOCAL_SUMMARY = 'results/nrmp_local_analysis_results.csv'

OUTPUT_DIRECTORY = Path("machine_learning_model/")

def generate_whole_matrix(result_summary_path,
                          all_result_directory):

    summary_df = pd.read_csv(result_summary_path)

    df_characteristics = [
        'n_programs',
        'n_positions',
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

    for file in all_result_directory.iterdir():
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

    return final_dataframe


if __name__ == "__main__":
    base_case_df = generate_whole_matrix(
        BASE_CASE_SUMMARY,
        Path(ALL_DATA_BASE_CASE)
    )
    base_case_df.to_csv(
        OUTPUT_DIRECTORY / 'full_data_matrix_base_case.csv',
        index=False
    )
    
    local_df = generate_whole_matrix(
        LOCAL_SUMMARY,
        Path(ALL_DATA_LOCAL)
    )
    local_df.to_csv(
        OUTPUT_DIRECTORY / 'full_data_matrix_local.csv',
        index=False
    )
    
    # combine both dataframes
    full_df = pd.concat([base_case_df, local_df], ignore_index=True)
    full_df.to_csv(
        OUTPUT_DIRECTORY / 'full_data_matrix.csv',
        index=False
    )
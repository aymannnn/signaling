import numpy as np
import pandas as pd
import os


EXPORT_FILE_PATH = 'constants_heatmap/randomized_constants.csv'
NUMBER_TO_GENERATE = 10000
SIMULATIONS_PER_S = 20

# use as a final upper bound, note that max applications 
# is minimum(programs, number below)
MAXIMUM_APPLICATIONS = 75  
# minimum applications is just that, must have >5 programs

MINIMUM_APPLICATIONS = 5
# signals will be from 3 to max applications - 3

# https: // www.nrmp.org/wp-content/uploads/2025/05/Main_Match_Results_and_Data_20250529_FINAL.pdf

# 817 family medicine programs
# 11,000 IM categorical positions, 17,000 applicants (4,500 US MD)
# there are some specialities with like 5-10 spots and 5-10 applicants, some
# even less so the lower bound will be low

# positions 

BOUNDS = {
    'n_programs': (10, 800), # add check that programs must be > 5
    'n_positions': (5, 10000), # lower bound will always check for min n_programs
    'n_applicants': (20, 18000),
    'interviews_per_spot': (5, 20)
}

FINAL_CONSTANTS_COLUMNS = [
    'n_programs',
    'n_applicants',
    'spots_per_program',
    'interviews_per_spot',
    'simulations_per_s',
    'max_applications',
    'study_min_signal',
    'study_max_signal',
    'result_file_prefix'
]

def get_generated_datasets() -> pd.DataFrame:
    df = None
    if not os.path.exists(EXPORT_FILE_PATH):
        df = pd.DataFrame(columns=FINAL_CONSTANTS_COLUMNS)
    else:
        df = pd.read_csv(EXPORT_FILE_PATH)
    return df
    

def bounds_check():
    """
    Simple bounds safety check to ensure we don't have odd inputs.
    """
    for key, (low, high) in BOUNDS.items():
        if low >= high:
            raise ValueError(f"Invalid bounds for {key}: {low} >= {high}")
    if BOUNDS['n_programs'][0] < MINIMUM_APPLICATIONS:
        raise ValueError(
            f"n_programs lower bound must be at least {MINIMUM_APPLICATIONS}")

def get_random_bounds() -> dict:
    """Generate a random set of constants within the given bounds."""
    constants = {}
    for key, (low, high) in BOUNDS.items():
        if key == 'n_positions':
            continue  # skip for now, will calculate after
        if isinstance(low, int) and isinstance(high, int):
            constants[key] = np.random.randint(low, high + 1)
        else:
            print(f"Bounds for {key} are not both integers.")
    constants['n_positions'] = np.random.randint(
        min(constants['n_programs'], BOUNDS['n_positions'][0]), 
        BOUNDS['n_positions'][1] + 1
    )
    max_applications = min(MAXIMUM_APPLICATIONS, constants['n_programs'])
    # generate now a random maximum applications
    constants['max_applications'] = np.random.randint(
        MINIMUM_APPLICATIONS, max_applications+1)
    
    return constants

def get_final_dataframe(constants):
    """Convert constants dict to a DataFrame row."""
    row = {}
    row['n_programs'] = constants['n_programs']
    row['n_applicants'] = constants['n_applicants']
    # now calculate some derived values:
    # spots_per_program from n_applicants and n_programs
    # HAVE to bound with 1 because imagine 800 programs, 5 positions, then 0
    row['spots_per_program'] = max(
        1, constants['n_positions'] // constants['n_programs'])
    row['interviews_per_spot'] = constants['interviews_per_spot']
    row['simulations_per_s'] = SIMULATIONS_PER_S
    row['max_applications'] = constants['max_applications']
    row['study_min_signal'] = 3
    row['study_max_signal'] = max( # can be 2 if max applications is 5 (by 3)
        3, min(40, constants['max_applications'] - 3))


    # study min and study max signal is "constant" for the conditions
    # in this file prefix
    # randomness is in n programs, n applicants, interviews per spot, 
    # and maximum applications
    # 4 dimensional heatmap
    row['result_file_prefix'] = (
        f"heatmap_np{row['n_programs']}"
        f"_pos{constants['n_positions']}"
        f"_sppcalc{row['spots_per_program']}"
        f"_na{row['n_applicants']}"
        f"_ips{row['interviews_per_spot']}"
        f"_ma{row['max_applications']}"
    )
    return pd.DataFrame([row])

def get_random_row():
    consts = get_random_bounds()
    df_row = get_final_dataframe(consts)
    return df_row

if __name__ == "__main__":
    bounds_check()
    iteration = 0
    current_datasets = get_generated_datasets()
    initial_starting_size = len(current_datasets)
    print(f"Starting with {len(current_datasets)} existing constant sets.")
    existing_prefixes = set(current_datasets['result_file_prefix'].values)
    
    while iteration < NUMBER_TO_GENERATE:
        row = get_random_row()
        prefix = row['result_file_prefix'].values[0]
        if prefix in existing_prefixes:
            continue  # skip duplicates
        iteration += 1 # only count unique additions
        existing_prefixes.add(prefix)
        current_datasets = pd.concat(
            [current_datasets, row], ignore_index=True)
        if (iteration > 0) and (iteration % 100 == 0):
            print(f"Generated {iteration} unique constant sets.")
    
    print(f"Generated {iteration} new constant sets. Final total: {len(current_datasets)}")
    current_datasets.to_csv(EXPORT_FILE_PATH, index=False)
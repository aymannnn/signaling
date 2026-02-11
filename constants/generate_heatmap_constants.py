import numpy as np
import pandas as pd
from scipy.stats import qmc

EXPORT_FILE_PATH = 'base_case.csv'
NUMBER_TO_GENERATE_BASE_2 = 10 # 2^10 = 1024 samples
SIMULATIONS_PER_S = 20 # always constant
SIGNAL_MAX = 40 # max number of signals

# https: // www.nrmp.org/wp-content/uploads/2025/05/Main_Match_Results_and_Data_20250529_FINAL.pdf

# randomized variables
BOUNDS = {
    'n_programs': (10, 800),
    'n_positions': (10, 12000),
    'n_applicants': (20, 18000),
    'interviews_per_spot': (5, 20),
    'max_applications': (5, 60)
}

FINAL_CONSTANTS_COLUMNS = [
    # randomized
    'n_programs',
    'n_positions',
    'n_applicants',
    'interviews_per_spot',
    'max_applications',
    # calculated/fixed
    'applicants_per_position',
    'minimum_unmatched',
    'spots_per_program',
    'simulations_per_s',
    'study_min_signal',
    'study_max_signal',
    'result_file_prefix'
]

INT_KEYS = {
    "n_programs", 
    "n_positions", 
    "n_applicants", 
    "interviews_per_spot",
    "max_applications"}


def sobol_samples(power_2: int, bounds: dict, seed: int = 0) -> pd.DataFrame:
    """
    Generate n Sobol samples over the given bounds (inclusive for integer dims).

    bounds: dict[name] = (low, high)
    Returns: DataFrame with one column per bound key.
    
    QMC = quasi-monte carlo
    
    Use n = 2^m for low discrepancy points, if don't use 2^m doesn't
    guarentee properties. Quadrature rules.
    
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html#scipy.stats.qmc.Sobol
    
    scaling: 
    https://docs.scipy.org/doc//scipy-1.16.2/reference/generated/scipy.stats.qmc.scale.html
    """
    if power_2 <= 0:
        raise ValueError("power_2 must be > 0")

    keys = list(bounds.keys())
    lows = np.array([bounds[k][0] for k in keys], dtype=float)
    highs = np.array([bounds[k][1] for k in keys], dtype=float)

    # simple min/max check
    if np.any(highs < lows):
        bad = [k for k in keys if bounds[k][1] < bounds[k][0]]
        raise ValueError(f"Invalid bounds (high < low) for: {bad}")

    # Sobol points in [0, 1)
    sampler = qmc.Sobol(d=len(keys), scramble=True, rng=seed)
    u = sampler.random_base2(m = power_2)  # shape (n, d)
    
    # to scale to bounds, see docs above
    # scaling is (b-a)*sample + a
    # Scale to bounds (float for now)
    x = (highs-lows) *u + lows

    # Convert integer dimensions to inclusive integer ranges
    for j, k in enumerate(keys):
        if k in INT_KEYS:
            lo, hi = int(bounds[k][0]), int(bounds[k][1])
            # Inclusive mapping: lo..hi
            x[:, j] = lo + np.floor(u[:, j] * (hi - lo + 1))
            x[:, j] = np.clip(x[:, j], lo, hi)

    df = pd.DataFrame(x, columns=keys)

    # Final dtype cast for int keys
    for k in keys:
        if k in INT_KEYS:
            df[k] = df[k].astype(int)

    return df

def process_row(row: pd.Series):
    """Process a single row to add calculated fields."""
    
    # random variables
    n_programs = row['n_programs']
    n_positions = row['n_positions']
    n_applicants = row['n_applicants']
    interviews_per_spot = row['interviews_per_spot']
    max_applications = row['max_applications']
    
    # quick checks:
    # must have more positions then programs
    # e.g. 10 programs and 5 positions is invalid
    # discard simulation
    if n_programs > n_positions:
        return "invalid"
    
    # cannot apply to more then number of programs
    # max applications has to be bounded by minimum programs
    # e.g. 10 applications, 5 programs is invalid
    if n_programs < max_applications:
        max_applications = n_programs
        row['max_applications'] = row['n_programs']

    # this will never be < 1 because always n_positions >= n_programs above
    
    row['spots_per_program'] = n_positions / n_programs

    # always fixed, NTD
    row['simulations_per_s'] = SIMULATIONS_PER_S
    # always 5+ applications per position so minimum is OK     
    row['study_min_signal'] = 0
    
    row['applicants_per_position'] = n_applicants / n_positions
    row['minimum_unmatched'] = max(0, n_applicants - n_positions)
    
    # for maximum signal: first, choose the minimum of the signal max (40)
    # or the maximum applications -3. For example, if max applications is 10,
    # then maximum signal is 7. If max applications is 45, then you choose 40.
    # if max applications is 5 (lower bound), then max signal is 2, but in
    # the second part of max() we force it to be 4 so you do at least 
    # 3/4 signal in that very small case
    
    row['study_max_signal'] = max(
        4, min(SIGNAL_MAX, row['max_applications'] - 1))
    
    # file prefix is a function of the FIVE randomized variables
    row['result_file_prefix'] = (
        f"heatmap_np{n_programs}"
        f"_npos{n_positions}"
        f"_napp{n_applicants}"
        f"_ips{interviews_per_spot}"
        f"_max_app{max_applications}"
    )
    
    return row

# 2 to the 9 is 512 samples
df = sobol_samples(power_2=NUMBER_TO_GENERATE_BASE_2, bounds=BOUNDS, seed=42)
final_df = pd.DataFrame()
first = True

for index, row in df.iterrows():
    processed = process_row(row)
    if isinstance(processed, str):
        if processed == "invalid":
            continue
        else:
            print(f"Unknown processing result: {processed}")
    else:
        if first:
            first = False
            final_df = pd.DataFrame(processed).T.reset_index(drop=True)
        else:
            final_df = pd.concat(
                [final_df, pd.DataFrame([processed])], ignore_index=True)

print(f"Generated {len(df)} total samples.")
print(f"Finalized {len(final_df)} valid constant sets. A total of {len(df) - len(final_df)} invalid samples were discarded, representing {(len(df) - len(final_df)) / len(df) * 100:.2f}% of samples.")

final_df.to_csv(EXPORT_FILE_PATH, index=False)
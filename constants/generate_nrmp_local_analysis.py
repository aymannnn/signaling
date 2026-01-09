import numpy as np
import pandas as pd

# the goal of this is to just do a local analysis around the NRMP variables
# where we randomize max applications and interviews per spot

EXPORT_FILE_PATH = 'local_nrmp_analysis_constants.csv'
SIMULATIONS_PER_S = 10 
SIGNAL_MAX = 40
LOCAL_SETS_PER_NRMP = 30 # how many sets to attempt to generate

# https: // www.nrmp.org/wp-content/uploads/2025/05/Main_Match_Results_and_Data_20250529_FINAL.pdf
nrmp_data = pd.read_csv('constants_nrmp.csv')

out = []
scenarios = set()

for _, row in nrmp_data.iterrows():
    for i in range(LOCAL_SETS_PER_NRMP):
        n_programs = int(row["n_programs"])
        app_upper = min(int(row["n_programs"]), 60)
        max_apps = int(np.random.randint(5, app_upper + 1))
        d = row.to_dict()
        d["max_applications"] = max_apps
        d["interviews_per_spot"] = int(np.random.randint(5, 21))
        d["simulations_per_s"] = SIMULATIONS_PER_S
        d["study_max_signal"] = min(SIGNAL_MAX, max_apps - 1)
        d["result_file_prefix"] = (
            f"{d['result_file_prefix']}_ma{max_apps}_ips{d['interviews_per_spot']}"
        )
        if d['result_file_prefix'] in scenarios:
            continue
        scenarios.add(d['result_file_prefix'])
        out.append(d)

final_df = pd.DataFrame(out)
final_df.to_csv(EXPORT_FILE_PATH, index=False)
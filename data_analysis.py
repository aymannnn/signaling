import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

GAMMA = True

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

results_directory = 'simulation_results/'
# will read in from constants file used in simulation.py
constants = pd.read_csv('constants_gamma.csv') if GAMMA else pd.read_csv('constants.csv')
file_prefix = constants.loc[constants['Variable'] ==
                            'result_file_prefix', 'Value'].values[0]
print(file_prefix)
if not file_prefix:
    file_prefix = ''
print(file_prefix)
graphs_prefix = results_directory + file_prefix
file_path = results_directory + file_prefix + '_simulation_results.csv'
df = pd.read_csv(file_path)

# signal values (column names except 'Parameter')
signal_values = df.columns.drop('Parameter')
signal_values = sorted(int(i) for i in signal_values)

parameters = ['Unmatched_Applicants', 'Unfilled_Spots', 'Reviews_Per_Program']
parameter_graph_names = [
    'Number of Unmatched Applicants',
    'Number of Unfilled Spots',
    'Average Number of Reviews Per Program',
]

# -------------------------------------------------------------------
# Global plotting style
# -------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

line_color = "black"
ci_color = "grey"

# -------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------


def compute_means_and_ci(param_data, signal_values, alpha=0.95):
    means = []
    ci_lower = []
    ci_upper = []

    for s in signal_values:
        col = str(s)
        values = param_data[col].values

        mean = np.nanmean(values)
        if len(values) > 1:
            sem = stats.sem(values, nan_policy='omit')
            ci = stats.t.interval(alpha, len(values) - 1, loc=mean, scale=sem)
            lower, upper = ci
        else:
            lower = upper = mean

        means.append(mean)
        ci_lower.append(lower)
        ci_upper.append(upper)

    return np.array(means), np.array(ci_lower), np.array(ci_upper)


# -------------------------------------------------------------------
# Joint figure with subplots
# -------------------------------------------------------------------
fig, axes = plt.subplots(
    len(parameters), 1,
    figsize=(6, 2.5 * len(parameters)),
    sharex=True
)

for ax, param, graph_name in zip(axes, parameters, parameter_graph_names):
    param_data = df[df['Parameter'] == param]
    means, ci_lower, ci_upper = compute_means_and_ci(param_data, signal_values)

    ax.plot(signal_values, means, color=line_color, linewidth=2,
            marker='o', label='Mean')
    ax.fill_between(signal_values, ci_lower, ci_upper,
                    color=ci_color, alpha=0.3, label='95% CI')

    ax.set_ylabel(graph_name)
    ax.set_title(graph_name)
    ax.set_xticks(signal_values)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(frameon=False)

axes[-1].set_xlabel('Number of Signals')

fig.tight_layout()
fig.savefig(graphs_prefix + '_joint_results.png', bbox_inches='tight')
plt.close(fig)

# -------------------------------------------------------------------
# Individual figures
# -------------------------------------------------------------------
for param, graph_name in zip(parameters, parameter_graph_names):
    param_data = df[df['Parameter'] == param]
    means, ci_lower, ci_upper = compute_means_and_ci(param_data, signal_values)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(signal_values, means, color=line_color, linewidth=2,
            marker='o', label='Mean')
    ax.fill_between(signal_values, ci_lower, ci_upper,
                    color=ci_color, alpha=0.3, label='95% CI')

    ax.set_xlabel('Number of Signals')
    ax.set_ylabel(graph_name)
    ax.set_title(f'{graph_name} vs Number of Signals')
    ax.set_xticks(signal_values)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(graphs_prefix + f'_{param.lower()}_plot.png',
                bbox_inches='tight')
    plt.close(fig)

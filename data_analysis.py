import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

df = pd.read_csv('simulation_results.csv')

# now create some graphs
indexes = df.columns.drop('Parameter')
indexes = [int(i) for i in indexes]
indexes.sort()

parameters = ['Unmatched_Applicants', 'Unfilled_Spots', 'Reviews_Per_Program']
parameter_graph_names = ['Number of Unmatched Applicants',
                         'Number of Unfilled Spots',
                         'Average Number of Reviews Per Program']

# jointo graph
fig, axes = plt.subplots(len(parameters), 1, figsize=(10, 4*len(parameters)))
for idx, (param, graph_name) in enumerate(zip(parameters, parameter_graph_names)):
    # Filter data for this parameter
    param_data = df[df['Parameter'] == param]

    # Calculate statistics for each signal value
    means = []
    ci_lower = []
    ci_upper = []

    for signal_val in indexes:
        values = param_data[str(signal_val)].values
        mean = np.mean(values)
        sem = stats.sem(values)  # Standard error of mean
        ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=sem)

        means.append(mean)
        ci_lower.append(ci[0])
        ci_upper.append(ci[1])

    # Plot
    ax = axes[idx]
    ax.plot(indexes, means, 'b-', linewidth=2, label='Mean')
    ax.fill_between(indexes, ci_lower, ci_upper, alpha=0.3, label='95% CI')

    ax.set_xlabel('Number of Signals', fontsize=12)
    ax.set_ylabel(graph_name, fontsize=12)
    ax.set_title(f'{graph_name} vs Number of Signals', fontsize=14)
    ax.set_xticks(indexes)  # Set x-axis to show all integer values
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Also create individual plots for more detailed view
for param, graph_name in zip(parameters, parameter_graph_names):
    param_data = df[df['Parameter'] == param]

    means = []
    ci_lower = []
    ci_upper = []

    for signal_val in indexes:
        values = param_data[str(signal_val)].values
        mean = np.mean(values)
        sem = stats.sem(values)
        ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=sem)

        means.append(mean)
        ci_lower.append(ci[0])
        ci_upper.append(ci[1])

    plt.figure(figsize=(10, 6))
    plt.plot(indexes, means, 'b-', linewidth=2, marker='o', label='Mean')
    plt.fill_between(indexes, ci_lower, ci_upper, alpha=0.3, label='95% CI')

    plt.xlabel('Number of Signals', fontsize=12)
    plt.ylabel(graph_name, fontsize=12)
    plt.title(f'{graph_name} vs Number of Signals', fontsize=14)
    plt.xticks(indexes)  # Set x-axis to show all integer values
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(f'{param.lower()}_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

graphs_to_generate = {
    'Distribution of Interviews per Position': (8, 12/8),
    'Distribution of Applications per Applicant (Mean 72)': (40, 72/40),
    'Distribution of Applications per Applicant (Mean 30)': (10, 30/10)
    
}

for graph_name, (SHAPE, SCALE) in graphs_to_generate.items():
    values = np.random.gamma(SHAPE, SCALE, size=100000)
    plt.hist(values, bins=30)
    plt.title(graph_name)
    savename = graph_name.replace(" ", "_").lower()
    plt.savefig(f"{savename}.png")
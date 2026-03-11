import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

graphs_to_generate = {
    'Distribution of Interviews per Position': (8, 12/8),
    'Distribution of Applications per Applicant (Mean 72)': (40, 72/40),
    'Distribution of Applications per Applicant (Mean 30)': (10, 30/10)
}

for graph_name, (SHAPE, SCALE) in graphs_to_generate.items():
    # Define the range for the x-axis (0 to Mean + 4*StdDev approx)
    mean = SHAPE * SCALE
    std = np.sqrt(SHAPE) * SCALE
    x = np.linspace(0, mean + 4*std, 1000)
    
    # Calculate theoretical PDF
    y = gamma.pdf(x, SHAPE, scale=SCALE)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f'Gamma(shape={SHAPE}, scale={SCALE:.2f})', color='royalblue', linewidth=2)
    plt.fill_between(x, y, alpha=0.2, color='royalblue')
    
    plt.title(graph_name)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    savename = graph_name.replace(" ", "_").lower().replace("(", "").replace(")", "")
    plt.savefig(f"../figures/gamma/{savename}.png")
    plt.close()

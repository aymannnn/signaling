# simple method to graph gamma distribution of max applications
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

OUTPUT_PATH = "results/gamma_graph.png"

def plot_gamma_distribution(shape, scale, max_apps, path = None):
    x = np.linspace(0, max_apps, 1000)
    y = gamma.pdf(x, a=shape, scale=scale)
    mean = shape * scale
    # variance = shape * (scale ** 2)
    std = np.sqrt(shape) * scale
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f'Gamma Distribution\nShape={shape:.2f}, Scale={scale:.2f}\nMean={mean:.2f}, Std={std:.2f}')
    plt.title('Gamma Distribution of Max Applications')
    plt.xlabel('Max Applications')
    plt.ylabel('Probability Density')
    plt.xlim(0, max_apps)
    plt.ylim(0, max(y) * 1.1)
    plt.grid()
    plt.legend()
    if path:
        plt.savefig(path)
    plt.show()
    
plot_gamma_distribution(shape=8.0, scale=30/8, max_apps=100, path=OUTPUT_PATH)
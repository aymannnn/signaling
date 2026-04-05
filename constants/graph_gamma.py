import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
import os

# Robust path handling: get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Target directory: signaling/figures/gamma_pdfs
save_dir = os.path.join(script_dir, "..", "figures", "gamma_pdfs")
os.makedirs(save_dir, exist_ok=True)

# Parameters
interviews_params = (8, 12/8)  # Shape, Scale
apps_72_params = (40, 72/40)
apps_30_params = (10, 30/10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left Panel: Distribution of Interviews per Position
shape, scale = interviews_params
mean = shape * scale
std = np.sqrt(shape) * scale
x = np.linspace(0, mean + 4*std, 1000)
y = gamma.pdf(x, shape, scale=scale)

ax1.plot(x, y, color='royalblue', linewidth=2, label=f'Mean={mean:.1f} (k={shape}, θ={scale:.2f})')
ax1.fill_between(x, y, alpha=0.2, color='royalblue')
ax1.set_title('Distribution of Interviews per Position')
ax1.set_xlabel('Interviews')
ax1.set_ylabel('Probability Density')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Right Panel: Distribution of Applications per Applicant
# Plot Mean 72
shape, scale = apps_72_params
mean = shape * scale
std = np.sqrt(shape) * scale
x = np.linspace(0, 150, 1000) # Use a common range for comparison
y = gamma.pdf(x, shape, scale=scale)
ax2.plot(x, y, color='forestgreen', linewidth=2, label=f'Mean 72 (k={shape}, θ={scale:.2f})')
ax2.fill_between(x, y, alpha=0.1, color='forestgreen')

# Plot Mean 30
shape, scale = apps_30_params
mean = shape * scale
std = np.sqrt(shape) * scale
y = gamma.pdf(x, shape, scale=scale)
ax2.plot(x, y, color='crimson', linewidth=2, label=f'Mean 30 (k={shape}, θ={scale:.2f})')
ax2.fill_between(x, y, alpha=0.1, color='crimson')

ax2.set_title('Distribution of Applications per Applicant')
ax2.set_xlabel('Applications')
ax2.set_ylabel('Probability Density')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

fig.suptitle('Gamma Distribution Probability Density Functions', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
save_path = os.path.join(save_dir, "combined_gamma_distributions.png")
plt.savefig(save_path)
# plt.show() # Commented out to avoid blocking in CLI environments


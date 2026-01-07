from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Imports + load
# ==============

INPUT_PATH = "results/base_case.csv"
df = pd.read_csv(INPUT_PATH)
df.head(5)

y = df['best_signal_rpp']


# 4 panel figure of the different variables
fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=False, sharey=False)

x_axis_labels = [
    'Number of Positions',
    'Number of Applicants',
    'Interviews per Spot',
    'Total Applications'
]

axs[0, 0].scatter(df['n_positions'], y, alpha=0.2, s=5)
axs[0, 1].scatter(df['n_applicants'], y, alpha=0.2, s=5)
axs[1, 0].scatter(df['interviews_per_spot'], y, alpha=0.2, s=8)
axs[1, 1].scatter(df['max_applications'], y, alpha=0.2, s=5)

panel_labels = ["A", "B", "C", "D"]

for ax, xlabel, plabel in zip(axs.ravel(), x_axis_labels, panel_labels):
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Optimal Signal")
    ax.text(
        -.12, 1.10, plabel,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12, fontweight="bold"
    )

fig.tight_layout()
plt.savefig('final_figures/global_analysis_4_panel_scatter.png', dpi=300)
plt.show()
plt.close()


# applicants vs. signals

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle

# Consistent full-series definitions (recommended)
x_full = df["applicants_per_position"]
y_full = df["best_signal_rpp"]
mask = x_full < 5

fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)

# Main: zoomed subset
ax.scatter(x_full[mask], y_full[mask],
           alpha=0.5, s=10, edgecolors="none", rasterized=True)
ax.set_xlabel("Applicants per Position")
ax.set_ylabel("Optimal Signal")
ax.set_xlim(0, 5)

# Inset: full data (top-right)
axins = inset_axes(ax, width="40%", height="40%",
                   loc="upper right", borderpad=1.0)
axins.scatter(x_full, y_full,
              alpha=0.4, s=3, edgecolors="none", rasterized=True)
axins.set_title("Complete Data", fontsize=9, pad=2)
axins.tick_params(labelsize=8)

# Set inset limits to full range
axins.set_xlim(x_full.min(), x_full.max())
axins.set_ylim(y_full.min(), y_full.max())

# Clean spines
for a in (ax, axins):
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

plt.savefig("final_figures/global_analysis_applicants_per_position_inset.png",
            dpi=300, bbox_inches="tight")
plt.show()
plt.close()

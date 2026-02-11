from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style to match previous visualizations
sns.set_theme(style="white")

# ----------------------------
# I/O + setup
# ----------------------------
out_dir = Path("figures")
out_dir.mkdir(parents=True, exist_ok=True)

# Load only the local analysis results
df = pd.read_csv("results/nrmp_local_analysis_results.csv")

# ----------------------------
# Data Processing
# ----------------------------

# Convert optimal signal (best_signal_rpp) to numeric
df["Optimal Signal"] = pd.to_numeric(df["best_signal_rpp"], errors="coerce")

# Extract the base program identifier from result_file_prefix
# The generator script appends "_ma{max_apps}_ips{ips}", so we split at "_ma"
# to group by the constant program/applicant/position set.
df["Program Group"] = df["result_file_prefix"].str.rsplit("_ma", n=1).str[0]

# Clean up any potential NaNs in the plotting columns
df = df.dropna(subset=["Optimal Signal", "Program Group"])

# ----------------------------
# Visualization
# ----------------------------

fig, ax = plt.subplots(figsize=(12, 7))

# Create violin plot
# palette="cool" matches the color scheme used in the UMAP/Pairplots
sns.violinplot(
    data=df.sort_values("Program Group"),
    x="Program Group",
    y="Optimal Signal",
    palette="cool",
    hue = 'Program Group',
    legend = False,
    ax=ax,
    inner="quartile",
    linewidth=1.2
)

# Aesthetics
ax.set_title(
    "Distribution of Optimal Signal per Program (Local Analysis)", fontsize=14, pad=15)
ax.set_xlabel("Program", fontsize=11)
ax.set_ylabel("Optimal Signal", fontsize=11)

# Rotate x-axis labels if there are many programs to prevent overlap
plt.xticks(rotation=45, ha="right")

# Save output
fig.tight_layout()
fig.savefig(out_dir / "violin_optimal_signal_local.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)

'''
The violin plots illustrate the distribution and density of the 'optimal signal' during local sensitivity analysis. The morphology of each violin represents the frequency of these optimal values, highlighting the robustness of signaling benefits even as institutional variables like application caps and interview availability fluctuate.
'''


## VISUALIZE MA / IPS

# ----------------------------
# Visualization: Parameter Impact Map
# ----------------------------

# We use a FacetGrid to show each specialty, visualizing how Optimal Signal
# responds to Max Applications, colored by Interview Capacity (IPS).
g = sns.relplot(
    data=df.sort_values("Program Group"),
    x="max_applications",
    y="Optimal Signal",
    hue="interviews_per_spot",
    col="Program Group",
    col_wrap=3,
    palette="viridis",  # Distinct from the "cool" palette to highlight the IPS variable
    alpha=0.8,
    s=60,
    kind="scatter",
    facet_kws={'sharey': False, 'sharex': False}
)

# Aesthetics
g.set_axis_labels("Max Applications", "Optimal Signal")
g.set_titles("{col_name}")
g.fig.suptitle(
    "Impact of Application Volume and Interview Capacity on Optimal Signaling", y=1.02, fontsize=16)

# Add a cleaner legend title
g._legend.set_title("Interviews\nper Spot")

# Save output
g.savefig(out_dir / "impact_analysis_parameter_map.png",
          dpi=300, bbox_inches="tight")
plt.close(g.fig)

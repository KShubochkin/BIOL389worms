import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
data = pd.read_csv("C:/Users/corna/Downloads/389 worms - data (4).csv")

# Separate groups
control = data[data['worm'].str.contains('control')].copy()
nicotine = data[data['worm'].str.contains('nicotine')].copy()

# Add 'group' column for easy identification
control['group'] = 'control'
nicotine['group'] = 'nicotine'

# Combine both groups
combined = pd.concat([control, nicotine])

# Variables to visualize
variables = ['Average pump rate during active periods', 'Percent Time Active', 'E-E IPI Mean', 'E-E IPI Median', 'E-E IPI STD',
             'R-E IPI Mean', 'R-E IPI Median', 'R-E IPI STD', 'Total Average Pump Rate (Hz)', 'Total Pumps', 'Total Time (s)',
             'Burst Frequency','Number of Inactive Periods','Mean E Spike Amplitude','Median E Spike Amplitude','Mean R Spike Amplitude','Median R Spike Amplitude','R/E Spike Mean Ratio','R/E Spike Median Ratio']

# Set up multiple subplots
num_vars = len(variables)
fig, axes = plt.subplots(nrows=(num_vars // 3) + (num_vars % 3 > 0), ncols=3, figsize=(15, num_vars * 1.5))
axes = axes.flatten()

# Loop through each variable and create an individual boxplot
for i, var in enumerate(variables):
    sns.boxplot(x='group', y=var, data=combined, hue="group", ax=axes[i])
    axes[i].set_title(var, fontsize=10)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Value", fontsize=9)

# Remove empty subplots if num_vars is not a multiple of 3
for i in range(len(axes)):
    if i >= num_vars:
        fig.delaxes(axes[i])

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

# Run Mann-Whitney U test
mw_results = {}
for var in variables:
    stat, p = stats.mannwhitneyu(control[var], nicotine[var],alternative='greater')
    mw_results[var] = {'statistic': stat, 'p-value': p}
    print(var,mw_results[var])

# Convert results to DataFrame
mw_df = pd.DataFrame(mw_results).T
mw_df.reset_index(inplace=True)
mw_df.columns = ['Variable', 'U Statistic', 'P Value']

# --- Boxplots for each variable ---
num_vars = len(variables)
fig, axes = plt.subplots(nrows=(num_vars // 3) + (num_vars % 3 > 0), ncols=3, figsize=(15, num_vars * 1.5))
axes = axes.flatten()

units = ("Hz", "Percent %", "Time (s)", "Time (s)", "Time (s)", "Time (s)", "Time (s)", "Time (s)", "Hz", "Count", "Time (s)", "Value", "Count", "log(Probability)", "log(Probability)", "log(Probability)", "log(Probability)", "Value", "Value")
for i, var in enumerate(variables):
    sns.boxplot(x='group', y=var, data=combined, hue="group", ax=axes[i])
    axes[i].set_title(var, fontsize=10)
    axes[i].set_xlabel("")
    axes[i].set_ylabel(units[i], fontsize=9)

for i in range(len(axes)):
    if i >= num_vars:
        fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

# --- Visualizing Mann-Whitney U Test Results ---
plt.figure(figsize=(12, 6))
sns.barplot(x='Variable', y='P Value', data=mw_df, hue="Variable",n_boot=1000, edgecolor='black')

# Significance threshold line (p = 0.05)
plt.axhline(y=0.05, color='red', linestyle='--', label='Significance Threshold (p = 0.05)')

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.ylabel("P-Value", fontsize=12)
plt.xlabel("Variable", fontsize=12)
plt.title("Mann-Whitney U Test Results (P-Values)", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()

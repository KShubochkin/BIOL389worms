import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv("C:/Users/corna/Downloads/389 worms - data (1).csv")

# Separate groups
control = data[data['worm'].str.contains('control')]
nicotine = data[data['worm'].str.contains('nicotine')]

# Descriptive statistics
control_desc = control.describe()
nicotine_desc = nicotine.describe()

# Mann-Whitney U Test for each variable
variables = ['Average pump rate during active periods', 'active time%', 'IPI mean', 'IPI median', 'IPI std',
             'ipi re mean', 'ipi re med', 'ipi re std', 'total avg pump rate', 'total pumps', 'total time', 'burst freq']

results = {}
for var in variables:
    stat, p = stats.mannwhitneyu(control[var], nicotine[var])
    results[var] = {'statistic': stat, 'p-value': p}

# Print results
for var, res in results.items():
    print(f"{var}: U = {res['statistic']}, p = {res['p-value']}")

# Visualization
plt.figure(figsize=(12, 6))
sns.boxplot(x='variable', hue='group',
            data=pd.melt(pd.concat([control.assign(group='control'), nicotine.assign(group='nicotine')]),
            id_vars=['group']))
plt.xticks(rotation=45)
plt.show()
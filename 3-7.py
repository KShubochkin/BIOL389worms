import numpy as np
import pandas as pd
import statistics as stats
import scipy.stats as st
from matplotlib import pyplot as plt
from pandas import DataFrame
import seaborn as sns

files = [
    "C:/Users/corna/Downloads/389 worms - control1rawxlsx.csv",
    "C:/Users/corna/Downloads/389 worms - control2rawxlsx.csv",
    "C:/Users/corna/Downloads/389 worms - control3rawxlsx.csv",
    "C:/Users/corna/Downloads/389 worms - control4rawxlsx.csv",
    "C:/Users/corna/Downloads/389 worms - control5rawxlsx.csv",
    "C:/Users/corna/Downloads/389 worms - nicotine1rawxlsx.csv",
    "C:/Users/corna/Downloads/389 worms - nicotine2rawxlsx.csv",
    "C:/Users/corna/Downloads/389 worms - nicotine3rawxlsx.csv",
    "C:/Users/corna/Downloads/389 worms - nicotine4rawxlsx.csv"
]
def read_data(file_path,index):
    df = pd.read_csv(file_path)
    sample_rate = 2000
    total_pumps = df["Total pumps"].iloc[0]

    e_locations = df["E_location"].dropna().tolist()
    e_times = [e / sample_rate for e in e_locations]

    r_locations = df["R_location"].dropna().tolist()
    r_times = [r / sample_rate for r in r_locations]

    R_amplitude = df["R_amplitude"].dropna().tolist()
    E_amplitude = df["E_amplitude"].dropna().tolist()

    RtoE = df["EtoR"].dropna().tolist()

    ipi_data = np.diff(e_locations).reshape(-1, 1)  # E-E intervals
    ipi_data_re = df["RtoE"].dropna().tolist()

    return df, e_locations, e_times, r_locations, total_pumps, sample_rate, r_times, ipi_data_re, R_amplitude, E_amplitude, RtoE

index=0
worm_name=["control1", "control2", "control3", "control4", "control5", "nicotine1", "nicotine2", "nicotine3", "nicotine4", "nicotine5"]
worm_type=['c','c','c','c','c','n','n','n','n']
valid_interval=[(0.0,247.06),(0.0,299.0),(0,120.84),(0,176.00),(0,180),(0,180),(0,180),(0,180),(0,180)]
control=[]
nicotine=[]
ipi_cspread_df = DataFrame()
ipi_nspread_df = DataFrame()

for file in files:
    df, e_locations, e_times, r_locations, total_pumps, sample_rate, r_times, ipi_data_re, R_amplitude, E_amplitude, RtoE = read_data(file,index)
    name = worm_name[index] #RETURN
    ipis = np.array(RtoE)
    start = valid_interval[index][0]
    end = valid_interval[index][1]
    time = end-start
    group=""
    pump_count=0 #RETURN
    for pump in e_times:
        if pump >start and pump<end:
            pump_count=pump_count+1
    std = stats.stdev(e_times) #RETURN
    inter_mean = stats.mean(RtoE)/sample_rate
    inter_median = stats.median(RtoE)/sample_rate
    inter_std = stats.stdev(RtoE)/sample_rate
    df = pd.DataFrame(RtoE)
    inter_skew = df.skew(skipna = True)
    #inter_skew = pd.
    print('interskew: ',inter_skew[0],'len, ',len(inter_skew))
    variance = stats.variance(e_times) #RETURN
    pump_rate = pump_count/time #RETURN
    eamp=stats.mean(E_amplitude)
    ramp=stats.mean(R_amplitude)
    if worm_type[index] == 'c':
        group = "control"
        control.append((name,pump_rate,pump_count,std,variance,eamp,ramp,inter_mean,inter_median,inter_std,inter_skew[0],ipis))
    elif worm_type[index] == 'n':
        group = "nicotine"
        nicotine.append((name, pump_rate, pump_count, std, variance,eamp,ramp,inter_mean,inter_median, inter_std, inter_skew[0],ipis))
    print (f"{name},pump rate {pump_rate}, pump count {pump_count},std {std}, variance {variance}, imean {inter_mean}, imed {inter_median}, istd {inter_std}\n")

    index=index+1

control_pump_rate = (stats.mean((control[0][1],control[1][1],control[2][1],control[3][1],control[4][1])),stats.median(((control[0][1],control[1][1],control[2][1],control[3][1],control[4][1]))))
control_std = stats.mean((control[0][3],control[1][3],control[2][3],control[3][3],control[4][3]))
control_var = stats.mean((control[0][4],control[1][4],control[2][4],control[3][4],control[4][4]))
control_Eamp = stats.mean((control[0][5], control[1][5], control[2][5], control[3][5], control[4][5]))
control_im = stats.mean((control[0][7],control[1][7],control[2][7],control[3][7],control[4][7]))
control_imd = stats.mean((control[0][8],control[1][8],control[2][8],control[3][8],control[4][8]))
control_istd = stats.mean((control[0][9],control[1][9],control[2][9],control[3][9],control[4][9]))
control_iskew = stats.mean((control[0][10],control[1][10],control[2][10],control[3][10],control[4][10]))
nicotine_pump_rate = stats.mean((nicotine[0][1],nicotine[1][1],nicotine[2][1],nicotine[3][1])),stats.median(((nicotine[0][1],nicotine[1][1],nicotine[2][1],nicotine[3][1])))
nicotine_std = stats.mean((nicotine[0][3],nicotine[1][3],nicotine[2][3],nicotine[3][3]))
nicotine_var = stats.mean((nicotine[0][4],nicotine[1][4],nicotine[2][4],nicotine[3][4]))
nicotine_im = stats.mean((nicotine[0][7],nicotine[1][7],nicotine[2][7],nicotine[3][7]))
nicotine_imd = stats.mean((nicotine[0][8],nicotine[1][8],nicotine[2][8],nicotine[3][8]))
nicotine_istd = stats.mean((nicotine[0][9],nicotine[1][9],nicotine[2][9],nicotine[3][9]))
nicotine_iskew = stats.mean((nicotine[0][10],nicotine[1][10],nicotine[2][10],nicotine[3][10]))
print (f"control:  pump rate {control_pump_rate}, im {control_im}, imd {control_imd}, istd {control_istd}, iskew {control_iskew}\n"
       f"nicotine: pump rate {nicotine_pump_rate}, im {nicotine_im}, imd {nicotine_imd}, istd {nicotine_istd}, iskew {nicotine_iskew}\n")
#I want to compare the difference in how much interpump time VARIES in control vs nicotine worms. In one worm,
# there are many interpump intervals. Each worm has a std of interpump intervals. For control and nicotine,
# average these stds. Then compare them with MWU.

pump_rate_stat = st.mannwhitneyu([x[1] for x in control],[x[1] for x in nicotine])
std_stat = st.mannwhitneyu([x[3] for x in control],[x[3] for x in nicotine])
var_stat = st.mannwhitneyu([x[4] for x in control],[x[4] for x in nicotine])
E_amp_stat = st.mannwhitneyu([x[5] for x in control],[x[5] for x in nicotine])
R_amp_stat = st.mannwhitneyu([x[6] for x in control],[x[6] for x in nicotine])
im_stat = st.mannwhitneyu([x[7] for x in control],[x[7] for x in nicotine])
imd_stat = st.mannwhitneyu([x[8] for x in control],[x[8] for x in nicotine])
print([x[9] for x in control],[x[9] for x in nicotine])
istd_stat = st.mannwhitneyu([x[9] for x in control],[x[9] for x in nicotine], alternative='greater')
iskew_stat = st.mannwhitneyu([x[10] for x in control],[x[10] for x in nicotine], alternative='greater')


# Extract inter_std values for control and nicotine groups
control_istds = [x[9] for x in control]
nicotine_istds = [x[9] for x in nicotine]

# Create a DataFrame for visualization
data = pd.DataFrame({
    'Group': ['Control'] * len(control_istds) + ['Nicotine'] * len(nicotine_istds),
    'Inter-pump Interval STD': control_istds + nicotine_istds
})

# Create the boxplot
plt.figure(figsize=(7, 10))
sns.boxplot(x='Group', y='Inter-pump Interval STD', data=data, palette=["blue", "orange"])
sns.stripplot(x='Group', y='Inter-pump Interval STD', data=data,size=4)
plt.title('Comparison of Inter-pump Interval Standard Deviation')
plt.ylabel('Standard Deviation of Inter-pump Intervals')
plt.xlabel('Experimental Group')
plt.show()

print(f"PR: {pump_rate_stat},\nstd: {std_stat}, \nvariance: {var_stat}, \nEamp: {E_amp_stat}, \nRamp: {R_amp_stat},\n IM: {im_stat},\nIMD: {imd_stat}, \nISTD: {istd_stat}, \nISkew: {iskew_stat}")

# Example data: Replace these with your actual distributions
control_data = [
    control[0][11],control[1][11],control[2][11],control[3][11],control[4][11]
]

nicotine_data = [
    nicotine[0][11],nicotine[1][11],nicotine[2][11],nicotine[3][11]
]

### Approach 1: Pooled Comparison ###
pooled_control = np.concatenate(control_data)
pooled_nicotine = np.concatenate(nicotine_data)

ks_stat_pooled, p_value_pooled = st.ks_2samp(pooled_control, pooled_nicotine)
print("Pooled Comparison:")
print(f"KS Statistic: {ks_stat_pooled}")
print(f"P-value: {p_value_pooled}")
if p_value_pooled < 0.05:
    print("The pooled distributions are significantly different (p < 0.05).\n")
else:
    print("The pooled distributions are not significantly different (p >= 0.05).\n")

### Approach 2: Trial-by-Trial Comparison ###
p_values = []
ks_stats = []

for control_trial in control_data:
    for nicotine_trial in nicotine_data:
        ks_stat, p_value = st.ks_2samp(control_trial, nicotine_trial)
        ks_stats.append(ks_stat)
        p_values.append(p_value)

# Summarize the results
avg_ks_stat = np.mean(ks_stats)
avg_p_value = np.mean(p_values)
sig_tests = sum(1 for p in p_values if p < 0.05)

print("Trial-by-Trial Comparison:")
print(f"Average KS Statistic: {avg_ks_stat}")
print(f"Average P-value: {avg_p_value}")
print(f"Number of significant tests (p < 0.05): {sig_tests} out of {len(p_values)}")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_figures(control, nicotine):
    # Convert data to DataFrame for easy plotting
    df_control = pd.DataFrame(control, columns=["Name", "Pump Rate", "Pump Count", "Std", "Variance", "E_amp", "R_amp",
                                                "Inter_Mean", "Inter_Median", "Inter_Std", "Inter_Skew", "IPIs"])
    df_nicotine = pd.DataFrame(nicotine,
                               columns=["Name", "Pump Rate", "Pump Count", "Std", "Variance", "E_amp", "R_amp",
                                        "Inter_Mean", "Inter_Median", "Inter_Std", "Inter_Skew", "IPIs"])
    df_control["Group"] = "Control"
    df_nicotine["Group"] = "Nicotine"
    df = pd.concat([df_control, df_nicotine])

    # Pump Rate Boxplot
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Group", y="Pump Rate", data=df)
    sns.stripplot(x="Group", y="Pump Rate", data=df, color="orange")
    plt.title("Pump Rate Comparison")
    plt.ylabel("Pump Rate (Hz)")
    plt.savefig("pump_rate_comparison.png")
    plt.show()

    # Interpump Interval Variability (Std) Boxplot
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Group", y="Inter_Std", data=df)
    plt.title("Interpump Interval Variability")
    plt.ylabel("Standard Deviation of Interpump Intervals")
    plt.savefig("interpump_variability.png")
    plt.show()

    # Violin Plot of Interpump Intervals
    ipi_control = np.concatenate(df_control["IPIs"].values)
    ipi_nicotine = np.concatenate(df_nicotine["IPIs"].values)

    plt.figure(figsize=(6, 4))
    sns.violinplot(data=[ipi_control, ipi_nicotine], inner="quartile")
    plt.xticks([0, 1], ["Control", "Nicotine"])
    plt.title("Distribution of Interpump Intervals")
    plt.ylabel("Interpump Interval (s)")
    plt.savefig("interpump_distribution.png")
    plt.show()

    # Boxplot for E and R Amplitudes
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Group", y="E_amp", data=df)
    plt.title("E-Amplitude Comparison")
    plt.ylabel("E-Amplitude")
    plt.savefig("E_amplitude_comparison.png")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Group", y="R_amp", data=df)
    plt.title("R-Amplitude Comparison")
    plt.ylabel("R-Amplitude")
    plt.savefig("R_amplitude_comparison.png")
    plt.show()

    # Histogram of Interpump Intervals
    plt.figure(figsize=(6, 4))
    sns.histplot(ipi_control, bins=30, kde=True, color='blue', label="Control", stat="density", alpha=0.6)
    sns.histplot(ipi_nicotine, bins=30, kde=True, color='red', label="Nicotine", stat="density", alpha=0.6)
    plt.title("Histogram of Interpump Intervals")
    plt.xlabel("Interpump Interval (s)")
    plt.legend()
    plt.savefig("interpump_histogram.png")
    plt.show()


# Call function after running your data analysis
plot_figures(control, nicotine)
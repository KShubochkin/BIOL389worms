from matplotlib import pyplot as plt
from numpy import array, mean, median
from scipy.stats import iqr, mode, skew, kurtosis, entropy
from sklearn.neighbors import KernelDensity
from statsmodels.tsa.stattools import acf
import code as a
import pandas as pd
import scipy.stats as stats
from scipy.signal import welch
import numpy as np

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
control_files = []
nicotine_files = []



results = []

for file in files:
    df, e_locations, e_times, r_locations, total_pumps, sample_rate, ipi_threshold, r_times, ipi_data_re, R_amplitude, E_amplitude = a.read_data(file)

    inactive_periods = a.detect_nonactive_segments(e_locations, r_locations, ipi_threshold, sample_rate)
    active_periods = a.list_active_segments(e_locations, r_locations, sample_rate, inactive_periods)
    total_active_time = 0
    for i in range(len(active_periods)):
        total_active_time += active_periods[i][1] - active_periods[i][0]
    active_pump_rate = a.find_active_pump_rate(active_periods, e_locations, sample_rate,total_active_time)
    avg_active_pump_rate = a.find_active_pump_rate(active_periods, e_locations, sample_rate, total_active_time)

    #print(f"File: {file}")
    #print(f"Active Pump Rate: {active_pump_rate:.2f} pumps/s")

    if 'control' in file:
        control_files.append(file)
    elif 'nicotine' in file:
        nicotine_files.append(file)


    a.plot_active_inactive_timeline(active_periods, inactive_periods)
    #a.plot_raster(e_times)

    total_time = e_times[-1] - e_times[0]
    inactive_time = total_time - total_active_time
    active_ratio = total_active_time / total_time
    inactive_ratio = inactive_time / total_time

    # New Data
    num_active_periods = len(active_periods)
    num_rest_periods = num_active_periods - 1  # Rest periods occur between active segments

    active_durations = [end - start for start, end in active_periods]
    rest_durations = [active_periods[i][0] - active_periods[i - 1][1] for i in
                      range(1, num_active_periods)] if num_active_periods > 1 else []

    avg_active_duration = sum(active_durations) / len(active_durations) if active_durations else 0
    avg_rest_duration = sum(rest_durations) / len(rest_durations) if rest_durations else 0
    max_active_duration = max(active_durations) if active_durations else 0
    max_rest_duration = max(rest_durations) if rest_durations else 0

    inter_pump_intervals = [e_times[i] - e_times[i - 1] for i in range(1, len(e_times))]
    mean_ipi = sum(inter_pump_intervals) / len(inter_pump_intervals) if inter_pump_intervals else 0
    median_ipi = sorted(inter_pump_intervals)[len(inter_pump_intervals) // 2] if inter_pump_intervals else 0
    std_dev_ipi = (sum((ipi - mean_ipi) ** 2 for ipi in inter_pump_intervals) / len(
        inter_pump_intervals)) ** 0.5 if inter_pump_intervals else 0

    inter_pump_intervals_re = [e_times[i] - r_times[i-1] for i in range(1, len(e_times))]
    mean_ipi_re = sum(inter_pump_intervals_re) / len(inter_pump_intervals_re) if inter_pump_intervals_re else 0
    median_ipi_re = sorted(inter_pump_intervals_re)[len(inter_pump_intervals_re) // 2] if inter_pump_intervals_re else 0
    std_dev_ipi_re = (sum((ipi - mean_ipi) ** 2 for ipi in inter_pump_intervals_re) / len(
        inter_pump_intervals_re)) ** 0.5 if inter_pump_intervals_re else 0

    burst_frequency = num_active_periods / total_time if total_time > 0 else 0  # How often bursts occur per second

    results.append({
        #"File": file.split("/")[-1],  # Get filename only
        #"Total Pumps": total_pumps,
        #"Total Time (s)": total_time,
        #"Active Time (s)": total_active_time,
        #"Inactive Time (s)": inactive_time,
        #"Active Ratio (%)": active_ratio * 100,
        #"Inactive Ratio (%)": inactive_ratio * 100,
        #"Avg Active Pump Rate": avg_active_pump_rate,
        #"Total Active Periods": num_active_periods,
        #"Total Rest Periods": num_rest_periods,
        #"Avg Active Period Duration (s)": avg_active_duration,
        #"Avg Rest Period Duration (s)": avg_rest_duration,
        #"Max Active Period Duration (s)": max_active_duration,
        #"Max Rest Period Duration (s)": max_rest_duration,
        #"Mean IPI EE (s)": mean_ipi,
        #"Median IPI EE (s)": median_ipi,
        #"Std Dev of IPI EE(s)": std_dev_ipi,
        #"Mean IPI RE (s)": mean_ipi_re,
        #"Median IPI RE (s)": median_ipi_re,
        #"Std Dev of IPI RE(s)": std_dev_ipi_re,
        #"Burst Frequency (Hz)": burst_frequency,
        #"Total Average Pump Rate": total_pumps/total_time
        #"Mean E Spike Amplitude": mean(E_amplitude),
        #"Median E Spike Amplitude": median(E_amplitude),
        #"Mean R Spike Amplitude": mean(R_amplitude),
        #"Median R Spike Amplitude": median(R_amplitude),
        #"E/R Spike Mean Ratio": mean(E_amplitude)/mean(R_amplitude),
        #"E/R Spike Median Ratio": median(E_amplitude) / median(R_amplitude)
    })

print(results)

all_files = control_files + nicotine_files

event_traces = []  # Store event times for raster plot
file_labels = []

for idx, file in enumerate(all_files):
    df, e_locations, e_times, r_locations, total_pumps, sample_rate, ipi_threshold, r_times, ipi_data_re, R_amplitude, E_amplitude = a.read_data(
        file)

    inactive_periods = a.detect_nonactive_segments(e_locations, r_locations, ipi_threshold, sample_rate)
    active_periods = a.list_active_segments(e_locations, r_locations, sample_rate, inactive_periods)
    total_active_time = sum(period[1] - period[0] for period in active_periods)
    active_pump_rate = a.find_active_pump_rate(active_periods, e_locations, sample_rate, total_active_time)

    #a.plot_active_inactive_timeline(active_periods, inactive_periods)
    #a.plot_raster(e_times)

    event_traces.append(e_times)
    file_labels.append(file)

# Create a combined raster plot
fig, ax = plt.subplots(figsize=(15, len(all_files) * 0.5),dpi=300)

for idx, (e_times, file) in enumerate(zip(event_traces, file_labels)):
    ax.vlines(e_times, idx - 0.2, idx + 0.4, color='black',linewidth=0.5)

    # Draw border between control and nicotine
    if file in nicotine_files and idx > 0 and file_labels[idx - 1] in control_files:
        ax.axhline(idx - 0.5, color='red', linestyle='--', linewidth=2)

ax.set_yticks(range(len(all_files)))
ax.set_yticklabels(file_labels)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Files")
ax.set_title("Raster Plot of All Traces")
#plt.show()

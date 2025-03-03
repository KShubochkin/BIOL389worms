import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.patches as mpatches
from sklearn.neighbors import KernelDensity

#e_e_ipi_threshold = df["3x E-E IPI"].dropna().iloc[0]
#r_e_ipi_threshold = df["3x R-E IPI"].dropna().iloc[0]
def read_data(file_path):
    df = pd.read_csv(file_path)
    sample_rate = 2000
    total_pumps = df["Total pumps"].iloc[0]
    e_locations = df["E_location"].dropna().tolist()
    e_times = [e / sample_rate for e in e_locations]
    r_locations = df["R_location"].dropna().tolist()
    ipi_data = np.diff(e_locations).reshape(-1, 1)  # E-E intervals
    gmm = GaussianMixture(n_components=2, random_state=42)  # Two states: active vs. inactive
    gmm.fit(ipi_data)

    e_e_ipi_threshold = np.mean(gmm.means_)  # Midpoint of the two clusters
    r_e_ipi_threshold = np.mean(gmm.means_)  # Midpoint of the two clusters

    # You can move the logic for setting thresholds here as well
    return df, e_locations, e_times, r_locations, total_pumps, sample_rate, e_e_ipi_threshold, r_e_ipi_threshold

#percentile model
#e_e_ipi_threshold = np.percentile(np.diff(e_locations), 90) * 1  # Adjust factor as needed
#r_e_ipi_threshold = np.percentile([e - r for e, r in zip(e_locations[1:], r_locations[:-1])], 90) * 1

from sklearn.mixture import GaussianMixture


def detect_nonactive_segments(e_locations,r_locations,r_e_ipi_threshold,sample_rate,e_e_ipi_threshold):
    nonactive_segments_RE = []

    for i in range(len(e_locations) - 1):
        #print("checking e from next ", e_locations[i+1]/sample_rate, "to R", r_locations[i]/sample_rate," and e ",e_locations[i]/sample_rate)
        gap1 = e_locations[i+1] - e_locations[i]
        gap2 = e_locations[i + 1] - r_locations[i]
        if gap2 > r_e_ipi_threshold:
            #print(gap1,">",e_e_ipi_threshold,gap2,">",r_e_ipi_threshold,"adding ",r_locations[i]/sample_rate)
            nonactive_segments_RE.append((r_locations[i]/sample_rate,e_locations[i+1]/sample_rate))
        elif gap1 > e_e_ipi_threshold:
            nonactive_segments_RE.append((e_locations[i] / sample_rate, e_locations[i + 1] / sample_rate))

    return nonactive_segments_RE

inactive_periods_RE = detect_nonactive_segments()
#print("inactive periods: ", inactive_periods_RE)

def list_active_segments(e_locations,sample_rate,r_locations):
    nonactive = detect_nonactive_segments()
    active = []
    if not nonactive:  # No nonactive segments, entire range is active
        return [(e_locations[0] / sample_rate, r_locations[-1] / sample_rate)]

        # If there's nonactive data, find active segments
    if nonactive[0][0] > e_locations[0] / sample_rate:
        active.append((e_locations[0] / sample_rate, nonactive[0][0]))

    for i in range(len(nonactive) - 1):
        active.append((nonactive[i][1], nonactive[i + 1][0]))

    if nonactive[-1][1] < r_locations[-1] / sample_rate:
        active.append((nonactive[-1][1], r_locations[-1] / sample_rate))

    return active


active_periods = list_active_segments()
#print("active periods: ", active_periods)
total_active_time = 0
for i in range(len(active_periods)):
    total_active_time += active_periods[i][1] - active_periods[i][0]
#print("total active time: ", total_active_time)

def find_active_pump_rate():
    # Compute weighted average pump rate within active periods
    avg_active_pump_rate = 0
    total_weight = 0  # Track total time weight sum

    for period in active_periods:
        #print("period:", period)
        rate = calculate_pump_rate(period)[0]
        time_weight = (period[1] - period[0]) / total_active_time
        #print("time weight is ",time_weight)
        avg_active_pump_rate += rate * time_weight
        total_weight += time_weight  # Track total weight for proper averaging
        #print("added ", rate * time_weight, " for ", rate, time_weight, "(pump rate of period)")

    # Normalize by total weight instead of len(active_periods)
    #print("total weight ",total_weight)
    return avg_active_pump_rate


def calculate_pump_rate(period,e_locations,sample_rate):
    start, end = period
    #print(start, end)
    length = end - start

    pumps = [e for e in e_locations if start <= e / sample_rate < end]

    if len(pumps) > 0:
        #print("there are ", len(pumps), "pumps in this period.")
        pump_rate = len(pumps) / length  # Proper rate calculation
        #print("average pump rate:", pump_rate)
    else:
        pump_rate = 0

    return pump_rate, end - start

def interpump_intervals(e_times):
    intervals = []
    for j in range(len(e_times) - 1):
        if e_times[j + 1] - e_times[j]>0 :
            intervals.extend([e_times[j + 1] - e_times[j]])
    if intervals:
        mean_ipi = np.mean(intervals)
        median_ipi = np.median(intervals)
        std_ipi = np.std(intervals)
        print(f"IPI - Mean: {mean_ipi}, Median: {median_ipi}, Std Dev: {std_ipi}")

        # Plot IPI histogram
        plt.hist(intervals, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel("Inter-Pump Interval (s)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Inter-Pump Intervals")
        plt.show()

        plt.boxplot(intervals)
        plt.xlabel("Inter-Pump Interval (s)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Inter-Pump Intervals")
        plt.show()
    else:
        print("No inter-pump intervals available.")

def active_vs_inactive_ratio(e_times):
    total_time = e_times[-1] - e_times[0]  # Total experiment duration
    inactive_time = total_time - total_active_time
    active_percentage = (total_active_time / total_time) * 100
    inactive_percentage = (inactive_time / total_time) * 100
    print(f"Active Time: {active_percentage:.2f}%, {total_active_time}s | Inactive Time: {inactive_percentage:.2f}%, {inactive_time}s | Total Time: {total_time:.2f}s")


def frequency_analysis(e_times):
    if len(e_times) < 2:
        print("Not enough data for frequency analysis.")
        return

    time_series = np.array(e_times)
    ipi_series = np.diff(time_series)  # Inter-pump intervals

    # Compute FFT
    y = fft(ipi_series)
    freq = fftfreq(len(ipi_series), d=np.mean(ipi_series))  # Frequency domain

    plt.plot(freq[:len(freq)//2], np.abs(y[:len(freq)//2]))  # Plot only positive frequencies
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Frequency Spectrum of Pumping Intervals")
    plt.show()

def plot_pump_count_sliding_window(e_times,window_size=5, step_size=1):
    """
    Plots the number of pumps over time using a sliding window approach.

    Args:
    - window_size: Size of the sliding window in seconds.
    - step_size: Step size in seconds.
    """
    max_time = e_times[-1]  # Last timestamp in dataset
    window_starts = np.arange(0, max_time, step_size)
    pump_counts = []

    for start in window_starts:
        end = start + window_size
        count = sum(start <= e <= end for e in e_times)
        pump_counts.append(count)

    plt.figure(figsize=(10, 5))
    plt.plot(window_starts, pump_counts, marker='o', linestyle='-', color='b')
    plt.xlabel("Time (s)")
    plt.ylabel("Pump Count")
    plt.title(f"Pump Count Over Time (Window Size = {window_size}s, Step = {step_size}s)")
    plt.grid()
    plt.show()


#plot_pump_count_sliding_window(window_size=50, step_size=1)

def plot_smoothed_pump_rate(e_times,window_size=10):
    """
    Plots a smoothed pump rate over time using a rolling window approach.
    """
    pump_rates = []
    time_centers = []

    # Sliding window over active periods
    for start, end in active_periods:
        times = [t for t in e_times if start <= t <= end]
        for i in range(len(times) - window_size):
            time_centers.append(times[i + window_size // 2])  # Middle of window
            rate = window_size / (times[i + window_size] - times[i])  # Rate calculation
            pump_rates.append(rate)

    plt.figure(figsize=(10, 5))
    plt.plot(time_centers, pump_rates, marker='o', linestyle='-', color='blue', alpha=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Pump Rate (Hz)")
    plt.title("Smoothed Pump Rate Over Time")
    plt.grid()
    plt.show()


def plot_active_inactive_timeline():
    """
    Plots a timeline showing active and inactive periods.
    """
    fig, ax = plt.subplots(figsize=(10, 2))

    # Plot active periods as horizontal bars
    for start, end in active_periods:
        ax.barh(1, end - start, left=start, color='green', height=0.3)

    # Plot inactive periods as horizontal bars
    for start, end in inactive_periods_RE:
        ax.barh(1, end - start, left=start, color='red', height=0.3)

    # Add labels & legend
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])  # Remove y-axis ticks for clarity
    ax.set_title(f"Active vs. Inactive Periods Over Time for Worm")

    active_patch = mpatches.Patch(color='green', label='Active')
    inactive_patch = mpatches.Patch(color='red', label='Inactive')
    ax.legend(handles=[active_patch, inactive_patch])

    plt.show()


def plot_raster(e_times):
    """
    Creates a raster plot of pumping events over time.
    Each event (pump) is marked as a short vertical line.
    """
    plt.figure(figsize=(10, 3))

    # Plot each pump as a vertical tick mark
    for e_time in e_times:
        plt.plot([e_time, e_time], [0, 1], color='black', linewidth=0.8)  # Short vertical line

    plt.xlabel("Time (s)")
    plt.ylabel("Pump Events")
    plt.title("Raster Plot of Pumping Events")
    plt.ylim(0, 1)  # Keep y-axis constant (only one row of events)
    plt.xlim(min(e_times), max(e_times))
    plt.grid(axis='x', linestyle='--', alpha=0.5)  # Optional grid for time reference
    plt.show()

import seaborn as sns

def plot_pump_density(e_times):
    plt.figure(figsize=(10, 3))
    sns.kdeplot(e_times, bw_adjust=0.5, fill=True, color='blue', alpha=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Density")
    plt.title("Density of Pumping Activity Over Time")
    plt.grid()
    plt.show()


def pump_rate_variability():
    pump_rates = [calculate_pump_rate(period)[0] for period in active_periods]
    std_dev = np.std(pump_rates)
    print(f"Pump Rate Variability (Std Dev): {std_dev:.2f} Hz")


def detect_bursts(e_times,burst_threshold=0.5):
    bursts = 0
    consecutive_pumps = 0

    for i in range(len(e_times) - 1):
        if e_times[i + 1] - e_times[i] < burst_threshold:
            consecutive_pumps += 1
            if consecutive_pumps >= 3:
                bursts += 1
        else:
            consecutive_pumps = 0  # Reset if gap too large

    print(f"Number of Pump Bursts: {bursts}")
    return bursts

def compare_early_vs_late(e_times):
    midpoint = e_times[len(e_times) // 2]  # Find middle time point
    early_pumps = [e for e in e_times if e < midpoint]
    late_pumps = [e for e in e_times if e >= midpoint]

    early_rate = len(early_pumps) / (midpoint - e_times[0])
    late_rate = len(late_pumps) / (e_times[-1] - midpoint)

    print(f"Early Pump Rate: {early_rate:.2f} Hz, Late Pump Rate: {late_rate:.2f} Hz")
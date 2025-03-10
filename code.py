import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.mixture import GaussianMixture

def read_data(file_path):
    df = pd.read_csv(file_path)
    sample_rate = 2000
    total_pumps = df["Total pumps"].iloc[0]

    e_locations = df["E_location"].dropna().tolist()
    e_times = [e / sample_rate for e in e_locations]


    r_locations = df["R_location"].dropna().tolist()
    r_times = [r / sample_rate for r in r_locations]

    R_amplitude = df["R_amplitude"].dropna().tolist()
    E_amplitude = df["E_amplitude"].dropna().tolist()

    ipi_data = np.diff(e_locations).reshape(-1, 1)  # E-E intervals
    ipi_data_re = df["RtoE"].dropna().tolist()

    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(ipi_data)

    ipi_threshold = np.mean(gmm.means_)  # Midpoint of the two clusters

    return df, e_locations, e_times, r_locations, total_pumps, sample_rate, ipi_threshold, r_times, ipi_data_re, R_amplitude, E_amplitude

def detect_nonactive_segments(e_locations, r_locations, ipi_threshold, sample_rate):
    nonactive_segments = []

    for i in range(len(e_locations) - 1):
        gap1 = e_locations[i + 1] - e_locations[i]  # E-E gap
        gap2 = e_locations[i + 1] - r_locations[i]  # R-E gap

        if gap2 > ipi_threshold:
            nonactive_segments.append((r_locations[i] / sample_rate, e_locations[i + 1] / sample_rate))
        elif gap1 > ipi_threshold:
            nonactive_segments.append((e_locations[i] / sample_rate, e_locations[i + 1] / sample_rate))

    return nonactive_segments

def list_active_segments(e_locations, r_locations, sample_rate, nonactive_segments):
    active_segments = []

    if not nonactive_segments:
        return [(e_locations[0] / sample_rate, r_locations[-1] / sample_rate)]

    if nonactive_segments[0][0] > e_locations[0] / sample_rate:
        active_segments.append((e_locations[0] / sample_rate, nonactive_segments[0][0]))

    for i in range(len(nonactive_segments) - 1):
        active_segments.append((nonactive_segments[i][1], nonactive_segments[i + 1][0]))

    if nonactive_segments[-1][1] < r_locations[-1] / sample_rate:
        active_segments.append((nonactive_segments[-1][1], r_locations[-1] / sample_rate))

    return active_segments

def calculate_pump_rate(period, e_locations, sample_rate):
    start, end = period
    pumps = [e for e in e_locations if start <= e / sample_rate < end]
    duration = end - start

    return (len(pumps) / duration) if duration > 0 else 0

def find_active_pump_rate(active_periods, e_locations, sample_rate,total_active_time):
    avg_active_pump_rate = 0
    total_weight = 0  # Track total time weight sum

    for period in active_periods:
        rate = calculate_pump_rate(period, e_locations, sample_rate)
        time_weight = (period[1] - period[0]) / total_active_time
        avg_active_pump_rate += rate * time_weight
        total_weight += time_weight  # Track total weight for proper averaging

    return avg_active_pump_rate

def interpump_intervals(e_times):
    intervals = np.diff(e_times)

    if intervals.size == 0:
        print("No inter-pump intervals available.")
        return

    print(f"IPI - Mean: {intervals.mean():.4f}, Median: {np.median(intervals):.4f}, Std Dev: {intervals.std():.4f}")

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

def interpump_intervals_RE(RtoE):
    intervals = RtoE

    if intervals.size == 0:
        print("No inter-pump intervals available.")
        return

    print(f"IPI - Mean: {intervals.mean():.4f}, Median: {np.median(intervals):.4f}, Std Dev: {intervals.std():.4f}")

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

def active_vs_inactive_ratio(active_periods, e_times):
    total_time = e_times[-1] - e_times[0]
    total_active_time = sum(end - start for start, end in active_periods)
    inactive_time = total_time - total_active_time

    active_percentage = (total_active_time / total_time) * 100
    inactive_percentage = (inactive_time / total_time) * 100

    print(f"Active Time: {active_percentage:.2f}%, {total_active_time:.2f}s | Inactive Time: {inactive_percentage:.2f}%, {inactive_time:.2f}s | Total Time: {total_time:.2f}s")

def plot_active_inactive_timeline(active_periods, inactive_periods):
    fig, ax = plt.subplots(figsize=(10, 2))

    for start, end in active_periods:
        ax.barh(1, end - start, left=start, color='green', height=0.3)
    for start, end in inactive_periods:
        ax.barh(1, end - start, left=start, color='red', height=0.3)
    ax.set_yticks([])
    if len(active_periods) == 13:
        ax.set_title("Active vs. Inactive Periods Over Time")
        ax.set_xlabel("Time (s)")
        ax.legend(handles=[
            mpatches.Patch(color='green', label='Active'),
            mpatches.Patch(color='red', label='Inactive')
        ])

    plt.show()

def plot_raster(e_times):
    plt.figure(figsize=(10, 3))
    for e_time in e_times:
        plt.plot([e_time, e_time], [0, 1], color='black', linewidth=0.8)

    plt.xlabel("Time (s)")
    plt.ylabel("Pump Events")
    plt.title("Raster Plot of Pumping Events")
    plt.ylim(0, 1)
    plt.xlim(min(e_times), max(e_times))
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.show()
from numpy import array
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
    "C:/Users/corna/Downloads/389 worms - nicotine1rawxlsx.csv"
]
results = []

for file in files:
    df, e_locations, e_times, r_locations, total_pumps, sample_rate, ipi_threshold, r_times, ipi_data_re = a.read_data(file)

    inactive_periods = a.detect_nonactive_segments(e_locations, r_locations, ipi_threshold, sample_rate)
    active_periods = a.list_active_segments(e_locations, r_locations, sample_rate, inactive_periods)
    total_active_time = 0
    for i in range(len(active_periods)):
        total_active_time += active_periods[i][1] - active_periods[i][0]
    active_pump_rate = a.find_active_pump_rate(active_periods, e_locations, sample_rate,total_active_time)
    avg_active_pump_rate = a.find_active_pump_rate(active_periods, e_locations, sample_rate, total_active_time)

    print(f"File: {file}")
    print(f"Active Pump Rate: {active_pump_rate:.2f} pumps/s")

    a.plot_active_inactive_timeline(active_periods, inactive_periods)
    a.plot_raster(e_times)

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

    burst_frequency = num_active_periods / total_time if total_time > 0 else 0  # How often bursts occur per second


    def calculate_advanced_stats():
        if len(inter_pump_intervals) > 1:
            ipiarray = array(inter_pump_intervals)
            skewness = stats.skew(ipiarray)
            kurtosis = stats.kurtosis(inter_pump_intervals)
            mode_ipi = stats.mode(inter_pump_intervals, keepdims=False)
            cv_ipi = np.std(inter_pump_intervals) / np.mean(inter_pump_intervals) if np.mean(
                inter_pump_intervals) != 0 else np.nan
            entropy_ipi = stats.entropy(np.histogram(inter_pump_intervals, bins=10)[0])

            acf_values = [np.corrcoef(inter_pump_intervals[:-lag], inter_pump_intervals[lag:])[0, 1] for lag in
                          range(1, min(10, len(inter_pump_intervals) - 1))]

            f, psd = welch(inter_pump_intervals, nperseg=min(len(inter_pump_intervals), 256))
            power_spectrum_peak = f[np.argmax(psd)] if len(f) > 0 else np.nan
        else:
            skewness = kurtosis = mode_ipi = cv_ipi = entropy_ipi = power_spectrum_peak = np.nan
            acf_values = []

        mad_active = np.mean(np.abs(active_durations - np.mean(active_durations))) if len(
            active_durations) > 0 else np.nan

        if len(active_durations) > 0 and len(rest_durations) > 0:
            longest_active_ratio = np.max(active_durations) / np.max(rest_durations)
            median_active_to_rest = np.median(active_durations) / np.median(rest_durations)
            time_normalized_activity = len(active_durations) / (sum(active_durations) + sum(rest_durations))
        else:
            longest_active_ratio = median_active_to_rest = time_normalized_activity = np.nan

        return {
            'Skewness_IPI': skewness,
            'Kurtosis_IPI': kurtosis,
            'Mode_IPI': mode_ipi,
            'CV_IPI': cv_ipi,
            'Entropy_IPI': entropy_ipi,
            'ACF_IPI': acf_values[:3],  # First 3 lags only
            'Power_Spectrum_Peak': power_spectrum_peak,
            'MAD_Active': mad_active,
            'Longest_Active_to_Rest_Ratio': longest_active_ratio,
            'Median_Active_to_Rest_Ratio': median_active_to_rest,
            'Time_Normalized_Activity': time_normalized_activity
        }


    extra_stats = calculate_advanced_stats()


    def compute_advanced_stats(datur):

        # Compute Inter-Pump Intervals (IPIs)
        ipis = np.diff(e_times) if len(e_times) > 1 else np.array([])

        # Basic Descriptive Stats
        results.append({
            "total_pumps": len(e_times),
            "total_rests": len(r_times),
            "median_pump_time": np.median(e_times) if len(e_times) > 0 else np.nan,
            "iqr_pump_time": iqr(e_times) if len(e_times) > 0 else np.nan,
            "median_rest_time": np.median(r_times) if len(r_times) > 0 else np.nan,
            "iqr_rest_time": iqr(r_times) if len(r_times) > 0 else np.nan,
            "min_pump_time": np.min(e_times) if len(e_times) > 0 else np.nan,
            "max_pump_time": np.max(e_times) if len(e_times) > 0 else np.nan,
            "min_rest_time": np.min(r_times) if len(r_times) > 0 else np.nan,
            "max_rest_time": np.max(r_times) if len(r_times) > 0 else np.nan,
        })

        # Variability and Entropy Measures
        if len(ipis) > 1:
            results.append({
                "ipi_mean": np.mean(ipis),
                "ipi_median": np.median(ipis),
                "ipi_std": np.std(ipis),
                "ipi_iqr": iqr(ipis),
                "ipi_mode": mode(ipis),
                "ipi_cv": np.std(ipis) / np.mean(ipis) if np.mean(ipis) != 0 else np.nan,
                "ipi_skew": skew(ipis),
                "ipi_kurtosis": kurtosis(ipis),
                "ipi_entropy": entropy(ipis),
            })

            # Temporal Patterning
            results.append({
                "autocorr_ipi": acf(ipis, nlags=1)[1] if len(ipis) > 1 else np.nan,
            })
        '''
            # Kernel Density Estimation (KDE) for IPI distribution
            kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
            kde.fit(ipis[:, None])
            log_density = kde.score_samples(ipis[:, None])
            results["ipi_density_peak"] = np.exp(log_density).max()
        '''
        return stats
    # Store data in a dictionary
    results.append({
        "File": file.split("/")[-1],  # Get filename only
        "Total Pumps": total_pumps,
        "Total Time (s)": total_time,
        "Active Time (s)": total_active_time,
        "Inactive Time (s)": inactive_time,
        "Active Ratio (%)": active_ratio * 100,
        "Inactive Ratio (%)": inactive_ratio * 100,
        "Avg Active Pump Rate": avg_active_pump_rate,
        "Total Active Periods": num_active_periods,
        "Total Rest Periods": num_rest_periods,
        "Avg Active Period Duration (s)": avg_active_duration,
        "Avg Rest Period Duration (s)": avg_rest_duration,
        "Max Active Period Duration (s)": max_active_duration,
        "Max Rest Period Duration (s)": max_rest_duration,
        "Mean IPI (s)": mean_ipi,
        "Median IPI (s)": median_ipi,
        "Std Dev of IPI (s)": std_dev_ipi,
        "Burst Frequency (Hz)": burst_frequency
    })
    results.append(extra_stats)
    #results.append(stats)
    compute_advanced_stats(df)

# Convert results to a DataFrame
df_results = pd.DataFrame(results)
print(df_results)

# Save to CSV
#df_results.to_csv("worms.csv", index=False)

print("Data successfully saved to worms.csv!")
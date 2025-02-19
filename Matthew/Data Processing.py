import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt
from scipy.signal import welch
import matplotlib.pyplot as plt


# file1: Force data (smartwheel)
# file2: Acceleration data (apple watch)
# file3: location data
# utilized thresholding to create a modified force/location file that start at "t=0"
# Utilized a low-pass filter
def process_files(file1, file2, file3):
    # Read CSVs
    df_force = pd.read_csv(file1)
    df_acceleration = pd.read_csv(file2)
    df_location = pd.read_csv(file3)

    # Set sampling frequencies and cutoff for filters
    aw_sampling_freq = 100
    sw_sampling_freq = 240

    # Finding the Cutoff Frequency using Spectral Analysis:
    cutoff_frequency = spectral_analysis(
        df_force["Mz [N*m]"], sw_sampling_freq, "Moment Z Spectral Analysis"
    )
    print(f"Suggested Cutoff Frequency: {cutoff_frequency:.2f} Hz")

    # Butterworth filter coefficients
    b_aw, a_aw = butter(4, 2 * cutoff_freq / aw_sampling_freq, "low")
    b_sw, a_sw = butter(4, 2 * cutoff_freq / sw_sampling_freq, "low")

    # Apply filters to every column of file1 and file2
    for column in df_acceleration.columns:
        df_acceleration[column] = filtfilt(
            b_aw, a_aw, gaussian_filter1d(df_acceleration[column], sigma=50)
        )

    for column in df_force.columns:
        df_force[column] = filtfilt(
            b_sw, a_sw, gaussian_filter1d(df_force[column], sigma=50)
        )

    # Thresholding for file1 (force data)
    idx_force = np.argmax(df_force["Fx [N]"] > 2 * np.mean(df_force["Fx [N]"][:100]))
    df_force = df_force.iloc[idx_force:]
    df_force["t"] = np.arange(0, len(df_force) * (1 / 240), 1 / 240)

    # Save or overwrite the CSV
    df_force.to_csv("modified_force.csv", index=False)

    # Calculate acceleration magnitude for file2
    aw_acc_mag = (
        np.sqrt(
            df_acceleration["x"] ** 2
            + df_acceleration["y"] ** 2
            + df_acceleration["z"] ** 2
        )
        * 9.81
    )
    idx_acceleration = np.argmax(aw_acc_mag > 2 * np.mean(aw_acc_mag[:100]))

    # Get the thresholding point for acceleration
    threshold_seconds = df_acceleration["seconds_elapsed"].iloc[idx_acceleration]

    # Find closest point in file3's 'seconds_elapsed'
    df_location["diff"] = (df_location["seconds_elapsed"] - threshold_seconds).abs()
    closest_row = df_location[df_location["diff"] == df_location["diff"].min()]

    # Remove rows before the thresholding point
    df_location = df_location[
        df_location["seconds_elapsed"] >= closest_row["seconds_elapsed"].values[0]
    ]

    # Create new 't' column
    df_location["t"] = (
        df_location["seconds_elapsed"] - df_location["seconds_elapsed"].iloc[0]
    )
    df_location.drop(
        columns=["diff"], inplace=True
    )  # Drop the 'diff' column, it's no longer needed

    # Save or overwrite the CSV
    df_location.to_csv("modified_location.csv", index=False)


# Spectral Analysis for finding optimal cut-off frequency, with a threshold of 99% (for now)
def spectral_analysis(
    data, sampling_freq, plot_title, nperseg=None, window="hann", threshold=99.9
):
    f, Pxx = welch(data, fs=sampling_freq, nperseg=nperseg, window=window)

    # Calculate cumulative power spectrum
    cumulative_power = np.cumsum(Pxx)

    # Normalize cumulative power to get a percentage
    cumulative_power_percent = cumulative_power / cumulative_power[-1] * 100

    # Find the index where the cumulative power levels off
    cutoff_index = np.argmax(
        cumulative_power_percent > threshold
    )  # Use the specified threshold

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Plot Power Spectral Density
    plt.subplot(2, 1, 1)
    plt.semilogy(f, Pxx)
    plt.title(plot_title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.grid(True)

    # Plot Cumulative Power Spectrum
    plt.subplot(2, 1, 2)
    plt.plot(f, cumulative_power_percent)
    plt.axvline(
        x=f[cutoff_index],
        color="r",
        linestyle="--",
        label=f"Cutoff Frequency: {f[cutoff_index]:.2f} Hz",
    )
    plt.title("Cumulative Power Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Cumulative Power (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return f[cutoff_index]


# Processing raw data to modified data I can use for power calculations:
process_files("Ian_NoFW - Foramt 2.csv", "Accelerometer.csv", "Location.csv")

# Example usage of the "spectral_analysis" function:
# cutoff_frequency = spectral_analysis(df_acceleration['x'], aw_sampling_freq, 'Acceleration Spectral Analysis - x')
# print(f'Suggested Cutoff Frequency: {cutoff_frequency:.2f} Hz')

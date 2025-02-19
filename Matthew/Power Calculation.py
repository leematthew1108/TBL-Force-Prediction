import matplotlib.pyplot as plt
import pandas as pd


def closest_time(time, time_series):
    """Find the closest time in a time series."""
    return time_series.iloc[(time_series - time).abs().argsort()[:1]]


# This is the original function, where it just finds the closest time (method #1)
# file1_path: location csv file
# file2_path: force csv file
# Assume both files start at t=0 (after thresholding)
# diameter in meters (25 inches = 0.635 m)
# Each file has a "t" column that starts at t=0 and is in seconds
def compute_power(file1_path, file2_path, diameter=0.635, save_plot_path=None):
    # Read the CSV files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    power_data = []

    for index, row in df1.iterrows():
        time_location = row["t"]
        speed = row["speed"]

        # Find the closest time in file2
        closest_t = closest_time(time_location, df2["t"]).values[0]
        moment = df2[df2["t"] == closest_t]["Mz [N*m]"].values[0]

        # Calculate power (power = torque * angular velocity)
        power = moment * (speed / diameter)
        power_data.append([time_location, power])

    # Convert to DataFrame for better handling and visualization
    power_df = pd.DataFrame(power_data, columns=["time", "power"])

    # Plotting
    plt.figure()  # Create a new figure
    # plt.plot(power_df['time'], power_df['power'], marker='o', markersize=1, linestyle='', label='Method #1: Power vs Time')
    plt.plot(power_df["time"], power_df["power"], label="Power vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title("Power vs Time: Cut-Off = 11Hz")
    plt.legend()

    if save_plot_path:
        plt.savefig(save_plot_path.replace(".png", "_method1.png"))

    return power_df


def find_force_window(time, time_series, window_size=5):
    """Find a window of force data points around the closest time."""
    closest_indices = (time_series - time).abs().argsort()[:window_size]
    return time_series.iloc[closest_indices]


# Alternative method for time syncing
# Allows for you to select the window size
def compute_power_method2(
    file1_path, file2_path, diameter=0.635, window_size=50, save_plot_path=None
):
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    power_data = []

    for index, row in df1.iterrows():
        time_location = row["t"]
        speed = row["speed"]

        # Find a window of 5 force data points around the closest time in file2
        force_window = find_force_window(time_location, df2["t"], window_size)
        moment = df2[df2["t"].isin(force_window)]["Mz [N*m]"].mean()

        # Calculate power (power = torque * angular velocity)
        power = moment * (speed / diameter)
        power_data.append([time_location, power])

    power_df = pd.DataFrame(power_data, columns=["time", "power"])

    # Plotting
    plt.figure()  # Create a new figure
    # plt.plot(power_df['time'], power_df['power'], marker='o', markersize=1, linestyle='', label='Method #2: Power vs Time')
    plt.plot(power_df["time"], power_df["power"], label="Method #2: Power vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title("Power vs Time - Method #2")
    plt.legend()

    if save_plot_path:
        plt.savefig(save_plot_path.replace(".png", "_method2.png"))

    return power_df


# Time Sync method #3: Everything since the last speed
def compute_power_method3(file1_path, file2_path, diameter=0.635, save_plot_path=None):
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    power_data = []

    for index, row in df1.iloc[1:].iterrows():  # Start from the second element of df1
        time_location = row["t"]
        speed = row["speed"]

        last_speed_time = df1.loc[
            df1.index[index - 1], "t"
        ]  # Get the time of the previous speed data point

        accumulated_moment = 0
        force_count = 0

        # Accumulate moments and count forces since the last speed
        force_subset = df2[(df2["t"] >= last_speed_time) & (df2["t"] < time_location)][
            "Mz [N*m]"
        ]
        accumulated_moment += force_subset.sum()
        force_count += len(force_subset)

        # Calculate power using the average of all forces since the last speed
        if force_count > 0:
            moment_average = accumulated_moment / force_count
            power = moment_average * (speed / diameter)
            power_data.append([time_location, power])

    power_df = pd.DataFrame(power_data, columns=["time", "power"])

    # Plotting and saving the plot (if save_plot_path is provided)
    plt.figure()
    plt.plot(power_df["time"], power_df["power"], label="Method #3")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title("Power vs Time - Method #3")
    plt.legend()

    if save_plot_path:
        plt.savefig(save_plot_path.replace(".png", "_method3.png"))

    return power_df


# Example: Ian Power v Time Data & Visualization
compute_power(
    "modified_location.csv", "modified_force.csv", save_plot_path="cutoff11_plot.png"
)
# compute_power_method2('modified_location.csv', 'modified_force.csv', save_plot_path='method2_plot_lines.png')
# compute_power_method3('modified_location.csv', 'modified_force.csv', save_plot_path='method3_plot_lines.png')


def overlay_plots(
    file1_path, file2_path, diameter=0.635, window_size_method2=50, save_plot_path=None
):
    # Call the existing functions to generate individual plots
    power_df1 = compute_power(file1_path, file2_path, diameter, save_plot_path=None)
    power_df2 = compute_power_method2(
        file1_path, file2_path, diameter, window_size_method2, save_plot_path=None
    )
    power_df3 = compute_power_method3(
        file1_path, file2_path, diameter, save_plot_path=None
    )

    # Create a new figure for the overlay
    plt.figure()

    # Overlay the individual plots
    plt.plot(power_df1["time"], power_df1["power"], label="Method #1")
    plt.plot(
        power_df2["time"],
        power_df2["power"],
        label=f"Method #2 (Window Size: {window_size_method2})",
    )
    plt.plot(power_df3["time"], power_df3["power"], label="Method #3")

    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title("Overlay of Power vs Time for all Methods")
    plt.legend()

    if save_plot_path:
        plt.savefig(save_plot_path.replace(".png", "power_overlay.png"))


# Example: Overlay of all three plots with adjustable window size for Method #2
# overlay_plots('modified_location.csv', 'modified_force.csv', diameter=0.635, window_size_method2=5, save_plot_path='overlay_plot_5.png')
# overlay_plots('modified_location.csv', 'modified_force.csv', diameter=0.635, window_size_method2=50, save_plot_path='overlay_plot_50.png')
# overlay_plots('modified_location.csv', 'modified_force.csv', diameter=0.635, window_size_method2=100, save_plot_path='overlay_plot_100.png')
# overlay_plots('modified_location.csv', 'modified_force.csv', diameter=0.635, window_size_method2=150, save_plot_path='overlay_plot_150.png')
# overlay_plots('modified_location.csv', 'modified_force.csv', diameter=0.635, window_size_method2=200, save_plot_path='overlay_plot_200.png')
# overlay_plots('modified_location.csv', 'modified_force.csv', diameter=0.635, window_size_method2=250, save_plot_path='overlay_plot_250.png')

plt.close()

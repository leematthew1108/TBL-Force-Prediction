import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Create the Tkinter root window
root = tk.Tk()
root.withdraw()

# Prompt the user to select a directory
directory = filedialog.askdirectory()

# Construct file paths
X_path = os.path.join(directory, "X_test.csv")
y_pred_path = os.path.join(directory, "y_pred.csv")
y_true_path = os.path.join(directory, "y_test.csv")

# Load data
X_full = pd.read_csv(X_path)
y_pred_full = pd.read_csv(y_pred_path)
y_truth_full = pd.read_csv(y_true_path)

# Close the Tkinter root window
root.destroy()

# Get unique cycle IDs
unique_ids = X_full['Participant_Cycle_ID'].unique()

# Collect data for each unique cycle ID
X_arrays = [X_full[X_full['Participant_Cycle_ID'] == id].copy() for id in unique_ids]

# Sort data by original index
for i in range(len(X_arrays)):
    X_arrays[i] = X_arrays[i].sort_values('Original_Index')

# Get corresponding predicted and true values
y_pred_arrays = [y_pred_full.iloc[array.index].copy() for array in X_arrays]
y_truth_arrays = [y_truth_full.iloc[array.index].copy() for array in X_arrays]

# Calculate median percentage error for each cycle
errors = []
for i in range(len(y_pred_arrays)):
    error = abs((y_truth_arrays[i] - y_pred_arrays[i]) / y_truth_arrays[i])
    median_error = error.median().mean()  # Compute the mean of median errors across all columns
    errors.append(median_error)

# Identify the cycle with the highest median error
max_error_index = errors.index(max(errors))

# Define sigma for Gaussian smoothing
sigma = 8  # Set sigma to 0 for no smoothing, or use a positive value to apply smoothing

# Optionally apply Gaussian smoothing based on sigma value
if sigma > 0:
    smoothed_data = gaussian_filter1d(y_pred_arrays[max_error_index], sigma=sigma, axis=0)
    smoothed_predictions = pd.DataFrame(smoothed_data, columns=y_pred_arrays[max_error_index].columns)
else:
    smoothed_predictions = y_pred_arrays[max_error_index]

# Plot only the data for the cycle with the highest median error
fig, axs = plt.subplots(6, 1, figsize=(12, 12))
for j, column in enumerate(smoothed_predictions.columns):
    axs[j].scatter(range(1, len(y_truth_arrays[max_error_index]) + 1), y_truth_arrays[max_error_index][column], color='tab:blue', label='True', s=14)
    axs[j].scatter(range(1, len(smoothed_predictions) + 1), smoothed_predictions[column], color='tab:orange', label='Predicted', s=14)
    axs[j].set_ylabel(column)
    if j < 5:
        axs[j].set_xlabel('')  # This removes the x-axis label for all but the last subplot
    else:
        axs[j].set_xlabel('Timestep')  # Only the last subplot will have the x-axis label
    # axs[j].legend()

plt.tight_layout()
plt.show()


import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d

# Create the Tkinter root window
root = tk.Tk()
root.withdraw()

directory = filedialog.askdirectory()

X_path = os.path.join(directory, "X_test.csv")
y_pred_path = os.path.join(directory, "y_pred.csv")
y_true_path = os.path.join(directory, "y_test.csv")

X_full = pd.read_csv(X_path)
y_pred_full = pd.read_csv(y_pred_path)
y_truth_full = pd.read_csv(y_true_path)

# Close the Tkinter root window
root.destroy()

unique_ids = X_full['Participant_Cycle_ID'].unique()

# Create a list of dataframes where each dataframe is the data corresponding to a unique id
X_arrays = [X_full[X_full['Participant_Cycle_ID'] == id].copy() for id in unique_ids]

# Sort each of these arrays by normalized cycle position
for i in range(len(X_arrays)):
    X_arrays[i] = X_arrays[i].sort_values('Original_Index')

y_pred_arrays = [y_pred_full.iloc[array.index].copy() for array in X_arrays]
y_truth_arrays = [y_truth_full.iloc[array.index].copy() for array in X_arrays]

import matplotlib.pyplot as plt

for i in range(len(y_pred_arrays)):
    fig, axs = plt.subplots(6, 1, figsize=(12, 12))
    
    for j, column in enumerate(y_pred_arrays[i].columns):
        axs[j].scatter(range(1, len(y_truth_arrays[i]) + 1), y_truth_arrays[i][column], color='tab:blue', label='True')
        #y_pred_smooth = gaussian_filter1d(y_pred_arrays[i][column], sigma=4)
        axs[j].scatter(range(1, len(y_pred_arrays[i]) + 1), y_pred_arrays[i][column], color='tab:orange', label='Predicted (Smoothed)')
        axs[j].set_xlabel('Data Index')
        axs[j].set_ylabel(column)
        axs[j].legend()
    
    plt.tight_layout()
    plt.show()
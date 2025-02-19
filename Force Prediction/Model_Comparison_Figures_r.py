import os
import pickle
import numpy as np
import csv
import math
import pandas as pd
from scipy.stats import shapiro
from scipy.stats import kruskal
import scikit_posthocs as sp
import seaborn as sns
from scipy.stats import ttest_ind
from scipy import stats
import matplotlib.pyplot as plt
from statannot import add_stat_annotation

# Specify the path to the model folders
os.chdir('./Force Prediction')
model_folders = ['./Linear', './BiLSTM', './CNN', './Dense', './ResNet', './TCN', './GBM', './RF']
key_order = ['Linear', 'BiLSTM', 'CNN', 'Dense', 'ResNet', 'TCN', 'GBM', 'RF']
intensity = [0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.5, 0.5, 0.9, 0.9, 0.5, 0.5, 0.9, 0.5, 0.9, 0.9, 0.5]

# Initialize empty dictionaries to store the loaded data
real_y_dict = {}
radius_predictions_dict = {}
error = {}
error_percent = {}
# Iterate over each model folder
for folder in model_folders:
    model = folder.split('/')[-1]
    # Load the pickle files for real_y and radius_predictions
    real_y_file = os.path.join(folder, f'real_y_{model}.pkl')
    radius_predictions_file = os.path.join(folder, f'radius_predictions_{model}.pkl')    

    with open(real_y_file, 'rb') as f:
        real_y = pickle.load(f)
        real_y_dict[model] = real_y
    
    with open(radius_predictions_file, 'rb') as f:
        radius_predictions = pickle.load(f)
        radius_predictions_dict[model] = radius_predictions

        mape_dict = {}


    real_y = real_y_dict[model]
    radius_predictions = radius_predictions_dict[model]
    
    error[model] = np.empty((radius_predictions.shape[0], 6))
    error_percent[model] = np.empty((radius_predictions.shape[0], 6))
    md_column = np.empty(6)
    std_column = np.empty(6)
    mean_column = np.empty(6)
    columns = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    for i in range(6):
        for j in range(radius_predictions.shape[0]):
            error[model][j, i] = np.median(np.abs((real_y[j, :, i] - radius_predictions[j, :, i]))) # / real_y[j, :, i])) * 100
            error_percent[model][j, i] = np.median(np.abs((real_y[j, :, i] - radius_predictions[j, :, i])/ real_y[j, :, i])) * 100             
             

feature_names = ['Fx [N]', 'Fy [N]', 'Fz [N]', 'Tx [Nm]', 'Ty [Nm]', 'Tz [Nm]']
num_features = len(feature_names)
num_models = len(error)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

# plt.figure(figsize=(24, 12))

# # Create a boxplot for each feature for each model
# for i in range(num_features):
#     for j, model in enumerate(['Linear', 'TCN', 'GBM']):
#         y = error[model][:, i]
#         x = i + 1 + j * 0.1  # offset each model's boxplot for visibility
#         plt.scatter([x]*len(y), y, alpha=0.7, color=colors[j])
#         # plt.boxplot([y], positions=[x], widths=0.05, patch_artist=True, 
#         #             boxprops=dict(color=colors[j], facecolor='None', linewidth=3),
#         #             medianprops=dict(color=colors[j], linewidth=3))
        
# for i in range(num_features):
#     for j, model in enumerate(['Linear', 'TCN', 'GBM']):
#         y = error[model][:, i]
#         x = i + 1 + j * 0.1  # offset each model's boxplot for visibility
#         # plt.scatter([x]*len(y), y, alpha=0.7, color=colors[j])
#         plt.boxplot([y], positions=[x], widths=0.05, patch_artist=True, 
#                     boxprops=dict(color=colors[j], facecolor='None', linewidth=3),
#                     medianprops=dict(color=colors[j], linewidth=3), showfliers=False, whiskerprops=dict(color=colors[j], linewidth=1.5))

# # Set the x-axis labels to be the feature names
# plt.xticks(ticks=np.arange(1, num_features + 1), labels=feature_names)
# # Set the y-axis limits to start at 0
# plt.ylim(bottom=0)
# # Set the title and labels
# plt.title('Boxplot of error for Each Feature for Each Model')
# plt.xlabel('Feature')
# plt.ylabel('r [%]')
# plt.legend(key_order)
# #plt.show()
# # Save the plot to an image file
# plt.savefig('model_comparison_plot_r.png')

# Calculate the mean and standard deviation of the observations of error for each model and feature
# Create a list to store the results
results = []

# Iterate over each model and feature
for model in key_order:
    for i, feature in enumerate(feature_names):
        error_values = error_percent[model][:, i]
        mean = np.mean(error_values)
        std = np.std(error_values)
        
        # Append the results to the list
        results.append([model, feature, mean, std])

# Specify the path to the output file
output_file = './model_comparison_results_r.xlsx'


# Create a DataFrame from the results
df = pd.DataFrame(results, columns=['Model', 'Feature', 'Mean', 'Standard Deviation'])

# Specify the order for the "Model" column
df['Model'] = pd.Categorical(df['Model'], categories=key_order, ordered=True)

# Sort the DataFrame by the "Model" column
df = df.sort_values('Model')

df = df.sort_values(by="Model", key=lambda x: pd.Categorical(x, categories=key_order, ordered=True))
# Round the mean and standard deviation values to two decimal places
df['Mean'] = df['Mean'].round(2)
df['Standard Deviation'] = df['Standard Deviation'].round(2)

# Pivot the DataFrame to have models as rows and features as columns for mean and standard deviation separately
pivot_mean = df.pivot(index='Model', columns='Feature', values='Mean')
pivot_std = df.pivot(index='Model', columns='Feature', values='Standard Deviation')

# Format the pivot DataFrames to have "mean (std)" in each cell
formatted_df = pivot_mean.astype(str) + " (" + pivot_std.astype(str) + ")"

# Save the formatted DataFrame to an Excel file
formatted_df.to_excel(output_file)

print(f"Results saved to {output_file}")

# Perform Shapiro-Wilk test for normality on each feature in each model
normality_results = {}

for model in key_order:
    normality_results[model] = {}
    for i, feature in enumerate(feature_names):
        error_values = error_percent[model][:, i]
        _, p_value = shapiro(error_values)
        normality_results[model][feature] = p_value

# Print the normality test results
for model, features in normality_results.items():
    for feature, p_value in features.items():
        if p_value > 0.05:
            print(f"{model}: {feature} values are normally distributed (p-value: {p_value})")
        else:
            print(f"{model}: {feature} values are not normally distributed (p-value: {p_value})")

# Perform Kruskal-Wallis test for differences between models for each feature

model_comparison_results = {}
for i, feature in enumerate(feature_names):
    model_comparison_results[feature] = {}
    feature_values = []
    for model in key_order:
        error_values = error_percent[model][:, i]
        feature_values.append(error_values)
    _, p_value = kruskal(*feature_values)
    model_comparison_results[feature]['p-value'] = p_value

# Print the model comparison results
for feature, results in model_comparison_results.items():
    p_value = results['p-value']
    if p_value > 0.05:
        print(f"Differences between models for {feature} are not statistically significant (p-value: {p_value})")
    else:
        print(f"Differences between models for {feature} are statistically significant (p-value: {p_value})")



model_comparison_results = {}
for i, feature in enumerate(feature_names):
    model_comparison_results[feature] = {}
    feature_values = []
    for model in key_order:
        error_values = error_percent[model][:, i]
        feature_values.append(error_values)
    _, p_value = kruskal(*feature_values)
    model_comparison_results[feature]['p-value'] = p_value

    # Perform Dunn's test if the Kruskal-Wallis test is significant
    if p_value < 0.05:
        posthoc_results = sp.posthoc_dunn(feature_values, p_adjust='bonferroni')
        posthoc_results.index = key_order
        posthoc_results.columns = key_order
        model_comparison_results[feature]['posthoc'] = posthoc_results

# Print the model comparison results
for feature, results in model_comparison_results.items():
    p_value = results['p-value']
    if p_value > 0.05:
        print(f"Differences between models for {feature} are not statistically significant (p-value: {p_value})")
    else:
        print(f"Differences between models for {feature} are statistically significant (p-value: {p_value})")
        print("Pairwise comparison results:")
        print(results['posthoc'])

        # Create a DataFrame for the significance levels
        significance = pd.DataFrame(index=results['posthoc'].index, columns=results['posthoc'].columns)
        significance[results['posthoc'] <= 0.001] = 3
        significance[(results['posthoc'] > 0.001) & (results['posthoc'] <= 0.01)] = 2
        significance[(results['posthoc'] > 0.01) & (results['posthoc'] <= 0.05)] = 1
        significance[results['posthoc'] > 0.05] = np.nan

        # Convert the DataFrame to float
        significance = significance.astype(float)

        # Create a mask for the annotations
        annot_mask = significance.isnull()

        # Plot the posthoc results
        plt.figure(figsize=(10, 8))
        sns.heatmap(significance, annot=results['posthoc'], fmt=".2g", cmap='Reds', cbar=False, mask=annot_mask, linewidths=1, linecolor='black')
        plt.title(f"Posthoc comparison for {feature}")
        #plt.show()

# Calculate the average performance of each model across all features
average_performance = {}
for model in key_order:
    model_performance = np.mean(error_percent[model].flatten())
    average_performance[model] = model_performance

# Sort the models based on their average performance in ascending order
sorted_models = sorted(average_performance, key=average_performance.get)

# Print the models in order of their average performance
print("Models ranked by average performance:")
for rank, model in enumerate(sorted_models, start=1):
    model_mean = np.mean(error_percent[model])
    model_std = np.std(error_percent[model])
    print(f"Rank {rank}: {model} - Average MAE: {model_mean:.2f}, Standard Deviation: {model_std:.2f}")

# # Compare the models based on their average performance
# model_comparison_results = {}
# feature = 'Average Performance'
# model_comparison_results[feature] = {}
# feature_values = []
# for model in key_order:
#     error_values = np.mean(error_percent[model])
#     feature_values.append(error_percent_values)
# _, p_value = kruskal(*feature_values)
# model_comparison_results[feature]['p-value'] = p_value

# # Print the model comparison results based on average performance
# p_value = model_comparison_results[feature]['p-value']
# if p_value > 0.05:
#     print(f"Differences between models based on average performance are not statistically significant (p-value: {p_value})")
# else:
#     print(f"Differences between models based on average performance are statistically significant (p-value: {p_value})")
#     # Perform Dunn's test if the Kruskal-Wallis test is significant
#     posthoc_results = sp.posthoc_dunn(feature_values, p_adjust='bonferroni')
#     posthoc_results.index = key_order
#     posthoc_results.columns = key_order
#     model_comparison_results[feature]['posthoc'] = posthoc_results
#     print("Pairwise comparison results:")
#     print(model_comparison_results[feature]['posthoc'])


# Define the representative cycle index
cycle_index = np.median(np.median(error['TCN'], axis=0)).argmin()
print(cycle_index)
# Define the features
features = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

# Define the models to plot
models = ['TCN']

# Define the figure size
fig = plt.figure(figsize=(6, 8))

# Iterate over each feature
for i, feature in enumerate(features):
    # Create a subplot for the current feature
    plt.subplot(len(features), 1, i + 1)
    
    # Iterate over each model
    for j, model in enumerate(models):
        # Get the data for the current feature and model
        real_y_data = real_y_dict[model][cycle_index, :, i]
        radius_prediction_data = radius_predictions_dict[model][cycle_index, :, i]
        
        # Plot the real_y data
        if j == 0:
            plt.plot(real_y_data, label='Ground Truth', linewidth=3, color=colors[j])
        
        # Plot the radius_prediction data for the current model
        plt.plot(radius_prediction_data, label=f'Prediction', linewidth=3, color=colors[j+1])
        
        # Set the title and labels for the subplot
        #plt.title(f'{feature}')
        plt.xticks(ticks=[0, 90, 180, 270, 360])
        if i == 5:
            plt.xlabel('Cycle Position [°]')
            plt.xticks(ticks=[0, 90, 180, 270, 360], labels=['0', '90', '180', '270', '360'])
        else:
            plt.xticks(ticks=[0, 90, 180, 270, 360], labels=['', '', '', '', ''])
        plt.ylabel(f'{feature} [N]' if 'F' in feature else f'{feature} [Nm]')  # Fix the line

        plt.xlim(0, 360)
        if i == 0:
            # Add a legend to the subplot
            plt.legend(fontsize='small')

# Adjust the spacing between subplots
plt.tight_layout()
fig.align_ylabels()
# # Show the figure
# plt.show()
# Save the figure
plt.savefig('./Model_Comparison_Representative_Cycle_r.png')

# Define the representative cycle index
cycle_index = np.random.randint(0, 23)
# Define the features
features = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

# Define the models to plot
models = ['GBM']

# Define the figure size
fig = plt.figure(figsize=(6, 8))

# Iterate over each feature
for i, feature in enumerate(features):
    # Create a subplot for the current feature
    plt.subplot(len(features), 1, i + 1)
    
    # Iterate over each model
    for j, model in enumerate(models):
        # Get the data for the current feature and model
        real_y_data = real_y_dict[model][cycle_index, :, i]
        radius_prediction_data = radius_predictions_dict[model][cycle_index, :, i]
        
        # Plot the real_y data
        if j == 0:
            plt.plot(real_y_data, label='Ground Truth', linewidth=3, color=colors[j])
        
        # Plot the radius_prediction data for the current model
        plt.plot(radius_prediction_data, label=f'Prediction', linewidth=3, color=colors[j+1])
        
        # Set the title and labels for the subplot
        #plt.title(f'{feature}')
        plt.xticks(ticks=[0, 90, 180, 270, 360])
        if i == 5:
            plt.xlabel('Cycle Position [°]')
            plt.xticks(ticks=[0, 90, 180, 270, 360], labels=['0', '90', '180', '270', '360'])
        else:
            plt.xticks(ticks=[0, 90, 180, 270, 360], labels=['', '', '', '', ''])
        plt.ylabel(f'{feature} [N]' if 'F' in feature else f'{feature} [Nm]')  # Fix the line

        plt.xlim(0, 360)
        if i == 0:
            # Add a legend to the subplot
            plt.legend(fontsize='small')

# Adjust the spacing between subplots
plt.tight_layout()
fig.align_ylabels()
# # Show the figure
# plt.show()
# Save the figure
plt.savefig('./Model_Comparison_Representative_Cycle_r_ResNet.png')


# Iterate over each model
for model in key_order:
    # Create a new figure
    fig = plt.figure(figsize=(6, 9), dpi=350)

    # Iterate over each feature
    for i, feature in enumerate(features):
        # Create a new subplot for the current feature
        plt.subplot(len(features), 1, i + 1)

        # Calculate the average of real_y data over all cycle_indexes
        avg_real_y = np.mean(real_y_dict[model][:, :, i], axis=0)
        # Calculate the standard deviation of real_y data over all cycle_indexes
        std_real_y = np.std(real_y_dict[model][:, :, i], axis=0)

        # Plot the ribbon for the standard deviation
        plt.fill_between(range(len(avg_real_y)), avg_real_y - std_real_y, avg_real_y + std_real_y, alpha=0.3)

        # Plot the averaged real_y data
        plt.plot(avg_real_y, label='Ground Truth')

        # Calculate the average of radius_predictions data over all cycle_indexes
        avg_radius_predictions = np.mean(radius_predictions_dict[model][:, :, i], axis=0)
        # Calculate the standard deviation of radius_predictions data over all cycle_indexes
        std_radius_predictions = np.std(radius_predictions_dict[model][:, :, i], axis=0)

        # Plot the ribbon for the standard deviation
        plt.fill_between(range(len(avg_radius_predictions)), avg_radius_predictions - std_radius_predictions, avg_radius_predictions + std_radius_predictions, alpha=0.3)

        # Plot the averaged radius_predictions data
        plt.plot(avg_radius_predictions, label=f'Prediction')
        #plt.title(f'{feature[:2]}')
        # Set the title and labels for the subplot
        if i == 0:
            plt.legend(fontsize='small')
        if i == 5:
            plt.xlabel('Cycle Position [°]')
        plt.ylabel(f'{feature} [N]' if 'F' in feature else f'{feature} [Nm]')

        # Adjust the spacing between subplots
        plt.tight_layout()
        # Set the x-axis ticks
        plt.xticks(ticks=[0, 90, 180, 270, 360])
        if i == 5:
            plt.xlabel('Cycle Position [°]')
            plt.xticks(ticks=[0, 90, 180, 270, 360], labels=['0', '90', '180', '270', '360'])
        else:
            plt.xticks(ticks=[0, 90, 180, 270, 360], labels=['', '', '', '', ''])
        plt.xlim(0, 360)
        # Save the figure
        fig.align_ylabels()
        fig.tight_layout()
        plt.savefig(f'Average_Performance_{model}_Model_r.png')
        


# Create a new figure with 2x3 subplots
fig, axes = plt.subplots(2, 3, figsize=(6, 4), dpi=350)
max_y_values = [150, 150, 150, 150, 150, 150]
# Iterate over each feature
for i, feature in enumerate(features):
    # Calculate the row and column index for the current subplot
    row = i // 3
    col = i % 3
    
    # Select the current subplot
    ax = axes[row, col]
    
    # Create a list to store the error values for each model
    error_values = []
    
    # Iterate over each model
    for j, model in enumerate(['Linear', 'TCN', 'GBM']):
        # Get the error values for the current feature and model
        feature_error = error_percent[model][:, i]
        
        # Append the error values to the list
        error_values.append(feature_error)
    
    # Create a swarmplot for the error values of each model
    sns.swarmplot(data=error_values, ax=ax, palette=colors[:3], size=3)
    # plot the mean line
    ax.boxplot(error_values, showmeans=False, meanline=False, 
               medianprops={'visible': True, "color": 'red', 'linewidth': 1, 'linestyle': '-'}, 
               whiskerprops={'visible': False}, 
               capprops={'visible': False}, 
               widths=0.8, patch_artist=True, 
               boxprops=dict(facecolor='none', edgecolor='none'), 
               flierprops={'visible': False}, 
               positions=range(len(error_values)), 
               zorder=10)
    ax.set_title(f'{feature[:2]}')
    if row == 1 and col == 1:
        ax.set_xlabel('Model')
    if col == 0:
        ax.set_ylabel('r [%]')
    ax.set_xticks(range(0, len(['Linear', 'TCN', 'GBM'])))
    if row == 1:
        ax.set_xticklabels(['Linear', 'TCN', 'GBM'], rotation='vertical')
    else:
        ax.set_xticklabels([])
        
    # Set the y-axis to start at 0
    ax.set_ylim(bottom=0)

    # Get the maximum y-value for this subplot from the list
    max_y = max_y_values[i]  # Replace 'i' with the appropriate index

    # Set the y-ticks to be at 0, half of the maximum y-value, and the maximum y-value
    ax.set_yticks([0, 50, 100, 150])
    ax.set_yticklabels([f'{tick:.1f}' if tick % 1 != 0 else f'{tick:.0f}' for tick in ax.get_yticks()])
    plt.subplots_adjust(left=0.1, right=0.98, bottom=0.2, top=0.9, wspace=0.3, hspace=0.25)

# Save the figure
plt.savefig('Model_Error_Boxplots_r.png')
    
    
# Iterate over each model
for model in key_order:
    # Create a new figure
    plt.figure(figsize=(6, 9))

    # Iterate over each feature
    for i, feature in enumerate(features):
        # Create a new subplot for the current feature
        plt.subplot(len(features), 1, i + 1)
        
        err = np.abs((real_y_dict[model][:, :, i] - radius_predictions_dict[model][:, :, i]) / real_y_dict[model][:, :, i]) * 100

        # Calculate the average of real_y data over all cycle_indexes
        avg_error = np.mean(err, axis=0)

        # Calculate the standard deviation of real_y data over all cycle_indexes
        std_error = np.std(err, axis=0)

        # Plot the ribbon for the standard deviation
        plt.fill_between(range(len(avg_error)), avg_error - std_error, avg_error + std_error, alpha=0.3)

        # Plot the averaged real_y data
        plt.plot(avg_error)

        # Set the title and labels for the subplot
        if i == 0:
            plt.title(f'Average Prediction Error for {model} Model')
        if i == 5:
            plt.xlabel('Cycle Position [°]')
        plt.xticks(ticks=[0, 90, 180, 270, 360], labels=['0', '90', '180', '270', '360'])
        plt.ylabel('APE [%]')
        plt.xlim(0, 360)
        # Adjust the spacing between subplots
        plt.tight_layout()

        # Save the figure
        plt.savefig(f'Average_Error_{model}_Model_APE.png')
        
# # Iterate over each model
# for model in key_order:
#     # Create a new figure
#     plt.figure(figsize=(12, 12))

#     # Iterate over each feature
#     for i, feature in enumerate(features):
#         # Create a new subplot for the current feature
#         plt.subplot(len(features), 1, i + 1)
#         mask = np.array(intensity) == 0.5
#         # Calculate the average of real_y data over all cycle_indexes for intensity = 0.5
#         avg_real_y_05 = np.mean(real_y_dict[model][mask, :, i], axis=0)
#         # Calculate the standard deviation of real_y data over all cycle_indexes for intensity = 0.5
#         std_real_y_05 = np.std(real_y_dict[model][mask, :, i], axis=0)

#         # Plot the ribbon for the standard deviation
#         plt.fill_between(range(len(avg_real_y_05)), avg_real_y_05 - std_real_y_05, avg_real_y_05 + std_real_y_05, alpha=0.3)

#         # Plot the averaged real_y data for intensity = 0.5
#         plt.plot(avg_real_y_05, label='Ground Truth (Intensity = 0.5)')

#         # Calculate the average of radius_predictions data over all cycle_indexes for intensity = 0.5
#         avg_radius_predictions_05 = np.mean(radius_predictions_dict[model][mask, :, i], axis=0)
#         # Calculate the standard deviation of radius_predictions data over all cycle_indexes for intensity = 0.5
#         std_radius_predictions_05 = np.std(radius_predictions_dict[model][mask, :, i], axis=0)

#         # Plot the ribbon for the standard deviation
#         plt.fill_between(range(len(avg_radius_predictions_05)), avg_radius_predictions_05 - std_radius_predictions_05, avg_radius_predictions_05 + std_radius_predictions_05, alpha=0.3)

#         # Plot the averaged radius_predictions data for intensity = 0.5
#         plt.plot(avg_radius_predictions_05, label=f'Prediction ({model} Model, Intensity = 0.5)')

#         # Set the title and labels for the subplot
#         if i == 0:
#             plt.title(f'Average Prediction Performance for {model} Model (Intensity = 0.5)')
#             plt.legend()
#         if i == 5:
#             plt.xlabel('Timestep')
#         plt.ylabel('r [%]')

#         # Adjust the spacing between subplots
#         plt.tight_layout()

#         # Save the figure
#         plt.savefig(f'Average_Performance_{model}_Model_Intensity_05_r.png')

#     # Create a new figure
#     plt.figure(figsize=(12, 12))

#     # Iterate over each feature
#     for i, feature in enumerate(features):
#         # Create a new subplot for the current feature
#         plt.subplot(len(features), 1, i + 1)
#         mask = np.array(intensity) == 0.9
#         # Calculate the average of real_y data over all cycle_indexes for intensity = 0.9
#         avg_real_y_09 = np.mean(real_y_dict[model][mask, :, i], axis=0)
#         # Calculate the standard deviation of real_y data over all cycle_indexes for intensity = 0.9
#         std_real_y_09 = np.std(real_y_dict[model][mask, :, i], axis=0)

#         # Plot the ribbon for the standard deviation
#         plt.fill_between(range(len(avg_real_y_09)), avg_real_y_09 - std_real_y_09, avg_real_y_09 + std_real_y_09, alpha=0.3)

#         # Plot the averaged real_y data for intensity = 0.9
#         plt.plot(avg_real_y_09, label='Ground Truth (Intensity = 0.9)')

#         # Calculate the average of radius_predictions data over all cycle_indexes for intensity = 0.9
#         avg_radius_predictions_09 = np.mean(radius_predictions_dict[model][mask, :, i], axis=0)
#         # Calculate the standard deviation of radius_predictions data over all cycle_indexes for intensity = 0.9
#         std_radius_predictions_09 = np.std(radius_predictions_dict[model][mask, :, i], axis=0)

#         # Plot the ribbon for the standard deviation
#         plt.fill_between(range(len(avg_radius_predictions_09)), avg_radius_predictions_09 - std_radius_predictions_09, avg_radius_predictions_09 + std_radius_predictions_09, alpha=0.3)

#         # Plot the averaged radius_predictions data for intensity = 0.9
#         plt.plot(avg_radius_predictions_09, label=f'Prediction ({model} Model, Intensity = 0.9)')

#         # Set the title and labels for the subplot
#         if i == 0:
#             plt.title(f'Average Prediction Performance for {model} Model (Intensity = 0.9)')
#             plt.legend()
#         if i == 5:
#             plt.xlabel('Timestep')
#         plt.ylabel('r [%]')

#         # Adjust the spacing between subplots
#         plt.tight_layout()

#         # Save the figure
#         plt.savefig(f'Average_Performance_{model}_Model_Intensity_09_r.png')


errors_mict = {}
mask = np.array(intensity) == 0.5
for model in key_order:   
    errors_mict[model] = error_percent[model][mask]
errors_hiit = {}
mask = np.array(intensity) == 0.9
for model in key_order:   
    errors_hiit[model] = error_percent[model][mask]
    
# Perform Shapiro-Wilk test for normality on each feature for the TCN model
normality_results = {}

for feature in feature_names:
    # Ensure that 'feature' is an integer index or slice
    feature_index = feature_names.index(feature)
    
    # Use the integer index to index the numpy array
    _, p_value = shapiro(errors_mict['ResNet'][:, feature_index])

# Perform a t-test to compare the HIIT error and MICT error for the TCN model for each feature
for feature in feature_names:
    feature_index = feature_names.index(feature)
    mict_error = errors_mict['ResNet'][:, feature_index]
    hiit_error = errors_hiit['ResNet'][:, feature_index]

    t_statistic, p_value = ttest_ind(mict_error, hiit_error)
    statistic, p_value = stats.wilcoxon(mict_error, hiit_error)
    print(p_value)
    if p_value < 0.05:
        print(f"HIIT error is significantly different from MICT error for the TCN model for feature {feature}")
    else:
        print(f"HIIT error is not significantly different from MICT error for the TCN model for feature {feature}")
     
fig, axes = plt.subplots(2, 3, figsize=(6, 4), dpi=350)
# max_y_values = [20, 30, 10, 2, 1, 0.4]
# Iterate over each feature
for i, feature in enumerate(features):
    # Create a new subplot for the current feature Calculate the row and column index for the current subplot
    # Calculate the row and column index for the current subplot
    row = i // 3
    col = i % 3
    
    # Select the current subplot
    ax = axes[row, col]
    
    # Create a swarmplot for the error values of each model
    sns.swarmplot(data=[errors_mict['TCN'][:,i], errors_hiit['TCN'][:,i]], ax=ax, palette=colors[:2], size=4)
    # plot the mean line
    # Create a boxplot for the MICT and HIIT errors
    ax.boxplot([errors_mict['TCN'][:,i], errors_hiit['TCN'][:,i]], showmeans=False, meanline=False, 
               medianprops={'visible': True, "color": 'red', 'linewidth': 1, 'linestyle': '-'}, 
               whiskerprops={'visible': False}, 
               capprops={'visible': False}, 
               widths=0.8, patch_artist=True, 
               boxprops=dict(facecolor='none', edgecolor='none'), 
               flierprops={'visible': False}, 
               positions=range(2), 
               zorder=10)
    # # Perform the Wilcoxon test
    # stat, p = stats.wilcoxon(errors_mict['TCN'][:,i], errors_hiit['TCN'][:,i])
    # # Significance bars
    # if p < 0.05:
    #     # Columns corresponding to the datasets of interest
    #     x1 = 0
    #     x2 = 1
    #     # What level is this bar among the bars above the plot?
    #     level = 1
    #     bottom, top = ax.get_ylim()
    #     y_range = top - bottom
    #     # Plot the bar
    #     bar_height = (y_range * 0.07 * level) + top
    #     bar_tips = bar_height - (y_range * 0.02)
    #     plt.plot(
    #         [x1, x1, x2, x2],
    #         [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k'
    #     )
    #     # Significance level
    #     if p < 0.001:
    #         sig_symbol = '***'
    #     elif p < 0.01:
    #         sig_symbol = '**'
    #     elif p < 0.05:
    #         sig_symbol = '*'
    #     text_height = bar_height + (y_range * 0.01)
    #     plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')

    ax.set_title(f'{feature[:2]}')
    if row == 1 and col == 1:
            ax.set_xlabel('Intensity')
    if col == 0:
        ax.set_ylabel('r [%]')
    if row == 1:
        ax.set_xticklabels(['MICT', 'HIIT'], rotation='horizontal')
    else:
        ax.set_xticklabels([])
        
    # Set the y-axis to start at 0
    ax.set_ylim(bottom=0)

    # Get the maximum y-value for this subplot from the list
    max_y = max_y_values[i]  # Replace 'i' with the appropriate index

    # Set the y-ticks to be at 0, half of the maximum y-value, and the maximum y-value
    ax.set_yticks([0, max_y / 2, max_y])
    ax.set_yticklabels([f'{tick:.1f}' if tick % 1 != 0 else f'{tick:.0f}' for tick in ax.get_yticks()])

# Adjust the spacing between subplots
plt.tight_layout()

# Save the figure
plt.savefig('Error_Comparison_Intensity_Boxplots_r_TCN.png')


fig, axes = plt.subplots(2, 3, figsize=(6, 4), dpi=350)
# max_y_values = [20, 30, 10, 2, 1, 0.4]
# Iterate over each feature
for i, feature in enumerate(features):
    # Create a new subplot for the current feature Calculate the row and column index for the current subplot
    # Calculate the row and column index for the current subplot
    row = i // 3
    col = i % 3
    
    # Select the current subplot
    ax = axes[row, col]
    
    # Create a swarmplot for the error values of each model
    sns.swarmplot(data=[errors_mict['TCN'][:,i], errors_hiit['TCN'][:,i]], ax=ax, palette=colors[:2], size=4)
    # plot the mean line
    # Create a boxplot for the MICT and HIIT errors
    ax.boxplot([errors_mict['TCN'][:,i], errors_hiit['TCN'][:,i]], showmeans=False, meanline=False, 
               medianprops={'visible': True, "color": 'red', 'linewidth': 1, 'linestyle': '-'}, 
               whiskerprops={'visible': False}, 
               capprops={'visible': False}, 
               widths=0.8, patch_artist=True, 
               boxprops=dict(facecolor='none', edgecolor='none'), 
               flierprops={'visible': False}, 
               positions=range(2), 
               zorder=10)
    # # Perform the Wilcoxon test
    # stat, p = stats.wilcoxon(errors_mict['TCN'][:,i], errors_hiit['TCN'][:,i])
    # # Significance bars
    # if p < 0.05:
    #     # Columns corresponding to the datasets of interest
    #     x1 = 0
    #     x2 = 1
    #     # What level is this bar among the bars above the plot?
    #     level = 1
    #     bottom, top = ax.get_ylim()
    #     y_range = top - bottom
    #     # Plot the bar
    #     bar_height = (y_range * 0.07 * level) + top
    #     bar_tips = bar_height - (y_range * 0.02)
    #     plt.plot(
    #         [x1, x1, x2, x2],
    #         [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k'
    #     )
    #     # Significance level
    #     if p < 0.001:
    #         sig_symbol = '***'
    #     elif p < 0.01:
    #         sig_symbol = '**'
    #     elif p < 0.05:
    #         sig_symbol = '*'
    #     text_height = bar_height + (y_range * 0.01)
    #     plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')

    ax.set_title(f'{feature[:2]}')
    if row == 1 and col == 1:
            ax.set_xlabel('Intensity')
    if col == 0:
        ax.set_ylabel('r [%]')
    if row == 1:
        ax.set_xticklabels(['MICT', 'HIIT'], rotation='horizontal')
    else:
        ax.set_xticklabels([])
        
    # Set the y-axis to start at 0
    ax.set_ylim(bottom=0)

    # Get the maximum y-value for this subplot from the list
    max_y = max_y_values[i]  # Replace 'i' with the appropriate index

    # Set the y-ticks to be at 0, half of the maximum y-value, and the maximum y-value
    ax.set_yticks([0, max_y / 2, max_y])
    ax.set_yticklabels([f'{tick:.1f}' if tick % 1 != 0 else f'{tick:.0f}' for tick in ax.get_yticks()])

# Adjust the spacing between subplots
plt.tight_layout()

# Save the figure
plt.savefig('Error_Comparison_Intensity_Boxplots_r_TCN.png')
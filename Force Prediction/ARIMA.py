## Package Dependencies
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import KFold
import pickle
import matplotlib.pyplot as plt

## Functions

def read_and_process_data(directory_path):
    data_frames = []
    columns_to_extract = ['radius_X', 'radius_Y', 'radius_Z', 'radius_Ox', 'radius_Oy', 'radius_Oz', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

    # Assuming each cycle has exactly 356 data points
    total_data_points = 356

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(directory_path, file_name)
            participant = int(os.path.basename(file_path).split('_')[0])
            cycle_id = os.path.basename(file_path).split('_')[1].split('.')[0]  # Extract cycle_id
            intensity = cycle_id[:4]  # Extract the first four characters of cycle_id as intensity

            # Read data from CSV and select only the desired columns
            df = pd.read_csv(file_path, usecols=columns_to_extract)

            # Add participant ID, cycle_id, and participant_cycle_id as features
            df['Participant'] = participant
            df['Cycle_ID'] = cycle_id
            df['Participant_Cycle_ID'] = f"{participant}_{cycle_id}"

            # Add normalized_cycle_position
            df['Normalized_Cycle_Position'] = df.index / (total_data_points - 1)
            df['Intensity'] = intensity  # this is either "HIIT" or "MICT"

            data_frames.append(df)

    # Concatenate all data frames
    processed_data = pd.concat(data_frames, ignore_index=True)

    # Merge with participant weights
    weights_df = pd.read_csv("./Force Prediction/Participant_Weights_Median_Imputed.csv")
    weights_df['Weight'] = weights_df['Weight'].astype(float)
    weights_df['Wingspan'] = weights_df['Wingspan'].astype(float)
    processed_data = pd.merge(processed_data, weights_df, left_on='Participant', right_on='Participant')

    return processed_data

import statsmodels.api as sm

def get_arima_model(feature):
    # Define the order of the ARIMA model
    order = (1, 0, 0)  # (p, d, q)

    # Create the ARIMA model
    model = sm.tsa.ARIMA(endog=train_y[feature], order=order)

    return model

## 

 # Set up paths
data_directory = "./Force Prediction/Processed Data for ML"

# Read and process data
data = read_and_process_data(data_directory)

os.chdir('Force Prediction/K-Fold ResNet')
data['Intensity'] = data['Intensity'].map({'HIIT': 0.9, 'MICT': 0.5})
data = data.drop(columns=['Cycle_ID', 'Participant_Cycle_ID', 'Normalized_Cycle_Position'])
data = data[data.columns[-3:].tolist() + data.columns[:-3].tolist()]

# Get unique participant IDs
participants = data['Participant'].unique()

# Create KFold object
kf = KFold(n_splits=10, shuffle=True, random_state=1)

model = [None] * 10
model_history = [None] * 10
radius_predictions = [None] * 10
real_y = [None] * 10
mdpe = [None] * 10
md_column = [None] * 10
std_column = [None] * 10
mean_column = [None] * 10


# Perform 10-fold cross-validation
for fold, (train_val_index, test_index) in enumerate(kf.split(participants)):
    train_val_participants = participants[train_val_index]
    test_participants = participants[test_index]

    # Further split train_val set into train and validation sets
    train_participants = train_val_participants[:int(0.9*len(train_val_participants))]
    val_participants = train_val_participants[int(0.9*len(train_val_participants)):]

    # Create train, validation and test sets
    train_data = data[data['Participant'].isin(train_participants)].drop(columns='Participant')
    validation_data = data[data['Participant'].isin(val_participants)].drop(columns='Participant')
    test_data = data[data['Participant'].isin(test_participants)].drop(columns='Participant')

    # Reshape and shuffle train_data, val_data and test_data as before...
    print(train_data.shape, validation_data.shape, test_data.shape)
    train_data2 = np.empty([int(len(train_data)/356),356,len(train_data.columns)])
    for i in range(int(len(train_data)/356)):
        train_data2[i,:,:] = train_data[356*i:i*356+356]
    np.random.shuffle(train_data2)
    print(train_data2.shape)

    test_data2 = np.empty([int(len(test_data)/356),356,len(test_data.columns)])
    for i in range(int(len(test_data)/356)):
        test_data2[i,:,:] = test_data[356*i:i*356+356]
    np.random.shuffle(test_data2)
    print(test_data2.shape)

    val_data2 = np.empty([int(len(validation_data)/356),356,len(validation_data.columns)])
    for i in range(int(len(validation_data)/356)):
        val_data2[i,:,:] = validation_data[356*i:i*356+356]
    np.random.shuffle(val_data2)
    print(val_data2.shape)
    
    # Initialize new StandardScaler instances
    train_scaler_X = StandardScaler()
    train_scaler_y = StandardScaler()
    val_scaler_X = StandardScaler()
    val_scaler_y = StandardScaler()
    test_scaler_X = StandardScaler()
    test_scaler_y = StandardScaler()

    # Separate X and y
    train_X = train_data2[:, :, :-6]
    train_y = train_data2[:, :, -6:]
    val_X = val_data2[:, :, :-6]
    val_y = val_data2[:, :, -6:]
    test_X = test_data2[:, :, :-6]
    test_y = test_data2[:, :, -6:]

     # Reshape data to fit ARIMA model's requirements
    train_X = train_X.reshape(-1, train_X.shape[-1])
    train_y = train_y.reshape(-1, train_y.shape[-1])
    test_X = test_X.reshape(-1, test_X.shape[-1])
    test_y = test_y.reshape(-1, test_y.shape[-1])

    # Create and train ARIMA model
    for i in range(6):
        if model[fold] is None:
            model[fold] = [None]*6
            model[fold][i] = get_arima_model(feature=i)
            print(train_X[:, i].flatten().shape, train_y[:, i].flatten().shape)
            model[fold][i].fit(train_X[:, i].flatten(), train_y[:, i].flatten())  # Ensure train_X is a 1D array

        # Predict outputs for the entire test set
        radius_predictions[fold][i] = model[fold][i].predict(test_X[:, i].flatten())  # Ensure test_X is a 1D array
        real_y[fold][i] = test_y[:, i]
        
        if len(radius_predictions[fold][i].shape) == 1:
            radius_predictions[fold][i] = np.expand_dims(radius_predictions[fold][i], axis=1)
        
        if len(real_y[fold][i].shape) == 1:
            real_y[fold][i] = np.expand_dims(real_y[fold][i], axis=1)

    # Predict outputs for the entire test set
    radius_predictions[fold] = model[fold].predict(test_X)
    real_y[fold] = test_y
    
    if len(radius_predictions[fold].shape) == 2:
        radius_predictions[fold] = np.expand_dims(radius_predictions[fold], axis=2)
    
    if len(real_y[fold].shape) == 2:
        real_y[fold] = np.expand_dims(real_y[fold], axis=2)
    
    mdpe[fold] = np.empty([radius_predictions[fold].shape[0],6])
    md_column[fold] = np.empty(6)
    std_column[fold] = np.empty(6)
    mean_column[fold] = np.empty(6)
    for i in range(6):
        for j in range(radius_predictions[fold].shape[0]):
            mdpe[fold][j,i] = np.median(np.abs(real_y[fold][j,:,i]-radius_predictions[fold][j,:,i])/real_y[fold][j,:,i])*100
        md_column[fold][i] = np.median(mdpe[fold][:,i])
        std_column[fold][i] = np.std(mdpe[fold][:,i])
        mean_column[fold][i] = np.mean(mdpe[fold][:,i])
        print(f"Median MDPE for {data.columns[i+9]}: {md_column[fold][i]}")
        print(f"Mean MDPE for {data.columns[i+9]}: {mean_column[fold][i]}")
        print(f"Standard Deviation of MDPE for {data.columns[i+9]}: {std_column[fold][i]}")
    mean_mdpe = np.mean(mdpe[fold], axis=1)
    lowest_mean_slice = np.argmax(mean_mdpe)
    print(lowest_mean_slice)

    # Create a list of feature names
    feature_names = ['Fx [N]', 'Fy [N]', 'Fz [N]', 'Tx [N*m]', 'Ty [N*m]', 'Tz [N*m]']
    plt.figure(figsize=(12,12))
    # Create a scatter plot for each feature
    for i in range(1, len(mdpe[fold][0]) + 1):
        y = mdpe[fold][:, i-1]
        x = np.random.normal(i, 0.04, size=len(y))  # add jitter to the x-axis
        plt.scatter(x, y, alpha=0.7, color='tab:blue')

    # Create a boxplot for each feature without color
    bp = plt.boxplot(mdpe[fold], labels=feature_names, patch_artist=True, 
                    positions=range(1, len(mdpe[fold][0]) + 1), showfliers=False,
                    boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.5),
                    medianprops=dict(color='black', linewidth=1.5))

    # Set the title and labels
    plt.title(f'Boxplot of MDPE for Each Feature: Fold {fold}')
    plt.xlabel('Feature')
    plt.ylabel('MDPE [%]')

    # Show the plot
    plt.savefig(f'Boxplot_MDPE_Fold_{fold}_ResNet_concat.png')
    # Plot to visualize ground truth of one cycle with predictions overlaid - quick interpretation of model accuracy
    cycle = lowest_mean_slice

    plt.figure(figsize=(12,12))
    plt.subplot(611)
    plt.plot(np.arange(real_y[fold].shape[1]),real_y[fold][cycle,:,0],'.')
    plt.plot(np.arange(real_y[fold].shape[1]),radius_predictions[fold][0,:,0],'.')
    #plt.plot(np.arange(testset_y.shape[1]),resnet_predictions[0,:,0],'.')
    plt.xlabel('Timestep')
    plt.ylabel('Fx [N]')
    plt.legend(['Actual','Radius Predicted'])
    plt.xlim([0,360])
    plt.subplot(612)
    plt.plot(np.arange(real_y[fold].shape[1]),real_y[fold][cycle,:,1],'.')
    plt.plot(np.arange(real_y[fold].shape[1]),radius_predictions[fold][0,:,1],'.')
    #plt.plot(np.arange(testset_y.shape[1]),resnet_predictions[0,:,1],'.')
    plt.xlabel('Timestep')
    plt.ylabel('Fy [N]')
    plt.xlim([0,360])
    plt.subplot(613)
    plt.plot(np.arange(real_y[fold].shape[1]),real_y[fold][cycle,:,2],'.')
    plt.plot(np.arange(real_y[fold].shape[1]),radius_predictions[fold][0,:,2],'.')
    #plt.plot(np.arange(testset_y.shape[1]),resnet_predictions[0,:,2],'.')
    plt.xlabel('Timestep')
    plt.ylabel('Fz [N]')
    plt.xlim([0,360])
    plt.subplot(614)
    plt.plot(np.arange(real_y[fold].shape[1]),real_y[fold][cycle,:,3],'.')
    plt.plot(np.arange(real_y[fold].shape[1]),radius_predictions[fold][0,:,3],'.')
    #plt.plot(np.arange(testset_y.shape[1]),resnet_predictions[0,:,3],'.')``
    plt.xlabel('Timestep')
    plt.ylabel('Tx [Nm]')
    plt.xlim([0,360])
    plt.subplot(615)
    plt.plot(np.arange(real_y[fold].shape[1]),real_y[fold][cycle,:,4],'.')
    plt.plot(np.arange(real_y[fold].shape[1]),radius_predictions[fold][0,:,4],'.')
    #plt.plot(np.arange(testset_y.shape[1]),resnet_predictions[0,:,4],'.')
    plt.xlabel('Timestep')
    plt.ylabel('Ty [Nm]')
    plt.xlim([0,360])
    plt.subplot(616)
    plt.plot(np.arange(real_y[fold].shape[1]),real_y[fold][cycle,:,5],'.')
    plt.plot(np.arange(real_y[fold].shape[1]),radius_predictions[fold][0,:,5],'.')
    #plt.plot(np.arange(testset_y.shape[1]),resnet_predictions[0,:,5],'.')
    plt.xlabel('Timestep')
    plt.ylabel('Tz [Nm]')
    plt.xlim([0,360])
    plt.savefig(f'Highest_Error_Cycle_Fold{fold}_ResNet_concat.png')
    
    mean_real_y = np.mean(real_y[fold], axis=0)
    mean_radius_predictions = np.mean(radius_predictions[fold], axis=0)

    plt.figure(figsize=(20,12))

    plt.subplot(611)
    plt.plot(np.arange(real_y[fold].shape[1]), mean_real_y[:,0], '.', markersize=2)
    plt.plot(np.arange(real_y[fold].shape[1]), mean_radius_predictions[:,0], '.', markersize=2)
    #plt.fill_between(np.arange(test_y.shape[1]), mean_real_y[:,0] - std_y[0], mean_real_y[:,0] + std_y[0], alpha=0.3)
    #plt.fill_between(np.arange(test_y.shape[1]), mean_radius_predictions[:,0] - std_y[0], mean_radius_predictions[:,0] + std_y[0], alpha=0.3)
    plt.xlabel('Timestep')
    plt.ylabel('Fx [N]')
    plt.legend(['Ground Truth (Mean)', 'Radius Predicted (Mean)'])
    plt.xlim([0,360])

    plt.subplot(612)
    plt.plot(np.arange(real_y[fold].shape[1]), mean_real_y[:,1], '.', markersize=2)
    plt.plot(np.arange(real_y[fold].shape[1]), mean_radius_predictions[:,1], '.', markersize=2)
    #plt.fill_between(np.arange(test_y.shape[1]), mean_real_y[:,1] - std_y[1], mean_real_y[:,1] + std_y[1], alpha=0.3)
    #plt.fill_between(np.arange(test_y.shape[1]), mean_radius_predictions[:,1] - std_y[1], mean_radius_predictions[:,1] + std_y[1], alpha=0.3)
    plt.xlabel('Timestep')
    plt.ylabel('Fy [N]')
    plt.xlim([0,360])

    plt.subplot(613)
    plt.plot(np.arange(real_y[fold].shape[1]), mean_real_y[:,2], '.', markersize=2)
    plt.plot(np.arange(real_y[fold].shape[1]), mean_radius_predictions[:,2], '.', markersize=2)
    #plt.fill_between(np.arange(test_y.shape[1]), mean_real_y[:,2] - std_y[2], mean_real_y[:,2] + std_y[2], alpha=0.3)
    #plt.fill_between(np.arange(test_y.shape[1]), mean_radius_predictions[:,2] - std_y[2], mean_radius_predictions[:,2] + std_y[2], alpha=0.3)
    plt.xlabel('Timestep')
    plt.ylabel('Fz [N]')
    plt.xlim([0,360])

    plt.subplot(614)
    plt.plot(np.arange(real_y[fold].shape[1]), mean_real_y[:,3], '.', markersize=2)
    plt.plot(np.arange(real_y[fold].shape[1]), mean_radius_predictions[:,3], '.', markersize=2)
    #plt.fill_between(np.arange(test_y.shape[1]), mean_real_y[:,3] - std_y[3], mean_real_y[:,3] + std_y[3], alpha=0.3)
    #plt.fill_between(np.arange(test_y.shape[1]), mean_radius_predictions[:,3] - std_y[3], mean_radius_predictions[:,3] + std_y[3], alpha=0.3)
    plt.xlabel('Timestep')
    plt.ylabel('Tx [Nm]')
    plt.xlim([0,360])

    plt.subplot(615)
    plt.plot(np.arange(real_y[fold].shape[1]), mean_real_y[:,4], '.', markersize=2)
    plt.plot(np.arange(real_y[fold].shape[1]), mean_radius_predictions[:,4], '.', markersize=2)
    #plt.fill_between(np.arange(test_y.shape[1]), mean_real_y[:,4] - std_y[4], mean_real_y[:,4] + std_y[4], alpha=0.3)
    #plt.fill_between(np.arange(test_y.shape[1]), mean_radius_predictions[:,4] - std_y[4], mean_radius_predictions[:,4] + std_y[4], alpha=0.3)
    plt.xlabel('Timestep')
    plt.ylabel('Ty [Nm]')
    plt.xlim([0,360])

    plt.subplot(616)
    plt.plot(np.arange(real_y[fold].shape[1]), mean_real_y[:,5], '.', markersize=2)
    plt.plot(np.arange(real_y[fold].shape[1]), mean_radius_predictions[:,5], '.', markersize=2)
    #plt.fill_between(np.arange(test_y.shape[1]), mean_real_y[:,5] - std_y[5], mean_real_y[:,5] + std_y[5], alpha=0.3)
    #plt.fill_between(np.arange(test_y.shape[1]), mean_radius_predictions[:,5] - std_y[5], mean_radius_predictions[:,5] + std_y[5], alpha=0.3)
    plt.xlabel('Timestep')
    plt.ylabel('Tz [Nm]')
    plt.xlim([0,360])

    plt.savefig(f'Mean_Cycle_Fold{fold}_ResNet_concat.png')
    
# Save variables to files
# Save
with open('radius_predictions_ResNet_concat.pkl', 'wb') as f:
    pickle.dump(radius_predictions, f)

with open('real_y_ResNet_concat.pkl', 'wb') as f:
    pickle.dump(real_y, f)

with open('mdpe_ResNet_concat.pkl', 'wb') as f:
    pickle.dump(mdpe, f)

with open('md_column_ResNet_concat.pkl', 'wb') as f:
    pickle.dump(md_column, f)

with open('std_column_ResNet_concat.pkl', 'wb') as f:
    pickle.dump(std_column, f)

with open('mean_column_ResNet_concat.pkl', 'wb') as f:
    pickle.dump(mean_column, f)
    
with open('model_history_ResNet_concat.pkl', 'wb') as f:
    pickle.dump(model_history.history, f)

for i, model in enumerate(model):
    model.save(f'model_ResNet_concat_fold{i}.h5')
## To load models
# loaded_models = []
# for i in range(len(models)):
#     loaded_models.append(tf.keras.models.load_model(f'model_CNN_concat_{i}.h5'))
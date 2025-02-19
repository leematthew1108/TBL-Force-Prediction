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

 # Set up paths
data_directory = "./Force Prediction/Processed Data for ML"

# Read and process data
data = read_and_process_data(data_directory)

data['Intensity'] = data['Intensity'].map({'HIIT': 0.9, 'MICT': 0.5})
data = data.drop(columns=['Cycle_ID', 'Participant_Cycle_ID', 'Normalized_Cycle_Position'])
data = data[data.columns[-3:].tolist() + data.columns[:-3].tolist()]

# Get unique participant IDs
participants = data['Participant'].unique()

# Load participant splits from text files
train_splits = []
test_splits = []
for fold in range(1, 11):
    with open(rf'./Force Prediction/Train and Test Data 10 Fold/Fold_{fold}/train_participants.txt', 'r') as f:
        train_splits.append([int(index) for index in f.read().splitlines()])
    with open(rf'./Force Prediction/Train and Test Data 10 Fold/Fold_{fold}/test_participants.txt', 'r') as f:
        test_splits.append([int(index) for index in f.read().splitlines()])

print(train_splits[0])
print(test_splits[0])
# Perform 10-fold cross-validation


for fold in range(1):
    train_participants = train_splits[fold]
    test_participants = test_splits[fold]
    full_train_data = data[data['Participant'].isin(train_participants)].drop(columns='Participant')

    # Reshape and shuffle full_train_data
    full_train_data_reshaped = np.empty([int(len(full_train_data)/356), 356, len(full_train_data.columns)])
    for i in range(int(len(full_train_data)/356)):
        full_train_data_reshaped[i,:,:] = full_train_data[356*i:i*356+356]
    np.random.shuffle(full_train_data_reshaped)
    print(full_train_data_reshaped.shape)

    # Further split train set into train and validation sets
    train_participants = train_participants[:int(0.9*len(train_participants))]
    val_participants = train_participants[int(0.9*len(train_participants)):]

    # Create train, validation and test sets
    train_data = data[data['Participant'].isin(train_participants)].drop(columns='Participant')
    validation_data = data[data['Participant'].isin(val_participants)].drop(columns='Participant')
    test_data = data[data['Participant'].isin(test_participants)].drop(columns='Participant')

    # Reshape and shuffle train_data, val_data and test_data as before...
    print(train_data.shape, validation_data.shape, test_data.shape)
    train_data_reshaped = np.empty([int(len(train_data)/356), 356, len(train_data.columns)])
    for i in range(int(len(train_data)/356)):
        train_data_reshaped[i,:,:] = train_data[356*i:i*356+356]
    np.random.shuffle(train_data_reshaped)
    print(train_data_reshaped.shape)

    test_data_reshaped = np.empty([int(len(test_data)/356), 356, len(test_data.columns)])
    for i in range(int(len(test_data)/356)):
        test_data_reshaped[i,:,:] = test_data[356*i:i*356+356]
    np.random.shuffle(test_data_reshaped)
    print(test_data_reshaped.shape)

    val_data_reshaped = np.empty([int(len(validation_data)/356), 356, len(validation_data.columns)])
    for i in range(int(len(validation_data)/356)):
        val_data_reshaped[i,:,:] = validation_data[356*i:i*356+356]
    np.random.shuffle(val_data_reshaped)
    print(val_data_reshaped.shape)

    # Save test_data_reshaped, train_data_reshaped, and val_data_reshaped as pickle files
    with open('./Force Prediction/Preprocessed Data/full_train_data_reshaped.pkl', 'wb') as f:
        pickle.dump(full_train_data_reshaped, f)
    with open('./Force Prediction/Preprocessed Data/test_data_reshaped.pkl', 'wb') as f:
        pickle.dump(test_data_reshaped, f)
    with open('./Force Prediction/Preprocessed Data/train_data_reshaped.pkl', 'wb') as f:
        pickle.dump(train_data_reshaped, f)
    with open('./Force Prediction/Preprocessed Data/val_data_reshaped.pkl', 'wb') as f:
        pickle.dump(val_data_reshaped, f)

    # Load test_data_reshaped, train_data_reshaped, and val_data_reshaped from pickle files
    with open('./Force Prediction/Preprocessed Data/full_train_data_reshaped.pkl', 'rb') as f:
        full_train_data_reshaped_loaded = pickle.load(f)
    with open('./Force Prediction/Preprocessed Data/test_data_reshaped.pkl', 'rb') as f:
        test_data_reshaped_loaded = pickle.load(f)
    with open('./Force Prediction/Preprocessed Data/train_data_reshaped.pkl', 'rb') as f:
        train_data_reshaped_loaded = pickle.load(f)
    with open('./Force Prediction/Preprocessed Data/val_data_reshaped.pkl', 'rb') as f:
        val_data_reshaped_loaded = pickle.load(f)

    # Check the shape of the loaded data
    print(full_train_data_reshaped_loaded.shape)
    print(test_data_reshaped_loaded.shape)
    print(train_data_reshaped_loaded.shape)
    print(val_data_reshaped_loaded.shape)
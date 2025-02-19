import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
# Load test_data_reshaped, train_data_reshaped, and val_data_reshaped from pickle files
with open('./Force Prediction/Preprocessed Data/full_train_data_reshaped.pkl', 'rb') as f:
    full_train_data = pickle.load(f)
with open('./Force Prediction/Preprocessed Data/test_data_reshaped.pkl', 'rb') as f:
    test_data = pickle.load(f)
with open('./Force Prediction/Preprocessed Data/train_data_reshaped.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('./Force Prediction/Preprocessed Data/val_data_reshaped.pkl', 'rb') as f:
    val_data = pickle.load(f)

print(test_data.shape)

# Initialize new StandardScaler instances
train_scaler_X = StandardScaler()
train_scaler_y = StandardScaler()
val_scaler_X = StandardScaler()
val_scaler_y = StandardScaler()
test_scaler_X = StandardScaler()
test_scaler_y = StandardScaler()

# Separate X and y
train_X = train_data[:, :, :-6]
train_y = train_data[:, :, -6:]
val_X = val_data[:, :, :-6]
val_y = val_data[:, :, -6:]
test_X = test_data[:, :, :-6]
test_y = test_data[:, :, -6:]

#Fit the scalers to the data and transform it
train_X = train_scaler_X.fit_transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
train_y = train_scaler_y.fit_transform(train_y.reshape(-1, train_y.shape[-1])).reshape(train_y.shape)
val_X = val_scaler_X.fit_transform(val_X.reshape(-1, val_X.shape[-1])).reshape(val_X.shape)
val_y = val_scaler_y.fit_transform(val_y.reshape(-1, val_y.shape[-1])).reshape(val_y.shape)
test_X = test_scaler_X.fit_transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
test_y = test_scaler_y.fit_transform(test_y.reshape(-1, test_y.shape[-1])).reshape(test_y.shape)

print(test_data[:,0,0])
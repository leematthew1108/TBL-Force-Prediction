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

def get_cnn_model(num_filters, kernel_size, num_layers):
    # Define the input layer
    input_layer = tf.keras.layers.Input(shape=(None, 9))  # Adjusted for one less feature

    # Add the Conv1D layers with dropout
    for i in range(num_layers):
        if i == 0:
            x = tf.keras.layers.Conv1D(
                num_filters, 
                kernel_size=kernel_size, 
                padding='causal', 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00001, l2=0.00001)  # Regularization on the weights (kernel)
            )(input_layer)
        else:
            x = tf.keras.layers.Conv1D(
                num_filters, 
                kernel_size=kernel_size, 
                padding='causal', 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00001, l2=0.00001)  # Regularization on the weights (kernel)
            )(x)
        # Batch normalization layer
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
    # Add a time distributed layer
    x = tf.keras.layers.Attention()([x,x])
    x = tf.keras.layers.Dense(256, activation='relu')(x) 
    x = tf.keras.layers.Dropout(0.5)(x)
    output_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(6))(x)

    # Create the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model

## 

 # Set up paths
data_directory = "./Force Prediction/Processed Data for ML"

# Read and process data
data = read_and_process_data(data_directory)

os.chdir('Force Prediction/K-Fold CNN')
data['Intensity'] = data['Intensity'].map({'HIIT': 0.9, 'MICT': 0.5})
data = data.drop(columns=['Cycle_ID', 'Participant_Cycle_ID', 'Normalized_Cycle_Position'])
data = data[data.columns[-3:].tolist() + data.columns[:-3].tolist()]

# Get unique participant IDs
participants = data['Participant'].unique()

model = [None] * 10
model_history = [None] * 10
radius_predictions = [None] * 10
real_y = [None] * 10
mdpe = [None] * 10
md_column = [None] * 10
std_column = [None] * 10
mean_column = [None] * 10

# Load participant splits from text files
train_splits = []
test_splits = []
for fold in range(1, 11):
    with open(rf'C:\Users\griff\VS Code\Field-Based-Wheelchair-Biomechanics-1\Force Prediction\Train and Test Data 10 Fold\Fold_{fold}\train_participants.txt', 'r') as f:
        train_splits.append([int(index) for index in f.read().splitlines()])
    with open(rf'C:\Users\griff\VS Code\Field-Based-Wheelchair-Biomechanics-1\Force Prediction\Train and Test Data 10 Fold\Fold_{fold}\test_participants.txt', 'r') as f:
        test_splits.append([int(index) for index in f.read().splitlines()])

print(train_splits[0])
print(test_splits[0])
# Perform 10-fold cross-validation
for fold in range(10):
    train_participants = train_splits[fold]
    test_participants = test_splits[fold]

    # Further split train set into train and validation sets
    train_participants = train_participants[:int(0.9*len(train_participants))]
    val_participants = train_participants[int(0.9*len(train_participants)):]

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

    #Fit the scalers to the data and transform it
    train_X = train_scaler_X.fit_transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    train_y = train_scaler_y.fit_transform(train_y.reshape(-1, train_y.shape[-1])).reshape(train_y.shape)
    val_X = val_scaler_X.fit_transform(val_X.reshape(-1, val_X.shape[-1])).reshape(val_X.shape)
    val_y = val_scaler_y.fit_transform(val_y.reshape(-1, val_y.shape[-1])).reshape(val_y.shape)
    test_X = test_scaler_X.fit_transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    test_y = test_scaler_y.fit_transform(test_y.reshape(-1, test_y.shape[-1])).reshape(test_y.shape)

    # Confirm these are the correct shapes
    print(train_X.shape)
    print(train_y.shape)
    print(test_X.shape)
    print(test_y.shape)
    print(val_X.shape)
    print(val_y.shape)
    
    model[fold] = get_cnn_model(num_filters=400, kernel_size=64, num_layers=2)
    print(model[fold].summary())

    model[fold].compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])

    checkpoint_path = f"Models/Fold{fold}/cp_{{epoch:02d}}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        monitor='val_mean_absolute_error',
        verbose=1, 
        save_weights_only=False,
        save_freq=600,
        save_best_only = False)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=50,  # Number of epochs with no improvement after which training will be stopped
        min_delta = 0.01, # Minimum change in the monitored quantity to qualify as an improvement
        restore_best_weights=True  # Restore model weights from the epoch with the best value
    )

    model_history[fold] = model[fold].fit(x=train_X,y=train_y, epochs=1000, batch_size=int(16),
                                validation_data=(val_X,val_y),
                                callbacks=[cp_callback, early_stopping_callback], verbose = 1)
    
    # Predict outputs for the entire test set
    radius_predictions[fold] = model[fold].predict(test_X, verbose=2)
    radius_predictions[fold] = test_scaler_y.inverse_transform(radius_predictions[fold].reshape(-1, radius_predictions[fold].shape[-1])).reshape(radius_predictions[fold].shape)
    real_y[fold] = test_scaler_y.inverse_transform(test_y.reshape(-1, test_y.shape[-1])).reshape(test_y.shape)
    
    
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
    plt.savefig(f'Boxplot_MDPE_Fold_{fold}_CNN.png')
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
    plt.savefig(f'Highest_Error_Cycle_Fold{fold}_CNN.png')
    
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

    plt.savefig(f'Mean_Cycle_Fold{fold}_CNN.png')
    
# Save variables to files
# Save
with open('radius_predictions_CNN.pkl', 'wb') as f:
    pickle.dump(radius_predictions, f)

with open('real_y_CNN.pkl', 'wb') as f:
    pickle.dump(real_y, f)

with open('mdpe_CNN.pkl', 'wb') as f:
    pickle.dump(mdpe, f)

with open('md_column_CNN.pkl', 'wb') as f:
    pickle.dump(md_column, f)

with open('std_column_CNN.pkl', 'wb') as f:
    pickle.dump(std_column, f)

with open('mean_column_CNN.pkl', 'wb') as f:
    pickle.dump(mean_column, f)
    
with open('model_history_CNN.pkl', 'wb') as f:
    pickle.dump(model_history.history, f)

for i, model in enumerate(model):
    model.save(f'model_CNN_fold{i}.h5')
## To load models
# loaded_models = []
# for i in range(len(models)):
#     loaded_models.append(tf.keras.models.load_model(f'model_CNN_concat_{i}.h5'))
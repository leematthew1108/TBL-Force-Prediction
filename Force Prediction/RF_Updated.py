## Package Dependencies
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
## Functions
def get_RF_model():
    # Define the RF regressor model
    RF_model = RandomForestRegressor(n_estimators=400, max_depth=10, n_jobs=-1)

    return RF_model

##
# Load test_data_reshaped, train_data_reshaped, and val_data_reshaped from pickle files
with open('./Force Prediction/Preprocessed Data/full_train_data_reshaped.pkl', 'rb') as f:
    full_train_data = pickle.load(f)
with open('./Force Prediction/Preprocessed Data/test_data_reshaped.pkl', 'rb') as f:
    test_data = pickle.load(f)
with open('./Force Prediction/Preprocessed Data/train_data_reshaped.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('./Force Prediction/Preprocessed Data/val_data_reshaped.pkl', 'rb') as f:
    val_data = pickle.load(f)

os.chdir('Force Prediction/RF')

model = None
model_history = None
radius_predictions = None
real_y = None
mdpe = None
md_column = None
std_column = None
mean_column = None

# Initialize new StandardScaler instances
train_scaler_X = StandardScaler()
train_scaler_y = StandardScaler()
val_scaler_X = StandardScaler()
val_scaler_y = StandardScaler()
test_scaler_X = StandardScaler()
test_scaler_y = StandardScaler()

# Separate X and y
train_X = full_train_data[:, :, :-6].reshape(-1, 9)
train_y = full_train_data[:, :, -6:].reshape(-1, 6)
val_X = val_data[:, :, :-6].reshape(-1, 9)
val_y = val_data[:, :, -6:].reshape(-1, 6)
test_X = test_data[:, :, :-6].reshape(-1, 9)
test_y = test_data[:, :, -6:].reshape(-1, 6)

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

model = get_RF_model()

model_history = model.fit(train_X, train_y) #, eval_set=[(val_X, val_y)])

# Save the model weights after the last training iteration
# Save the model after the last training iteration
#model.save_model("RF_model.json")

# Predict outputs for the entire test set
radius_predictions = model.predict(test_X)
radius_predictions = test_scaler_y.inverse_transform(radius_predictions.reshape(-1, radius_predictions.shape[-1])).reshape(radius_predictions.shape).reshape(-1, 356, 6)
real_y = test_scaler_y.inverse_transform(test_y.reshape(-1, test_y.shape[-1])).reshape(test_y.shape).reshape(-1, 356, 6)

mdpe = np.empty([radius_predictions.shape[0], 6])
md_column = np.empty(6)
std_column = np.empty(6)
mean_column = np.empty(6)
columns = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
for i in range(6):
    for j in range(radius_predictions.shape[0]):
        mdpe[j, i] = np.median(np.abs(real_y[j, :, i] - radius_predictions[j, :, i]) / real_y[j, :, i]) * 100
    md_column[i] = np.median(mdpe[:, i])
    std_column[i] = np.std(mdpe[:, i])
    mean_column[i] = np.mean(mdpe[:, i])
    print(f"Median MDPE for {columns[i]}: {md_column[i]}")
    print(f"Mean MDPE for {columns[i]}: {mean_column[i]}")
    print(f"Standard Deviation of MDPE for {columns[i]}: {std_column[i]}")
mean_mdpe = np.mean(mdpe, axis=1)
lowest_mean_slice = np.argmax(mean_mdpe)
print(lowest_mean_slice)

# Create a list of feature names
feature_names = ['Fx [N]', 'Fy [N]', 'Fz [N]', 'Tx [N*m]', 'Ty [N*m]', 'Tz [N*m]']
plt.figure(figsize=(12, 12))
# Create a scatter plot for each feature
for i in range(1, len(mdpe[0]) + 1):
    y = mdpe[:, i-1]
    x = np.random.normal(i, 0.04, size=len(y))  # add jitter to the x-axis
    plt.scatter(x, y, alpha=0.7, color='tab:blue')

# Create a boxplot for each feature without color
bp = plt.boxplot(mdpe, labels=feature_names, patch_artist=True, 
                positions=range(1, len(mdpe[0]) + 1), showfliers=False,
                boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.5),
                medianprops=dict(color='black', linewidth=1.5))

# Set the title and labels
plt.title('Boxplot of MDPE for Each Feature')
plt.xlabel('Feature')
plt.ylabel('MDPE [%]')

# Show the plot
plt.savefig('Boxplot_MDPE_RF.png')
# Plot to visualize ground truth of one cycle with predictions overlaid - quick interpretation of model accuracy
cycle = lowest_mean_slice

plt.figure(figsize=(12, 12))
plt.subplot(611)
plt.plot(np.arange(real_y.shape[1]), real_y[cycle, :, 0], '.')
plt.plot(np.arange(real_y.shape[1]), radius_predictions[0, :, 0], '.')
#plt.plot(np.arange(testset_y.shape[1]), resnet_predictions[0, :, 0], '.')
plt.xlabel('Timestep')
plt.ylabel('Fx [N]')
plt.legend(['Actual', 'Radius Predicted'])
plt.xlim([0, 360])
plt.subplot(612)
plt.plot(np.arange(real_y.shape[1]), real_y[cycle, :, 1], '.')
plt.plot(np.arange(real_y.shape[1]), radius_predictions[0, :, 1], '.')
#plt.plot(np.arange(testset_y.shape[1]), resnet_predictions[0, :, 1], '.')
plt.xlabel('Timestep')
plt.ylabel('Fy [N]')
plt.xlim([0, 360])
plt.subplot(613)
plt.plot(np.arange(real_y.shape[1]), real_y[cycle, :, 2], '.')
plt.plot(np.arange(real_y.shape[1]), radius_predictions[0, :, 2], '.')
#plt.plot(np.arange(testset_y.shape[1]), resnet_predictions[0, :, 2], '.')
plt.xlabel('Timestep')
plt.ylabel('Fz [N]')
plt.xlim([0, 360])
plt.subplot(614)
plt.plot(np.arange(real_y.shape[1]), real_y[cycle, :, 3], '.')
plt.plot(np.arange(real_y.shape[1]), radius_predictions[0, :, 3], '.')
#plt.plot(np.arange(testset_y.shape[1]), resnet_predictions[0, :, 3], '.')``
plt.xlabel('Timestep')
plt.ylabel('Tx [Nm]')
plt.xlim([0, 360])
plt.subplot(615)
plt.plot(np.arange(real_y.shape[1]), real_y[cycle, :, 4], '.')
plt.plot(np.arange(real_y.shape[1]), radius_predictions[0, :, 4], '.')
#plt.plot(np.arange(testset_y.shape[1]), resnet_predictions[0, :, 4], '.')
plt.xlabel('Timestep')
plt.ylabel('Ty [Nm]')
plt.xlim([0, 360])
plt.subplot(616)
plt.plot(np.arange(real_y.shape[1]), real_y[cycle, :, 5], '.')
plt.plot(np.arange(real_y.shape[1]), radius_predictions[0, :, 5], '.')
#plt.plot(np.arange(testset_y.shape[1]), resnet_predictions[0, :, 5], '.')
plt.xlabel('Timestep')
plt.ylabel('Tz [Nm]')
plt.xlim([0, 360])
plt.savefig('Highest_Error_Cycle_RF.png')

mean_real_y = np.mean(real_y, axis=0)
mean_radius_predictions = np.mean(radius_predictions, axis=0)

plt.figure(figsize=(20, 12))

plt.subplot(611)
plt.plot(np.arange(real_y.shape[1]), mean_real_y[:, 0], '.', markersize=2)
plt.plot(np.arange(real_y.shape[1]), mean_radius_predictions[:, 0], '.', markersize=2)
#plt.fill_between(np.arange(test_y.shape[1]), mean_real_y[:,0] - std_y[0], mean_real_y[:,0] + std_y[0], alpha=0.3)
#plt.fill_between(np.arange(test_y.shape[1]), mean_radius_predictions[:,0] - std_y[0], mean_radius_predictions[:,0] + std_y[0], alpha=0.3)
plt.xlabel('Timestep')
plt.ylabel('Fx [N]')
plt.legend(['Ground Truth (Mean)', 'Radius Predicted (Mean)'])
plt.xlim([0, 360])

plt.subplot(612)
plt.plot(np.arange(real_y.shape[1]), mean_real_y[:, 1], '.', markersize=2)
plt.plot(np.arange(real_y.shape[1]), mean_radius_predictions[:, 1], '.', markersize=2)
#plt.fill_between(np.arange(test_y.shape[1]), mean_real_y[:,1] - std_y[1], mean_real_y[:,1] + std_y[1], alpha=0.3)
#plt.fill_between(np.arange(test_y.shape[1]), mean_radius_predictions[:,1] - std_y[1], mean_radius_predictions[:,1] + std_y[1], alpha=0.3)
plt.xlabel('Timestep')
plt.ylabel('Fy [N]')
plt.xlim([0, 360])

plt.subplot(613)
plt.plot(np.arange(real_y.shape[1]), mean_real_y[:, 2], '.', markersize=2)
plt.plot(np.arange(real_y.shape[1]), mean_radius_predictions[:, 2], '.', markersize=2)
#plt.fill_between(np.arange(test_y.shape[1]), mean_real_y[:,2] - std_y[2], mean_real_y[:,2] + std_y[2], alpha=0.3)
#plt.fill_between(np.arange(test_y.shape[1]), mean_radius_predictions[:,2] - std_y[2], mean_radius_predictions[:,2] + std_y[2], alpha=0.3)
plt.xlabel('Timestep')
plt.ylabel('Fz [N]')
plt.xlim([0, 360])

plt.subplot(614)
plt.plot(np.arange(real_y.shape[1]), mean_real_y[:, 3], '.', markersize=2)
plt.plot(np.arange(real_y.shape[1]), mean_radius_predictions[:, 3], '.', markersize=2)
#plt.fill_between(np.arange(test_y.shape[1]), mean_real_y[:,3] - std_y[3], mean_real_y[:,3] + std_y[3], alpha=0.3)
#plt.fill_between(np.arange(test_y.shape[1]), mean_radius_predictions[:,3] - std_y[3], mean_radius_predictions[:,3] + std_y[3], alpha=0.3)
plt.xlabel('Timestep')
plt.ylabel('Tx [Nm]')
plt.xlim([0, 360])

plt.subplot(615)
plt.plot(np.arange(real_y.shape[1]), mean_real_y[:, 4], '.', markersize=2)
plt.plot(np.arange(real_y.shape[1]), mean_radius_predictions[:, 4], '.', markersize=2)
#plt.fill_between(np.arange(test_y.shape[1]), mean_real_y[:,4] - std_y[4], mean_real_y[:,4] + std_y[4], alpha=0.3)
#plt.fill_between(np.arange(test_y.shape[1]), mean_radius_predictions[:,4] - std_y[4], mean_radius_predictions[:,4] + std_y[4], alpha=0.3)
plt.xlabel('Timestep')
plt.ylabel('Ty [Nm]')
plt.xlim([0, 360])

plt.subplot(616)
plt.plot(np.arange(real_y.shape[1]), mean_real_y[:, 5], '.', markersize=2)
plt.plot(np.arange(real_y.shape[1]), mean_radius_predictions[:, 5], '.', markersize=2)
#plt.fill_between(np.arange(test_y.shape[1]), mean_real_y[:,5] - std_y[5], mean_real_y[:,5] + std_y[5], alpha=0.3)
#plt.fill_between(np.arange(test_y.shape[1]), mean_radius_predictions[:,5] - std_y[5], mean_radius_predictions[:,5] + std_y[5], alpha=0.3)
plt.xlabel('Timestep')
plt.ylabel('Tz [Nm]')
plt.xlim([0, 360])

plt.savefig('Mean_Cycle_RF.png')

# Save variables to files
# Save
with open('radius_predictions_RF.pkl', 'wb') as f:
    pickle.dump(radius_predictions, f)

with open('real_y_RF.pkl', 'wb') as f:
    pickle.dump(real_y, f)

with open('mdpe_RF.pkl', 'wb') as f:
    pickle.dump(mdpe, f)

with open('md_column_RF.pkl', 'wb') as f:
    pickle.dump(md_column, f)

with open('std_column_RF.pkl', 'wb') as f:
    pickle.dump(std_column, f)

with open('mean_column_RF.pkl', 'wb') as f:
    pickle.dump(mean_column, f)

## To load models
# loaded_models = []
# for i in range(len(models)):
#     loaded_models.append(tf.keras.models.load_model(f'model_RF_concat_{i}.h5'))
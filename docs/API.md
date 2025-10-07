# API Documentation

## Data Processing Module (`src.data_processing`)

### Functions

#### `load_and_prepare_data(file_path)`
Load and prepare trajectory data from CSV file.

**Parameters:**
- `file_path` (str): Path to the CSV file containing trajectory data

**Returns:**
- `pd.DataFrame`: Loaded and prepared dataset with datetime conversion

#### `extract_features(trajectory)`
Extract statistical and geographical features from trajectory data.

**Parameters:**
- `trajectory` (np.ndarray): Trajectory data with columns [longitude, latitude, timestamp]

**Returns:**
- `list`: List of features [mean_lon, mean_lat, std_lon, std_lat, avg_speed]

#### `haversine_distance(longitudes, latitudes)`
Calculate total distance using Haversine formula.

**Parameters:**
- `longitudes` (array-like): Array of longitude values
- `latitudes` (array-like): Array of latitude values

**Returns:**
- `float`: Total distance in kilometers

## Models Module (`src.models`)

### Functions

#### `create_rnn_model(input_shape)`
Create and compile RNN model for driver classification.

**Parameters:**
- `input_shape` (tuple): Input shape for the model

**Returns:**
- `keras.models.Sequential`: Compiled RNN model

#### `create_lstm_model(input_shape)`
Create and compile LSTM model for driver classification.

**Parameters:**
- `input_shape` (tuple): Input shape for the model

**Returns:**
- `keras.models.Sequential`: Compiled LSTM model

#### `save_model(model, filepath)`
Save trained model to disk.

**Parameters:**
- `model`: Trained Keras model
- `filepath` (str): Path where to save the model

#### `load_model(filepath)`
Load trained model from disk.

**Parameters:**
- `filepath` (str): Path to the saved model

**Returns:**
- `keras.models.Sequential`: Loaded model

## Prediction Module (`src.prediction`)

### Functions

#### `run_prediction(input_data, trained_model)`
Run predictions using trained model.

**Parameters:**
- `input_data` (array-like): Input data for prediction
- `trained_model`: Trained model for prediction

**Returns:**
- `float`: Prediction result

#### `predict_from_trajectory(trajectory, model_path)`
Make prediction from trajectory data using saved model.

**Parameters:**
- `trajectory` (np.ndarray): Trajectory data
- `model_path` (str): Path to the saved model

**Returns:**
- `float`: Prediction accuracy/result

## Main Module (`src.main`)

### Functions

#### `train_models(data_path, epochs=50, batch_size=16)`
Train both RNN and LSTM models on trajectory data.

**Parameters:**
- `data_path` (str): Path to the trajectory data CSV file
- `epochs` (int): Number of training epochs (default: 50)
- `batch_size` (int): Batch size for training (default: 16)

**Returns:**
- `tuple`: (rnn_model, lstm_model)

#### `evaluate_models(test_data_path)`
Evaluate trained models on test data.

**Parameters:**
- `test_data_path` (str): Path to the test data pickle file

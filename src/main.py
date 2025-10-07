"""
Main execution script for trajectory analysis.

This script demonstrates how to train and evaluate RNN and LSTM models
for driver classification using trajectory data.
"""

import pickle
from .data_processing import load_and_prepare_data, preprocess_data
from .models import create_rnn_model, create_lstm_model, save_model


def train_models(data_path, epochs=50, batch_size=16):
    """
    Train both RNN and LSTM models on the trajectory data.
    
    Args:
        data_path (str): Path to the trajectory data CSV file
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (rnn_model, lstm_model)
    """
    print("Loading and preprocessing data...")
    # Load and preprocess data
    data = load_and_prepare_data(data_path)
    X, y = preprocess_data(data)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Build, train, and save RNN model
    print("Training RNN model...")
    rnn_model = create_rnn_model(input_shape=(1, 5))
    rnn_model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    save_model(rnn_model, 'models/rnn_model.pkl')
    print("RNN model saved to models/rnn_model.pkl")
    
    # Build, train, and save LSTM model
    print("Training LSTM model...")
    lstm_model = create_lstm_model(input_shape=(1, 5))
    lstm_model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    save_model(lstm_model, 'models/lstm_model.pkl')
    print("LSTM model saved to models/lstm_model.pkl")
    
    return rnn_model, lstm_model


def evaluate_models(test_data_path):
    """
    Evaluate trained models on test data.
    
    Args:
        test_data_path (str): Path to the test data pickle file
    """
    from .prediction import predict_from_trajectory
    
    print("Loading test data...")
    test_data = pickle.load(open(test_data_path, 'rb'))
    
    print("Evaluating RNN model...")
    for i, test_trajectory in enumerate(test_data):
        result = predict_from_trajectory(test_trajectory, 'models/rnn_model.pkl')
        print(f"RNN Test {i+1} accuracy: {result}")
    
    print("Evaluating LSTM model...")
    for i, test_trajectory in enumerate(test_data):
        result = predict_from_trajectory(test_trajectory, 'models/lstm_model.pkl')
        print(f"LSTM Test {i+1} accuracy: {result}")


if __name__ == "__main__":
    # Example usage
    import os
    
    # Check if data file exists
    data_path = 'data/merged_data.csv'
    test_path = 'data/test.pkl'
    
    if os.path.exists(data_path):
        train_models(data_path)
    else:
        print(f"Data file not found: {data_path}")
        print("Please ensure the trajectory data is available.")
    
    if os.path.exists(test_path):
        evaluate_models(test_path)
    else:
        print(f"Test data file not found: {test_path}")
        print("Please ensure the test data is available.")

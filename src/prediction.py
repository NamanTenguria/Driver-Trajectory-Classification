"""
Prediction module for trajectory analysis.

This module contains functions for making predictions using trained models.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from .data_processing import process_data


def run_prediction(input_data, trained_model):
    """
    Run predictions using the trained model.
    
    Args:
        input_data (array-like): Input data for prediction
        trained_model: Trained model for prediction
        
    Returns:
        float: Prediction result
    """
    # Ensure input data is a NumPy array and reshape
    input_data = np.array(input_data).reshape(1, -1)

    # Initialize a new StandardScaler, as the model's training process directly scales the data
    scaler = StandardScaler()

    # Normalize the input data (note: StandardScaler is fit on the same data in the model's training process)
    normalized_data = scaler.fit_transform(input_data)

    # Make predictions using the trained model
    predictions = trained_model.predict(np.array([normalized_data])).mean()
    return predictions


def predict_from_trajectory(trajectory, model_path):
    """
    Make prediction from trajectory data using a saved model.
    
    Args:
        trajectory (np.ndarray): Trajectory data
        model_path (str): Path to the saved model
        
    Returns:
        float: Prediction accuracy/result
    """
    from .models import load_model
    
    # Process the trajectory data to extract features
    features = process_data(trajectory)
    
    # Load the trained model
    model = load_model(model_path)
    
    # Make prediction
    result = run_prediction(features, model)
    
    return result

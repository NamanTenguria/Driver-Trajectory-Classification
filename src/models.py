"""
Model definitions for trajectory analysis.

This module contains the RNN and LSTM model architectures for driver classification.
"""

import pickle
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, SimpleRNN
from keras.optimizers import Adam


def create_rnn_model(input_shape):
    """
    Define and compile the RNN model.
    
    Args:
        input_shape (tuple): Input shape for the model
        
    Returns:
        keras.models.Sequential: Compiled RNN model
    """
    model = Sequential()
    model.add(SimpleRNN(64, return_sequences=True, input_shape=input_shape))
    model.add(SimpleRNN(32, return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def create_lstm_model(input_shape):
    """
    Define and compile the LSTM model.
    
    Args:
        input_shape (tuple): Input shape for the model
        
    Returns:
        keras.models.Sequential: Compiled LSTM model
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='sigmoid'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def save_model(model, filepath):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained Keras model
        filepath (str): Path where to save the model
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath):
    """
    Load a trained model from disk.
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        keras.models.Sequential: Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

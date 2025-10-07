"""
Data processing module for trajectory analysis.

This module contains functions for loading, preprocessing, and feature extraction
from vehicle trajectory data.
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(file_path):
    """
    Load and prepare the dataset from CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing trajectory data
        
    Returns:
        pd.DataFrame: Loaded and prepared dataset
    """
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    return df


def trip_duration(trip_df):
    """
    Calculate the duration of a trip.
    
    Args:
        trip_df (pd.DataFrame): DataFrame containing trip data
        
    Returns:
        float: Duration of the trip in seconds
    """
    start_time = trip_df['time'].min()
    end_time = trip_df['time'].max()
    return (end_time - start_time).total_seconds()


def haversine_distance(longitudes, latitudes):
    """
    Calculate the total distance using the Haversine formula.
    
    Args:
        longitudes (array-like): Array of longitude values
        latitudes (array-like): Array of latitude values
        
    Returns:
        float: Total distance in kilometers
    """
    earth_radius = 6371.0  # Earth's radius in kilometers
    total_distance = 0.0

    for i in range(1, len(longitudes)):
        lon1, lat1 = radians(longitudes[i - 1]), radians(latitudes[i - 1])
        lon2, lat2 = radians(longitudes[i]), radians(latitudes[i])

        dlon, dlat = lon2 - lon1, lat2 - lat1

        # Haversine formula
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        total_distance += earth_radius * c

    return total_distance


def extract_features(trajectory):
    """
    Generate features from a trajectory dataset.
    
    Args:
        trajectory (np.ndarray): Trajectory data with columns [longitude, latitude, timestamp]
        
    Returns:
        list: List of extracted features [mean_lon, mean_lat, std_lon, std_lat, avg_speed]
    """
    longitudes = trajectory[:, 0]
    latitudes = trajectory[:, 1]
    timestamps = pd.to_datetime(trajectory[:, 2]).values.astype(np.int64) // 10**9

    mean_lon = np.mean(longitudes)
    mean_lat = np.mean(latitudes)
    std_lon = np.std(longitudes)
    std_lat = np.std(latitudes)

    # Calculate total distance and average speed
    distance_travelled = haversine_distance(longitudes, latitudes)
    avg_speed = distance_travelled / np.sum(timestamps)

    return [mean_lon, mean_lat, std_lon, std_lat, avg_speed]


def group_by_driver(df_group):
    """
    Group the dataset by 'plate' and sort the data.
    
    Args:
        df_group (pd.DataFrame): Grouped DataFrame by driver plate
        
    Returns:
        list: [sorted_trajectory, driver_id]
    """
    sorted_trajectory = np.array(sorted(df_group.values[:, 1:], key=lambda x: x[2]))
    driver_id = df_group.iloc[0][0]
    return [sorted_trajectory, driver_id]


def preprocess_data(dataset):
    """
    Normalize features and prepare the labels for training.
    
    Args:
        dataset (pd.DataFrame): Input dataset
        
    Returns:
        tuple: (normalized_data, labels_array)
    """
    scaler = StandardScaler()
    feature_data = []
    labels = []

    for traj, label in dataset.groupby('plate').apply(group_by_driver, include_groups=False):
        features = extract_features(traj)
        feature_data.append(features)
        labels.append(label)

    normalized_data = scaler.fit_transform(feature_data)
    normalized_data = normalized_data.reshape(len(normalized_data), 1, len(normalized_data[0]))
    labels_array = np.array(labels).reshape(-1, 1)

    return normalized_data, labels_array


def process_data(trajectory):
    """
    Function to preprocess and extract features from trajectory data.
    
    Args:
        trajectory (np.ndarray): Trajectory data
        
    Returns:
        list: List of extracted features
    """
    # Extract longitude, latitude, and time differences from trajectory
    longitudes = trajectory[:, 0]
    latitudes = trajectory[:, 1]
    time_intervals = pd.to_datetime(trajectory[:, 2]).values.astype(np.int64) // 10**9

    # Calculate mean and standard deviation of longitude and latitude
    avg_longitude = np.mean(longitudes)
    avg_latitude = np.mean(latitudes)
    stddev_longitude = np.std(longitudes)
    stddev_latitude = np.std(latitudes)

    # Compute total distance and mean speed
    total_distance = compute_total_distance(longitudes, latitudes)
    avg_speed = total_distance / np.sum(time_intervals)

    # Combine extracted features into a list
    features = [avg_longitude, avg_latitude, stddev_longitude, stddev_latitude, avg_speed]
    return features


def compute_total_distance(longitudes, latitudes):
    """
    Function to calculate total distance using Haversine formula.
    
    Args:
        longitudes (array-like): Array of longitude values
        latitudes (array-like): Array of latitude values
        
    Returns:
        float: Total distance in kilometers
    """
    radius_earth_km = 6371.0  # Earth's radius in kilometers
    total_distance = 0.0

    for i in range(1, len(longitudes)):
        lat1_rad = radians(latitudes[i - 1])
        lon1_rad = radians(longitudes[i - 1])
        lat2_rad = radians(latitudes[i])
        lon2_rad = radians(longitudes[i])

        # Calculate differences in coordinates
        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad

        # Apply the Haversine formula
        a = sin(delta_lat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        # Add the calculated distance to the total distance
        total_distance += radius_earth_km * c

    return total_distance

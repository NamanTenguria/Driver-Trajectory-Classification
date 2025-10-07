"""
Tests for data processing module.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_processing import (
    haversine_distance,
    extract_features,
    process_data,
    compute_total_distance
)


class TestHaversineDistance:
    """Test cases for haversine distance calculation."""
    
    def test_haversine_distance_same_point(self):
        """Test distance calculation for same point."""
        longitudes = [0.0, 0.0]
        latitudes = [0.0, 0.0]
        distance = haversine_distance(longitudes, latitudes)
        assert distance == 0.0
    
    def test_haversine_distance_known_distance(self):
        """Test distance calculation for known coordinates."""
        # Approximate distance between New York and Los Angeles
        longitudes = [-74.0, -118.0]
        latitudes = [40.0, 34.0]
        distance = haversine_distance(longitudes, latitudes)
        # Should be approximately 3944 km
        assert 3900 <= distance <= 4000


class TestFeatureExtraction:
    """Test cases for feature extraction."""
    
    def test_extract_features(self):
        """Test feature extraction from trajectory data."""
        # Create sample trajectory data
        trajectory = np.array([
            [0.0, 0.0, '2023-01-01 00:00:00'],
            [1.0, 1.0, '2023-01-01 00:01:00'],
            [2.0, 2.0, '2023-01-01 00:02:00']
        ])
        
        features = extract_features(trajectory)
        
        assert len(features) == 5
        assert isinstance(features[0], float)  # mean_lon
        assert isinstance(features[1], float)  # mean_lat
        assert isinstance(features[2], float)  # std_lon
        assert isinstance(features[3], float)  # std_lat
        assert isinstance(features[4], float)  # avg_speed
    
    def test_process_data(self):
        """Test process_data function."""
        trajectory = np.array([
            [0.0, 0.0, '2023-01-01 00:00:00'],
            [1.0, 1.0, '2023-01-01 00:01:00']
        ])
        
        features = process_data(trajectory)
        
        assert len(features) == 5
        assert all(isinstance(f, float) for f in features)


class TestDistanceCalculation:
    """Test cases for distance calculation functions."""
    
    def test_compute_total_distance(self):
        """Test compute_total_distance function."""
        longitudes = [0.0, 1.0, 2.0]
        latitudes = [0.0, 1.0, 2.0]
        
        distance = compute_total_distance(longitudes, latitudes)
        
        assert isinstance(distance, float)
        assert distance > 0


if __name__ == "__main__":
    pytest.main([__file__])

#  Trajectory Analysis with Deep Learning

This project implements driver classification using vehicle trajectory data through Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) models.

## Project Overview

The goal of this project is to classify drivers based on their trajectory patterns using deep learning techniques. The system extracts features from GPS trajectory data and trains neural network models to identify different driving behaviors.

## Features

- **Data Processing**: Comprehensive trajectory data preprocessing with Haversine distance calculation
- **Feature Extraction**: Statistical and geographical feature extraction from GPS coordinates
- **Deep Learning Models**: Implementation of RNN and LSTM architectures for driver classification
- **Model Evaluation**: Prediction and evaluation pipeline for trained models

## Project Structure

```
ADM_A3/
├── src/                    # Source code modules
│   ├── __init__.py        # Package initialization
│   ├── data_processing.py # Data loading and preprocessing functions
│   ├── models.py          # Neural network model definitions
│   ├── prediction.py      # Prediction and evaluation functions
│   └── main.py           # Main execution script
├── data/                  # Data directory (not tracked in git)
├── models/                # Saved model files (not tracked in git)
├── docs/                  # Documentation
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore patterns
└── README.md             # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ADM_A3
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

Place your trajectory data in CSV format in the `data/` directory. The expected format should include columns for:
- `plate`: Driver identifier
- `longitude`: GPS longitude coordinates
- `latitude`: GPS latitude coordinates
- `time`: Timestamp of the GPS reading

### Training Models

To train both RNN and LSTM models:

```python
from src.main import train_models

# Train models with default parameters
train_models('data/merged_data.csv', epochs=50, batch_size=16)
```

Or run the main script directly:

```bash
python -m src.main
```

### Making Predictions

To make predictions using trained models:

```python
from src.prediction import predict_from_trajectory

# Load trajectory data and make prediction
result = predict_from_trajectory(trajectory_data, 'models/lstm_model.pkl')
print(f"Prediction accuracy: {result}")
```

### Evaluation

To evaluate models on test data:

```python
from src.main import evaluate_models

# Evaluate models on test data
evaluate_models('data/test.pkl')
```

## Model Architecture

### RNN Model
- SimpleRNN layers: 64 → 32 units
- Dense layer: 32 units with ReLU activation
- Dropout: 0.2 for regularization
- Output layer: 5 units with sigmoid activation

### LSTM Model
- LSTM layers: 64 → 32 units
- Dense layer: 32 units with ReLU activation
- Dropout: 0.2 for regularization
- Output layer: 5 units with sigmoid activation

## Feature Engineering

The system extracts the following features from trajectory data:
1. **Mean longitude and latitude**: Central tendency of the trajectory
2. **Standard deviation of coordinates**: Variability in movement
3. **Average speed**: Calculated using Haversine distance formula

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.1.0
- keras >= 2.10.0
- tensorflow >= 2.10.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- jupyter >= 1.0.0

## Results

The models achieve the following performance on the test dataset:
- **RNN Model**: ~50.34% accuracy
- **LSTM Model**: ~49.98% accuracy

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is part of an academic assignment and is intended for educational purposes.

## Author

**Naman Tenguria**
- Academic Project: ADM Assignment 3

## Acknowledgments

- Course: Advanced Data Mining (ADM)
- University: [Your University Name]
- Semester: 3

---

*Note: This project is part of an academic assignment focused on deep learning applications in trajectory analysis.*

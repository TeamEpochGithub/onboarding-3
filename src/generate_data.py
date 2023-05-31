import numpy as np
import pandas as pd


def get_dataset(num_samples, num_features):
    # Generate random feature data
    features = np.random.randn(num_samples, num_features)

    # Generate random target labels
    weights = np.random.randn(num_features)  # Coefficients for each feature
    bias = 3.0  # Bias term
    noise = 0.2 * np.random.randn(num_samples)  # Random noise
    targets = np.dot(features, weights) + bias + noise

    # Combine features and targets into a DataFrame
    data = pd.DataFrame(np.column_stack((features, targets)), columns=[f'feature_{i}' for i in range(num_features)] + ['target'])
    return data


def generate_inference(inf_num_samples, num_features):
    inf_features = np.random.randn(inf_num_samples, num_features)
    return inf_features

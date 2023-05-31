import torch
import torch.nn as nn
import torch.optim as optim

import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.generate_data import get_dataset
from src.model_factory import ModelFactory


# --------------------------------------------
# Exercise 1: speed up the code by making use of gpu
# --------------------------------------------

# Define the number of samples and features (do not change for the exercises)
num_samples = int(1e6)
num_features = 100


def train():
    data: pd.DataFrame = get_dataset(num_samples, num_features)

    # Split the data into features and targets
    features = data.iloc[:, :-1].values
    targets = data.iloc[:, -1].values

    # Convert the data to PyTorch tensors
    features = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    # Define a simple (but relatively large) model
    model = ModelFactory.create_simple_model(num_features)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 20
    for epoch in tqdm(range(num_epochs)):
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs.view(-1), targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

    return model


if __name__ == '__main__':
    # if done correctly, you should be looking at less than 3 seconds for 20 epochs,
    # and less than 20 for 100 epochs
    train()

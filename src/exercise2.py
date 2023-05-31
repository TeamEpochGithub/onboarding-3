import torch

import time
import numpy as np
from tqdm import tqdm

from src.exercise1 import train, num_features
from src.generate_data import generate_inference

# --------------------------------------------
# Exercise 2: optimize the model inference
# --------------------------------------------

# get the trained model from exercise 1
model = train()

# some data we dont know the answers for
inf_num_samples = int(1e6)  # ("inf" for inference, not infinity)

inf_features = generate_inference(inf_num_samples, num_features)
inf_features = torch.tensor(inf_features, dtype=torch.float32)

predictions = []
for i in tqdm(range(len(inf_features))):
    prediction = model(inf_features[i])
    predictions.append(prediction)

print(f'average prediction: {sum(predictions) / len(inf_features)}')
# the correct code should complete in less than half a second

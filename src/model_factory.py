import torch.nn as nn

class ModelFactory:
    @staticmethod
    def create_simple_model(input_dim):
        model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        return model

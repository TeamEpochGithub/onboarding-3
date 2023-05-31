import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

import time


DEVICE = 'cuda'

device = torch.device(DEVICE)

model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3),
    nn.Flatten(),
    nn.Linear(32*26*26, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor())

# Define dataloaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

start_time = time.time()

for epoch in range(1):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # Move data to device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print epoch loss
    print('Epoch %d, Loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

end_time = time.time()
training_time = end_time - start_time

print(f'Training time on {device}: {training_time} seconds')
# about 50 seconds on CPU
# about 8 seconds on GPU

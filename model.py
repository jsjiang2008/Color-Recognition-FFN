import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Generate training data
def generate_synthetic_data(num_samples = 100000):
    # Generate 3 random numbers for rgb values
    X = np.random.rand(num_samples, 3)
    y = np.array([(1 if r > g + b else 0) for r, g, b in X])
    return X, y

# Generate data and split into testing and training sets
# X is the rgb colors
# y is correct lable for each color
# test size means 20% reserved for testing, 80% for training
# random state fixes randomness so that the same set would be split in the same way every time
X, y = generate_synthetic_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define neural network
# Inherits from nn.Module, base class for all neural networks in PyTorch
class ColorClassifier(nn.Module):
    # __init__ runs upon object creation
    def __init__(self):
        super(ColorClassifier, self).__init__()
        # fc is short for "fully-connected", meaning weighted edges between neurons in two adjacent layers
        self.fc1 = nn.Linear(3, 10)  # 3 input features (R, G, B) and 10 hidden neurons
        self.fc2 = nn.Linear(10, 1)  # 1 output (binary classification)

    # forward defines how data is passed through the network during forward pass
    def forward(self, x):
        # ReLU activation function for hidden layer neurons
        x = torch.relu(self.fc1(x))
        # sigmoid activation function for the output, which is 0 or 1
        # Hopefully 0 is cold 1 is warm
        x = torch.sigmoid(self.fc2(x))
        return x

# Instantiate neural network
model = ColorClassifier()

# Criterion is loss function (binary cross-entropy)
criterion = nn.BCELoss()

# SGD: stochastic gradient descent, breaks training set to mini-batches
# Makes it more efficient to update the weights iteratively 
optimizer = optim.SGD(model.parameters(), lr=0.01)

batch_size = 64  # Example batch size, number of training examples in a mini-batch
# TensorDataset and DataLoader breaks the training set to mini batches
# Shuffles data around for better distribution
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
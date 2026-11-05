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
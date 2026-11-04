import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Generate training data
def generateSyntheticData(numSamples = 100000):
    # Generate 3 random numbers for rgb values
    X = np.random.rand(numSamples, 3)
    y = np.array([(1 if r > g + b else 0) for r, g, b in X])
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# Check for NVIDIA GPU
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")

# Choose GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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
model = ColorClassifier().to(device)

# Criterion is loss function (binary cross-entropy)
criterion = nn.BCELoss()

# SGD: stochastic gradient descent, breaks training set to mini-batches
# Makes it more efficient to update the weights iteratively 
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Larger batches reduce GPU launch overhead for this tiny network.
# Batch size affects convergence; lower it if accuracy suffers.
batch_size = 1024 if device.type == "cuda" else 64

# The training tensors total about 1.3 MB, so keep them on the device.
X_train = X_train.to(device)
y_train = y_train.to(device)

# Train the model

# Define number of loops
num_epochs = 1000

for epoch in range(num_epochs):
    # Set model to training mode
    model.train()

    # Shuffle on the training device once per epoch, then take batch views.
    order = torch.randperm(X_train.size(0), device=device)
    shuffled_X = X_train[order]
    shuffled_y = y_train[order]
    for batch_X, batch_y in zip(
        shuffled_X.split(batch_size), shuffled_y.split(batch_size)
    ):

        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        # Makes predictions into outputs and calculates the loss
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        # PyTorch uses chain rule to calculate how each weight contributed to that loss
        # This is called backpropagation
        loss.backward()
        # Optimizer reads those gadients and updates weights and biases
        optimizer.step()

    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# Test model
# Set to evaluation mode
model.eval()
# Disable gradient calculation
with torch.no_grad():
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    predictions = model(X_test)

    # Convert predictions to binary (round to 0 or 1)
    predictions = (predictions > 0.5).float()
    accuracy = (predictions == y_test).sum() / y_test.size(0)
    print(f"Accuracy: {accuracy:.4f}")

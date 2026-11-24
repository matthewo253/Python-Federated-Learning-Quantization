import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)   # GPU parity (no-op on CPU)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple MLP model
class Model(nn.Module):
    def __init__(self):

        super(Model, self).__init__()
        # torch.nn.Linear(size of each input sample, size of each output sample)
        self.fc1 = nn.Linear(784,390) # Input size should be 784 (28*28*1), which is the size of the sample in MNIST dataset. Output size is the size of first hidden layer(your design parameter).
        self.fc2 = nn.Linear(390, 200) # Input size should be the size of first hidden layer. Output size is the size of second hidden layer(your design parameter).
        self.fc3 = nn.Linear(200, 10) # Input size should be the size of second hidden layer. Output size should be 10 (the number of classes).

    def forward(self, x):

        x = x.view(-1, 1*28*28) # Change the shape([batch_size, channels, row, col]) to [batch_size, channels*row*col] for fully-connected layer
        x = F.relu(self.fc1(x)) # Use relu function after fully-connected layer
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # Note that softmax function is not applied here.

        return x

# Load MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)


# Complete the function to create data splits for clients.
# For Non-IID:
def create_data_splits(dataset, num_clients=5, iid=True):
    targets = np.array(dataset.targets)
    indices = np.arange(len(dataset))
    if iid:

       # Shuffles the indices randomly
       np.random.shuffle(indices)

       # Splits the shuffled indices evenly
       client_indices = np.array_split(indices, num_clients)
    else:

       # Sort the indices by their labels so identical digits are grouped together
       sorted_indices = indices[np.argsort(targets)]

       # Split the sorted indices evenly
       client_indices = np.array_split(sorted_indices, num_clients)
    return client_indices

# Local training process at each client
def local_train(model, train_loader, epochs=1):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
    return model.state_dict()

# Implement federated averaging of model weights from clients
def average_weights(weight_list):
    averaged_weights = {}

    # Loop through each layer of the model and stacks the matching layers from all clients and calculate their mathematical average
    for key in weight_list[0].keys():
        averaged_weights[key] = torch.stack(
            [weights[key].float() for weights in weight_list], dim=0
        ).mean(dim=0)

    return averaged_weights

# Evaluate accuracy on test set
def evaluate(model, test_loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            # Get raw model outputs
            outputs = model(x)

            # Determine the predicted digit
            _, predicted = torch.max(outputs, 1)


            # Track total samples and count how many predictions match the ground truth
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = correct / total


    return accuracy


# Overall federated learning process
def federated_learning(iid_flag, num_rounds, num_clients, epochs_per_client):
    client_indices = create_data_splits(train_dataset, num_clients=num_clients, iid=iid_flag)
    global_model = Model().to(device)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    accuracy_list = []
    for r in range(num_rounds): # For each global round
        local_weights = []
        for i in range(num_clients): # For each client
            client_model = Model().to(device)
            client_model.load_state_dict(global_model.state_dict())
            client_loader = DataLoader(Subset(train_dataset, client_indices[i]), batch_size=32, shuffle=True)
            w = local_train(client_model, client_loader, epochs=epochs_per_client)
            local_weights.append(w)
        averaged_weights = average_weights(local_weights)
        global_model.load_state_dict(averaged_weights)
        acc = evaluate(global_model, test_loader)
        accuracy_list.append(acc)
        print(f"IID={iid_flag} Round {r+1}/{num_rounds} - Accuracy: {acc*100:.2f}%")
    return accuracy_list


# Run training and evalutation
num_rounds = 10
num_clients=5
epochs_per_client=1

print("Training IID federated learning...")
iid_acc = federated_learning(iid_flag=True, num_rounds=num_rounds, num_clients=num_clients, epochs_per_client=epochs_per_client)

print("\nTraining Non-IID federated learning...")
non_iid_acc = federated_learning(iid_flag=False, num_rounds=num_rounds, num_clients=num_clients, epochs_per_client=epochs_per_client)

# Plotting
plt.figure(figsize=(8,6))
plt.plot(range(1, num_rounds+1), iid_acc, marker='o', label='IID')
plt.plot(range(1, num_rounds+1), non_iid_acc, marker='s', label='Non-IID')
plt.title('Federated Learning Accuracy vs Rounds')
plt.xlabel('Round')
plt.ylabel('Test Accuracy')
plt.legend()
plt.grid(True)
plt.show()
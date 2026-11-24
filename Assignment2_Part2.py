import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import copy

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)   # GPU parity (no-op on CPU)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data: MNIST
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=64,  shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=256, shuffle=False)

# Define a simple MLP model same as Task 1
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(784, 390)
        self.fc2 = nn.Linear(390, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the full-precision Base model
def train(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: Train Loss = {total_loss / len(loader):.4f}")

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, epoch)
fp_acc = evaluate(model, test_loader)
print(f"[Base] Full-precision (32-bit) Test Accuracy: {fp_acc:.2f}%")


def linear_quantize_tensor(W, num_bits):
    # 2's complement integer range for num_bits (provided)
    q_min = -(2 ** (num_bits - 1))
    q_max = 2 ** (num_bits - 1) - 1
    W_min, W_max = W.min(), W.max()

    # Degenerate case: all weights identical -> nothing to quantize (avoids divide-by-zero)
    if torch.isclose(W_max, W_min):
        return W.clone()

    # Compute scale and zero-point
    S = (W_max - W_min) / float(q_max - q_min)
    Z = q_min - W_min / S

    # Quantize
    q = torch.round(W / S + Z)
    q = torch.clamp(q, q_min, q_max)

    # Dequantize
    W_hat = (q - Z) * S


    return W_hat



def kmeans_quantize_tensor(W, num_bits, seed=SEED, max_iter=20):
    shape = W.shape
    flat = W.detach().cpu().numpy().reshape(-1).astype(np.float64)   # 1-D weights (provided)
    n_clusters = min(2 ** num_bits, len(np.unique(flat)))           # K = 2**num_bits (capped by #unique values)

    # Step 0 (provided): randomly initialize the centroids from distinct weight values
    rng = np.random.default_rng(seed)
    centroids = flat[rng.choice(len(flat), size=n_clusters, replace=False)].copy()
    labels = np.full(len(flat), -1)

    for _ in range(max_iter):

        # Calculate absolute distance
        distances = np.abs(flat[:, None] - centroids[None, :])

        # Assign each weight to its closest centroid
        new_labels = np.argmin(distances, axis=1)


        # Convergence (provided): stop when the assignments no longer change
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for k in range(n_clusters):

          # Get all weights assigned to cluster k
          assigned = flat[labels == k]
          if len(assigned) > 0:
              centroids[k] = assigned.mean()


    # Reconstruct each weight as the value of its assigned centroid (provided)
    W_hat = centroids[labels].reshape(shape)
    return torch.tensor(W_hat, dtype=W.dtype, device=W.device)


# Apply quantization to the model: quantize WEIGHTS only (per layer); biases stay FP32
def quantize_model(model, num_bits, method):
    qmodel = copy.deepcopy(model)
    with torch.no_grad():
        for name, p in qmodel.named_parameters():
            if not name.endswith('weight'):
                continue
            if method == 'linear':
                p.data = linear_quantize_tensor(p.data, num_bits)
            else:
                p.data = kmeans_quantize_tensor(p.data, num_bits)
    return qmodel

# per-layer weight/bias sizes
weight_sizes = [p.numel() for n, p in model.named_parameters() if n.endswith('weight')]
bias_sizes   = [p.numel() for n, p in model.named_parameters() if n.endswith('bias')]

# Mean squared quantization error over all quantized WEIGHTS: ||W - W_hat||^2 / M
# (compare the Base model's weights `model` with the quantized model's weights `qmodel`; skip biases).
def weight_mse(model, qmodel):
    total_error = 0.0
    total_weights = 0

    for (name1, p1), (name2, p2) in zip(
            model.named_parameters(),
            qmodel.named_parameters()):

        # Only calculate error for weight layers
        if not name1.endswith('weight'):
            continue

        # Sum up the squared differences between the original weights and quantized weights
        total_error += torch.sum((p1 - p2) ** 2).item()

        # Keep track of the total number of weights processed
        total_weights += p1.numel()

    return total_error / total_weights



# Full-model storage in bits
def storage_bits(weight_sizes, bias_sizes, num_bits, method):
    M, B, L, N = sum(weight_sizes), sum(bias_sizes), len(weight_sizes), num_bits
    if method == 'original':
        return 32 * (M + B)
    if method == 'linear':
        return M * N + L * (N + 32) + 32 * B
    return M * N + L * (32 * (2 ** N)) + 32 * B

# Sweep both methods over bit-widths; collect accuracy and quantization error
bit_widths = [2, 3, 4, 5, 6]
acc = {'linear': [], 'kmeans': []}
mse = {'linear': [], 'kmeans': []}
for N in bit_widths:
    for method in ['linear', 'kmeans']:
        qmodel = quantize_model(model, N, method)
        acc[method].append(evaluate(qmodel, test_loader))
        mse[method].append(weight_mse(model, qmodel))
        print(f"{method:7s} | {N}-bit | Acc: {acc[method][-1]:6.2f}% | weight MSE: {mse[method][-1]:.3e}")

# Plots: Accuracy and quantization error vs. bit-width
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(bit_widths, acc['linear'], marker='o', label='Linear')
ax[0].plot(bit_widths, acc['kmeans'], marker='s', label='K-Means')
ax[0].axhline(fp_acc, color='gray', linestyle='--', label='Base (FP32)')
ax[0].set_xlabel('Bit-width (N)'); ax[0].set_ylabel('Test Accuracy (%)')
ax[0].set_title('Accuracy vs. Bit-width'); ax[0].legend(); ax[0].grid(True)
ax[1].plot(bit_widths, mse['linear'], marker='o', label='Linear')
ax[1].plot(bit_widths, mse['kmeans'], marker='s', label='K-Means')
ax[1].set_yscale('log')
ax[1].set_xlabel('Bit-width (N)'); ax[1].set_ylabel('Weight MSE (log scale)')
ax[1].set_title('Quantization Error vs. Bit-width'); ax[1].legend(); ax[1].grid(True)
plt.tight_layout(); plt.show()

# Full-model storage / compression-ratio table (Base = original FP32)
print("\n N | Base(bits) |  linear(bits) |  kmeans(bits) | linear x | kmeans x")
for N in bit_widths:
    o  = storage_bits(weight_sizes, bias_sizes, N, 'original')
    l  = storage_bits(weight_sizes, bias_sizes, N, 'linear')
    km = storage_bits(weight_sizes, bias_sizes, N, 'kmeans')
    print(f" {N} | {o:>10} | {l:>12} | {km:>12} | {o/l:>7.2f} | {o/km:>7.2f}")
# Federated Learning & Model Quantization on MNIST

This repository contains two PyTorch implementations exploring distributed machine learning concepts and model compression techniques using the MNIST dataset:
1. **Federated Learning (IID vs. Non-IID)**: Simulates a multi-client federated training environment and evaluates global model convergence under different data distribution scenarios.
2. **Model Quantization (Linear & K-Means)**: Implements post-training weight quantization across multiple bit-widths to measure accuracy degradation, quantization error (MSE), and memory footprint reduction.

---

## 🛠️ Project Overview

### Part 1: Federated Learning (`Assignment2_Part1.py`)
* **Architecture**: Simple Multi-Layer Perceptron (MLP: $784 \rightarrow 390 \rightarrow 200 \rightarrow 10$).
* **Mechanism**: 
  * Splits the MNIST training set across multiple clients ($N=5$).
  * Supports both **IID** (randomly shuffled splits) and **Non-IID** (sorted by class labels) data partitions.
  * Performs local SGD training on each client, aggregates weights via **Federated Averaging (FedAvg)**, and updates the global model across multiple rounds.
* **Evaluation**: Plots test accuracy progression over training rounds comparing IID vs. Non-IID performance.

### Part 2: Model Quantization (`Assignment2_Part2.py`)
* **Base Model**: Pre-trained FP32 MLP trained on MNIST.
* **Quantization Strategies**:
  * **Linear (Uniform) Quantization**: Maps continuous weight ranges uniformly to integer bins based on bit-width ($N$).
  * **K-Means (Non-Uniform) Quantization**: Clusters weights into $2^N$ centroids using an iterative assignment-update loop.
* **Metrics Tracked**:
  * Test Accuracy (%) per bit-width ($2$ to $6$ bits).
  * Weight Mean Squared Error ($\text{MSE}$) against the baseline FP32 model.
  * Total model storage size (bits) and compression ratios.

---

## 📦 Requirements

* Python 3.8+
* PyTorch (`torch`, `torchvision`)
* NumPy (`numpy`)
* Matplotlib (`matplotlib`)

Install dependencies via pip:
```bash
pip install torch torchvision numpy matplotlib

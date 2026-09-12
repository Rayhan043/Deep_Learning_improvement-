# Deep Learning Foundations: Loss Functions & Feature Scaling

Welcome to this module guide! This document breaks down core optimization concepts required to move beyond basic PyTorch `nn.Module` structures and build fast, stable, and well-performing Neural Networks.

---

## 📋 Table of Contents
1. [Module 1: Loss Functions](#module-1-loss-functions)
   - [1.1 Mean Squared Error (MSE) Loss](#11-mean-squared-error-mse-loss)
   - [1.2 Mean Absolute Error (MAE) Loss](#12-mean-absolute-error-mae-loss)
   - [1.3 Huber Loss](#13-huber-loss)
   - [1.4 Binary Cross-Entropy (BCE) Loss](#14-binary-cross-entropy-bce-loss)
   - [1.5 Hinge Loss](#15-hinge-loss)
   - [Loss Functions Comparison](#loss-functions-comparison)
2. [Module 2: How to Improve Neural Networks (Feature Scaling)](#module-2-how-to-improve-neural-networks-feature-scaling)
   - [2.1 Why Feature Scaling Matters](#21-why-feature-scaling-matters)
   - [2.2 Standardization (Z-Score Normalization)](#22-standardization-z-score-normalization)
   - [2.3 Normalization (Min-Max Scaling)](#23-normalization-min-max-scaling)
   - [Standardization vs. Normalization](#standardization-vs-normalization)
3. [Key Takeaways](#key-takeaways)

---

## Module 1: Loss Functions

A loss function (or cost function) measures how far a model's prediction is from the actual target value. During training, backpropagation uses this error value to calculate gradients and update network parameters.

---

### 1.1 Mean Squared Error (MSE) Loss

#### What is it?
MSE calculates the average of the squared differences between predicted values ($\hat{y}$) and actual target values ($y$).

#### Why and when to use it
It is the standard loss function for **Regression** tasks. Because it squares the errors, it severely penalizes large mistakes compared to small ones.

#### Mathematical Formula
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

#### PyTorch Code
```python
import torch
import torch.nn as nn

predictions = torch.tensor([2.5, 0.0, 2.1], requires_grad=True)
targets = torch.tensor([3.0, -0.5, 2.0])

criterion = nn.MSELoss()
loss = criterion(predictions, targets)
print(f"MSE Loss: {loss.item():.4f}")

```

---

### 1.2 Mean Absolute Error (MAE) Loss

#### What is it?

MAE takes the average of the absolute differences between targets and predictions.

#### Why and when to use it

Unlike MSE, MAE does not square errors. This makes it far less sensitive to outliers in your dataset, giving a stable loss signal when noisy data is present.

#### Mathematical Formula

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \vert{}y_i - \hat{y}_i\vert{}$$

#### PyTorch Code

```python
import torch
import torch.nn as nn

predictions = torch.tensor([2.5, 0.0, 2.1], requires_grad=True)
targets = torch.tensor([3.0, -0.5, 2.0])

criterion = nn.L1Loss()  # L1Loss is MAE in PyTorch
loss = criterion(predictions, targets)
print(f"MAE Loss: {loss.item():.4f}")

```

---

### 1.3 Huber Loss

#### What is it?

Huber Loss combines the best properties of MSE and MAE. It acts quadratically for small errors and switches to linear behavior for larger errors beyond a threshold $\delta$ (delta).

#### Why and when to use it

Use Huber Loss when you want the quick convergence of MSE near zero error, combined with the outlier robustness of MAE for large errors.

#### Mathematical Formula

$$\text{Huber Loss} = \begin{cases} \frac{1}{2}(y_i - \hat{y}_i)^2 & \text{if } \vert{}y_i - \hat{y}_i\vert{} \le \delta \\ \delta \vert{}y_i - \hat{y}_i\vert{} - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

#### PyTorch Code

```python
import torch
import torch.nn as nn

predictions = torch.tensor([2.5, 0.0, 2.1], requires_grad=True)
targets = torch.tensor([3.0, -0.5, 2.0])

criterion = nn.HuberLoss(delta=1.0)
loss = criterion(predictions, targets)
print(f"Huber Loss: {loss.item():.4f}")

```

---

### 1.4 Binary Cross-Entropy (BCE) Loss

#### What is it?

BCE (Log Loss) measures performance in binary classification tasks where predictions are probabilities bounded between $0$ and $1$.

#### Why and when to use it

It penalizes incorrect predictions exponentially when the model is confident but wrong (e.g., predicting $0.99$ probability for a ground truth label of $0$).

#### Mathematical Formula

$$\text{BCE} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

#### PyTorch Code

```python
import torch
import torch.nn as nn

# Predictions must pass through a Sigmoid activation first
probabilities = torch.tensor([0.9, 0.2, 0.8], requires_grad=True)
targets = torch.tensor([1.0, 0.0, 1.0])

criterion = nn.BCELoss()
loss = criterion(probabilities, targets)
print(f"BCE Loss: {loss.item():.4f}")

# Numerical stability tip: Use BCEWithLogitsLoss directly on raw output logits
logits = torch.tensor([2.2, -1.3, 1.5], requires_grad=True)
robust_criterion = nn.BCEWithLogitsLoss()
robust_loss = robust_criterion(logits, targets)

```

---

### 1.5 Hinge Loss

#### What is it?

Hinge Loss is an error metric primarily used in maximum-margin classifiers like Support Vector Machines (SVMs). It expects targets to be labeled as $-1$ or $+1$.

#### Why and when to use it

Instead of just classifying data correctly, Hinge Loss penalizes predictions that cross or fall within the safety margin of the decision boundary.

#### Mathematical Formula

$$\text{Hinge Loss} = \max(0, 1 - y \cdot \hat{y})$$

#### PyTorch Code

```python
import torch
import torch.nn as nn

predictions = torch.tensor([0.8, -0.5, 0.2], requires_grad=True)
targets = torch.tensor([1.0, -1.0, -1.0])  # Targets must be 1 or -1

criterion = nn.HingeEmbeddingLoss()
loss = criterion(predictions, targets)
print(f"Hinge Loss: {loss.item():.4f}")

```

---

### Loss Functions Comparison

| Loss Function | Primary Application | Outlier Robustness | Penalty Behavior |
| --- | --- | --- | --- |
| **MSE** | Regression | Low | Quadratic (squares large errors) |
| **MAE** | Regression | High | Linear |
| **Huber** | Regression | High | Quadratic for small errors, Linear for large |
| **BCE** | Binary Classification | Medium | Logarithmic / Exponential |
| **Hinge** | Margin Classification (SVM) | High | Linear once boundary margin is breached |

---

## Module 2: How to Improve Neural Networks (Feature Scaling)

---

### 2.1 Why Feature Scaling Matters

Neural networks optimize parameters using Gradient Descent. If input features exist on wildly different scales (e.g., `Age` from 18 to 80 vs. `Salary` from $20,000 to $200,000), the loss contours become distorted ellipses.

```
Unscaled Features (Slow & Oscillating)       Scaled Features (Fast & Direct)
        \     /     /                              \   |   /
         \   /     /                                \  |  /
        (  O  )   /                                  ( O )

```

Scaling realigns input features so gradients descend along smooth paths, speeding up training and preventing numerical instability.

---

### 2.2 Standardization (Z-Score Normalization)

#### What is it?

Standardization rescales feature distributions to have a **mean ($\mu$) of 0** and a **standard deviation ($\sigma$) of 1**.

#### Mathematical Formula

$$X_{\text{std}} = \frac{X - \mu}{\sigma}$$

#### Step-by-Step Math Example

Let $X = [10, 20, 30]$:

1. **Calculate Mean ($\mu$):**

$$\mu = \frac{10 + 20 + 30}{3} = 20$$


2. **Calculate Standard Deviation ($\sigma$):**

$$\sigma^2 = \frac{(10-20)^2 + (20-20)^2 + (30-20)^2}{3} = \frac{100 + 0 + 100}{3} \approx 66.67$$


$$\sigma = \sqrt{66.67} \approx 8.165$$


3. **Standardize value $X = 30$:**

$$X_{\text{std}} = \frac{30 - 20}{8.165} = \frac{10}{8.165} \approx 1.225$$



#### PyTorch Code

```python
import torch

X = torch.tensor([[10.0], [20.0], [30.0]])

mean = X.mean(dim=0)
std = X.std(dim=0, unbiased=False)

X_scaled = (X - mean) / std
print("Standardized Output:\n", X_scaled)

```

---

### 2.3 Normalization (Min-Max Scaling)

#### What is it?

Normalization rescales data values strictly into a fixed range, usually **[0, 1]**.

#### Mathematical Formula

$$X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$

#### Step-by-Step Math Example

Let $X = [10, 20, 30]$ where $X_{\min} = 10$ and $X_{\max} = 30$:

1. **Normalize value $X = 20$:**

$$X_{\text{norm}} = \frac{20 - 10}{30 - 10} = \frac{10}{20} = 0.5$$



#### PyTorch Code

```python
import torch

X = torch.tensor([[10.0], [20.0], [30.0]])

X_min = X.min(dim=0).values
X_max = X.max(dim=0).values

X_normalized = (X - X_min) / (X_max - X_min)
print("Normalized Output:\n", X_normalized)

```

---

### Standardization vs. Normalization

| Feature | Standardization (Z-score) | Normalization (Min-Max) |
| --- | --- | --- |
| **Output Range** | Unbounded ($\approx [-3, 3]$ typically) | Strictly bounded ($[0, 1]$) |
| **Outlier Handling** | Robust (preserves relative distances) | Sensitive (outliers compress non-outliers) |
| **Best Used For** | General Neural Networks, Gaussian data | Image pixels, bounded ranges, distance algorithms |

---

## Key Takeaways

1. Match loss functions directly to your task: **MSE/Huber** for continuous targets, **BCE** for binary labels.
2. Always scale input features before feeding them into neural network layers to avoid oscillating gradients and slow convergence.
3. **Standardization** is generally preferred over Min-Max scaling for deep learning unless your data relies on bounded range constraints (like image pixels).

```

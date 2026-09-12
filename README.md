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

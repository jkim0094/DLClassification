# DL_Classification

## Project Overview

This repository contains the full implementation and experiments for a deep learning project.
The project is organized into three parts, gradually increasing in complexity, from manually implementing a feed-forward neural network to building a robust convolutional model capable of withstanding adversarial attacks.

---

### Project Structure

### **Part 1: Manual Feed-forward Neural Network on MNIST**

**Objective**: Learn the fundamentals of deep learning by building everything from scratch using PyTorch's auto-differentiation.

* Implement `MyLinear` and `MyFFN` classes manually (no high-level PyTorch modules used)
* Create and apply custom optimizers:

  * `SGD`, `SGD with Momentum`, `Adagrad`
* Train the model on the MNIST dataset for **50 epochs**
* Evaluate accuracy and loss using self-defined functions

---

### **Part 2: DNN for FashionMNIST + Generalization Enhancement**

**Objective**: Design a tunable deep neural network (DNN) and apply modern generalization techniques.

* Network architecture:
  `784 → n1(act) → n2(act) → 10(softmax)`
  where:
  `n1 ∈ {20, 40}`, `n2 ∈ {20, 40}`, `act ∈ {sigmoid, tanh, relu}`
* Implement custom loss:
  $\text{Loss}(p, y) = \text{CrossEntropy}(1_y, p) + \lambda H(p) \quad (\lambda = 0.1)$
  where $H(p) = -\sum p_i \log p_i$ is entropy
* Apply **Sharpness-Aware Minimization (SAM)** to improve generalization
  → [Official SAM Paper (ICLR 2021)](https://openreview.net/pdf?id=6Tm1mposlrM)

---

### **Part 3: Custom CNN + Adversarial Robustness**

**Objective**: Build a modular CNN for multi-class image classification (20 classes) and improve robustness via regularization and adversarial training.

* Define CNN blocks with customizable features:

  * `Conv → BatchNorm → Act → Conv → BatchNorm → Act → MaxPool → Dropout`
* Architecture flexibility via:

  * `list_feature_maps`, `drop_rate`, `batch_norm`, `use_skip`
* Experiments include:

  * **Mixup** and **CutMix** for generalization improvement
  * **One-vs-All (OVA)** loss vs. standard cross-entropy
  * Implement and test **PGD attack** with ε = 0.0313, k = 20, η = 0.002
  * **Adversarial Training** using PGD
    → Evaluate robustness under both PGD and FGSM attacks


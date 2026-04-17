# Concepts Glossary
<!-- Add entries here as you encounter new terms. Link them from your phase pages. -->
<!-- Format: ## Term \n **What it is:** \n **Where it shows up:** \n **My understanding:** -->

---

## Tensor
- **What it is:** An n-dimensional array. A scalar is a 0D tensor, a vector is 1D, a matrix is 2D, and anything beyond is an nd tensor.
- **Where it shows up:** Core data structure in PyTorch and numpy. Everything in ML flows through tensors.
- **My understanding:** 

---

## Gradient
- **What it is:** The partial derivative of the loss with respect to a parameter. Tells you the direction and magnitude to adjust each weight.
- **Where it shows up:** Backpropagation, autograd, optimizers.
- **My understanding:** 

---

## Loss Function
- **What it is:** A function that measures how wrong the model's predictions are. Training minimizes this.
- **Where it shows up:** Every supervised learning problem.
- **My understanding:** 

---

## Backpropagation
- **What it is:** The algorithm that computes gradients by applying the chain rule backwards through the computation graph.
- **Where it shows up:** Training neural networks — Karpathy's micrograd makes this very concrete.
- **My understanding:** 

---

## Learning Rate
- **What it is:** A scalar that controls how big a step the optimizer takes in the direction of the gradient.
- **Where it shows up:** Every optimizer — SGD, Adam, etc.
- **My understanding:** 

---

## Overfitting
- **What it is:** When a model learns the training data too well and fails to generalize to new data.
- **Where it shows up:** Everywhere. Fixed with regularization, dropout, more data, early stopping.
- **My understanding:** 

---

## Epoch
- **What it is:** One full pass through the entire training dataset.
- **Where it shows up:** Training loops.
- **My understanding:** 

---

## Batch / Mini-batch
- **What it is:** A subset of the training data processed together before updating weights.
- **Where it shows up:** SGD → Mini-batch SGD. Controls memory usage and training stability.
- **My understanding:** 

---

<!-- Add new concepts below as you encounter them -->

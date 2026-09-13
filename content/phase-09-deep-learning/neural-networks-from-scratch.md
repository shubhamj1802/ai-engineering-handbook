---
title: Neural Networks from Scratch
order: 1
difficulty: Advanced
duration: 18
badges: ["Hands-on", "Deep dive"]
summary: "Neurons, layers, activations, loss and backpropagation — implemented in NumPy so the PyTorch version that follows contains no magic."
prereqs: ["Aggregations, Linear Algebra and Random Numbers", "The Core Algorithms"]
keyConcepts: ["neuron", "activation", "forward pass", "backpropagation", "gradient descent"]
---

## Why this matters

You will not train foundation models. You *will* read architecture diagrams, debug shape
errors, fine-tune small models, and reason about why training diverged. All of that rests on
one loop — forward pass, loss, gradients, update — and implementing it once in forty lines
of NumPy makes it permanent.

## Mental Model

```text
ONE NEURON:   z = w·x + b          →   a = f(z)
              weighted sum             non-linearity

A LAYER:      Z = XW + b           X (batch, in) · W (in, out) → Z (batch, out)

A NETWORK:    X → Layer → ReLU → Layer → ReLU → Layer → output

TRAINING LOOP:
  1. forward    compute predictions
  2. loss       measure how wrong
  3. backward   compute ∂loss/∂w for every weight   ← backpropagation
  4. update     w ← w - learning_rate · gradient    ← gradient descent
  repeat
```

Without the non-linearity, stacking layers is pointless: a composition of linear maps is
still a linear map. The activation function is what makes depth meaningful.

## Core Concepts

### The neuron

```python
import numpy as np

x = np.array([0.5, -1.2, 3.0])      # inputs
w = np.array([0.8, 0.3, -0.5])      # learned weights
b = 0.1                              # learned bias

z = x @ w + b                        # -1.71  pre-activation
a = max(0.0, z)                      # 0.0    ReLU activation
```

### Layers as matrix multiplication

```python
batch, n_in, n_out = 32, 10, 64

X = np.random.randn(batch, n_in)                 # (32, 10)
W = np.random.randn(n_in, n_out) * 0.1           # (10, 64)
b = np.zeros(n_out)                              # (64,)

Z = X @ W + b                                    # (32, 64)  broadcasting adds b to each row
A = np.maximum(0, Z)                             # ReLU
```

Every "layer" in every framework is this, plus bookkeeping.

### Activation functions

| Function | Formula | Range | Use |
| --- | --- | --- | --- |
| ReLU | `max(0, z)` | [0, ∞) | hidden layers, the default |
| Leaky ReLU | `max(0.01z, z)` | (−∞, ∞) | when ReLU units die |
| GELU | `z·Φ(z)` | (−∞, ∞) | transformers |
| Sigmoid | `1/(1+e⁻ᶻ)` | (0, 1) | binary output layer |
| Softmax | `eᶻⁱ/Σeᶻʲ` | (0,1), sums to 1 | multi-class output layer |
| Tanh | `tanh(z)` | (−1, 1) | older RNNs |

ReLU won because it is cheap and its gradient is exactly 1 for positive inputs, which avoids
the vanishing-gradient problem that made deep sigmoid networks untrainable.

### Loss functions

```python
# regression
mse = np.mean((y_true - y_pred) ** 2)

# binary classification (with probabilities from sigmoid)
eps = 1e-12
bce = -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

# multi-class (with probabilities from softmax)
ce = -np.mean(np.log(y_pred[np.arange(len(y_true)), y_true] + eps))
```

Cross-entropy is the right loss for classification because its gradient is proportional to
the error: confidently wrong predictions produce large gradients, so the model is corrected
fastest exactly where it is worst.

### Backpropagation — the chain rule, applied backwards

To update a weight you need `∂loss/∂w`. Backprop computes all of them in one backward sweep
by reusing intermediate results.

```text
forward:   X → Z₁ → A₁ → Z₂ → ŷ → L

backward:  ∂L/∂ŷ → ∂L/∂Z₂ → ∂L/∂W₂, ∂L/∂A₁ → ∂L/∂Z₁ → ∂L/∂W₁

for a linear layer Z = A_prev @ W + b:
  ∂L/∂W      = A_prev.T @ ∂L/∂Z
  ∂L/∂b      = sum(∂L/∂Z, axis=0)
  ∂L/∂A_prev = ∂L/∂Z @ W.T
```

Those three lines are the entire algorithm. Everything else is layer-specific detail.

### Gradient descent

```python
W -= learning_rate * dW
b -= learning_rate * db
```

| Variant | Batch size | Character |
| --- | --- | --- |
| Batch GD | all data | stable, slow, memory-hungry |
| SGD | 1 | noisy, can escape shallow minima |
| **Mini-batch** | 32–512 | the practical default |

The learning rate is the most important hyperparameter: too small and training crawls, too
large and the loss diverges to NaN.

## Minimal Example

A two-layer network for binary classification, complete, in NumPy.

```python title="nn_from_scratch.py"
"""A neural network in ~70 lines. No frameworks, no magic."""
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(42)


def relu(z):                 return np.maximum(0, z)
def relu_grad(z):            return (z > 0).astype(float)
def sigmoid(z):              return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


class TwoLayerNet:
    def __init__(self, n_in: int, n_hidden: int, n_out: int = 1) -> None:
        # He initialisation: variance 2/fan_in keeps activations from vanishing with ReLU
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        self.Z1 = X @ self.W1 + self.b1          # (batch, hidden)
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2    # (batch, 1)
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def loss(self, y_true: np.ndarray) -> float:
        eps = 1e-12
        p = np.clip(self.A2.ravel(), eps, 1 - eps)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    def backward(self, y_true: np.ndarray) -> dict[str, np.ndarray]:
        m = y_true.shape[0]
        y = y_true.reshape(-1, 1)

        # sigmoid + binary cross-entropy combine to this simple gradient
        dZ2 = (self.A2 - y) / m                            # (batch, 1)
        dW2 = self.A1.T @ dZ2                              # (hidden, 1)
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ self.W2.T                              # (batch, hidden)
        dZ1 = dA1 * relu_grad(self.Z1)
        dW1 = self.X.T @ dZ1                               # (in, hidden)
        db1 = dZ1.sum(axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def step(self, grads: dict[str, np.ndarray], learning_rate: float) -> None:
        for name, gradient in grads.items():
            setattr(self, name, getattr(self, name) - learning_rate * gradient)


def train(net, X, y, *, epochs=400, batch_size=64, learning_rate=0.1, X_val=None, y_val=None):
    n = X.shape[0]
    history = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)                 # reshuffle every epoch
        epoch_loss = 0.0

        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            net.forward(X[idx])
            epoch_loss += net.loss(y[idx]) * len(idx)
            net.step(net.backward(y[idx]), learning_rate)

        if epoch % 50 == 0 or epoch == 1:
            train_loss = epoch_loss / n
            row = {"epoch": epoch, "train_loss": round(train_loss, 4)}
            if X_val is not None:
                net.forward(X_val)
                row["val_loss"] = round(net.loss(y_val), 4)
                row["val_acc"] = round(float(((net.A2.ravel() > 0.5) == y_val).mean()), 4)
            history.append(row)
            print(row)
    return history


if __name__ == "__main__":
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Two interleaving crescents: not linearly separable, so a hidden layer is required.
    X, y = make_moons(n_samples=4_000, noise=0.22, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    net = TwoLayerNet(n_in=2, n_hidden=32)
    train(net, X_train, y_train.astype(float), epochs=400, learning_rate=0.3,
          X_val=X_test, y_val=y_test.astype(float))

    net.forward(X_test)
    accuracy = float(((net.A2.ravel() > 0.5) == y_test).mean())
    print(f"\nfinal test accuracy: {accuracy:.4f}")

    from sklearn.linear_model import LogisticRegression
    linear = LogisticRegression().fit(X_train, y_train)
    print(f"logistic regression: {linear.score(X_test, y_test):.4f}   ← cannot bend the boundary")
```

```text
{'epoch': 1, 'train_loss': 0.6021, 'val_loss': 0.4718, 'val_acc': 0.842}
{'epoch': 50, 'train_loss': 0.2103, 'val_loss': 0.2214, 'val_acc': 0.913}
{'epoch': 100, 'train_loss': 0.1884, 'val_loss': 0.2041, 'val_acc': 0.925}
{'epoch': 200, 'train_loss': 0.1721, 'val_loss': 0.1958, 'val_acc': 0.931}
{'epoch': 400, 'train_loss': 0.1642, 'val_loss': 0.1943, 'val_acc': 0.933}

final test accuracy: 0.9330
logistic regression: 0.8760   ← cannot bend the boundary
```

The 5.7-point gap over logistic regression is the entire value of the hidden layer: it can
represent a curved decision boundary. That is what "depth" buys.

## Real-World Example

Verifying your gradients — the technique that catches the bug you will otherwise chase for
hours.

```python title="gradient_check.py"
"""Numerical gradient checking.

Backprop bugs do not raise exceptions - they just make training slightly worse.
Comparing analytic gradients against finite differences catches them immediately.
"""
from __future__ import annotations

import numpy as np


def gradient_check(net, X: np.ndarray, y: np.ndarray, *, epsilon: float = 1e-5,
                   tolerance: float = 1e-6, samples: int = 12) -> dict[str, float]:
    """For a few random weights, compare the analytic gradient with (L(w+ε)-L(w-ε))/2ε."""
    rng = np.random.default_rng(0)

    net.forward(X)
    analytic = net.backward(y)
    results: dict[str, float] = {}

    for name in ["W1", "b1", "W2", "b2"]:
        parameter = getattr(net, name)
        flat = parameter.ravel()
        flat_grad = analytic[name].ravel()

        worst_relative_error = 0.0
        for index in rng.choice(flat.size, size=min(samples, flat.size), replace=False):
            original = flat[index]

            flat[index] = original + epsilon
            net.forward(X)
            loss_plus = net.loss(y)

            flat[index] = original - epsilon
            net.forward(X)
            loss_minus = net.loss(y)

            flat[index] = original                      # restore

            numeric = (loss_plus - loss_minus) / (2 * epsilon)
            analytic_value = flat_grad[index]
            denominator = max(abs(numeric) + abs(analytic_value), 1e-12)
            worst_relative_error = max(worst_relative_error,
                                       abs(numeric - analytic_value) / denominator)

        results[name] = worst_relative_error

    net.forward(X)                                       # leave the net in a clean state
    return results


if __name__ == "__main__":
    from nn_from_scratch import TwoLayerNet

    rng = np.random.default_rng(1)
    X = rng.normal(size=(24, 5))
    y = (rng.random(24) > 0.5).astype(float)

    net = TwoLayerNet(n_in=5, n_hidden=8)
    errors = gradient_check(net, X, y)

    for name, error in errors.items():
        verdict = "OK" if error < 1e-6 else "SUSPECT" if error < 1e-4 else "BROKEN"
        print(f"{name:<4} max relative error {error:.2e}  {verdict}")
```

```text
W1   max relative error 3.41e-11  OK
b1   max relative error 1.02e-11  OK
W2   max relative error 8.77e-12  OK
b2   max relative error 4.16e-12  OK
```

Relative error below 1e-6 means the backward pass matches the forward pass. Above 1e-4, the
derivation is wrong. This check costs thirty lines and has saved countless afternoons — and
it is the technique PyTorch's `torch.autograd.gradcheck` automates.

## Common Mistakes

:::mistake
```python
# 1. No non-linearity
Z1 = X @ W1 + b1
Z2 = Z1 @ W2 + b2            # a linear model with extra steps

# 2. Zero initialisation
self.W1 = np.zeros((n_in, n_hidden))     # every neuron computes the same thing forever
# use He (ReLU) or Xavier/Glorot (tanh) initialisation

# 3. Unscaled inputs
# features on wildly different scales make the loss surface a ravine; gradients oscillate

# 4. Learning rate too high
# loss → nan in a few steps. Halve it until it descends smoothly.

# 5. Forgetting to average over the batch
dZ2 = (A2 - y)               # gradient scales with batch size → effective LR changes
dZ2 = (A2 - y) / m           # correct

# 6. log(0)
np.log(p)                    # -inf → nan
np.log(np.clip(p, 1e-12, 1 - 1e-12))

# 7. Not shuffling between epochs
# the network learns the order of the data
```
:::

## Debugging

A checklist that resolves most training failures:

1. **Overfit a tiny batch first.** Take 10 examples and train until the loss is ~0. If you
   cannot, the implementation is broken, not the hyperparameters.
2. **Check the initial loss.** For balanced binary classification it should be ≈ `ln(2)` =
   0.693. A wildly different value means the output layer or the loss is wrong.
3. **Gradient-check.** As above.
4. **Print activation statistics per layer.** All zeros means dead ReLUs; exploding means the
   learning rate or initialisation is wrong.
5. **Plot the loss.** Not decreasing → learning rate too low or a bug. Spiky → too high.
   Decreasing then rising on validation → overfitting.

```python
for name, activation in [("Z1", net.Z1), ("A1", net.A1), ("Z2", net.Z2)]:
    print(f"{name}: mean={activation.mean():+.4f} std={activation.std():.4f} "
          f"dead={(activation == 0).mean():.1%}")
```

## Performance Considerations

- Matrix multiplication dominates. A layer's cost is `batch × n_in × n_out` multiply-adds.
- Larger batches use hardware better but generalise slightly worse; 32–256 is the usual
  range.
- `float32` everywhere; `float64` doubles memory and bandwidth for no benefit in training.
- NumPy on CPU is fine for toy networks; anything real needs a GPU and a framework, which is
  the next lesson.

## Hands-on Exercise

:::exercise Extend the network
Starting from `nn_from_scratch.py`:

1. Generalise to any number of hidden layers: `NeuralNet([2, 32, 16, 1])`.
2. Add L2 regularisation: add `λ/2 · Σw²` to the loss and `λ·W` to each weight gradient.
3. Add momentum: keep a velocity per parameter, `v = βv + (1-β)g`, and update with `v`.
4. Add early stopping on validation loss with a patience of 20 epochs.
5. Compare on `make_moons(noise=0.35)`: no regularisation, L2 only, L2 + momentum. Report
   the best validation loss and the epoch it occurred.

Gradient-check the multi-layer version before you trust any of its numbers.
:::

:::solution Key parts
```python title="deeper_net.py"
class NeuralNet:
    def __init__(self, sizes: list[int], *, l2: float = 0.0, momentum: float = 0.9) -> None:
        self.sizes, self.l2, self.momentum = sizes, l2, momentum
        self.W = [rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.vW = [np.zeros_like(w) for w in self.W]      # momentum buffers
        self.vb = [np.zeros_like(b) for b in self.b]

    def forward(self, X):
        self.A = [X]
        self.Z = []
        for i, (W, b) in enumerate(zip(self.W, self.b, strict=True)):
            Z = self.A[-1] @ W + b
            self.Z.append(Z)
            self.A.append(sigmoid(Z) if i == len(self.W) - 1 else relu(Z))
        return self.A[-1]

    def backward(self, y):
        m = y.shape[0]
        dZ = (self.A[-1] - y.reshape(-1, 1)) / m
        dW, db = [None] * len(self.W), [None] * len(self.b)

        for i in reversed(range(len(self.W))):
            dW[i] = self.A[i].T @ dZ + self.l2 * self.W[i]     # L2 term
            db[i] = dZ.sum(axis=0)
            if i > 0:
                dZ = (dZ @ self.W[i].T) * relu_grad(self.Z[i - 1])
        return dW, db

    def step(self, dW, db, learning_rate: float) -> None:
        for i in range(len(self.W)):
            self.vW[i] = self.momentum * self.vW[i] + (1 - self.momentum) * dW[i]
            self.vb[i] = self.momentum * self.vb[i] + (1 - self.momentum) * db[i]
            self.W[i] -= learning_rate * self.vW[i]
            self.b[i] -= learning_rate * self.vb[i]
```

```text
configuration            best_val_loss  best_epoch  test_acc
no regularisation               0.3218         142     0.8712
L2 (lambda=1e-3)                0.3104         198     0.8790
L2 + momentum                   0.3041          96     0.8824
```

Momentum reached a better loss in half the epochs — the same acceleration that makes Adam
the default optimiser in the next lesson.
:::

## Challenge

:::challenge Implement a softmax classifier
Extend the network to multi-class: softmax output, categorical cross-entropy loss, and the
matching backward pass (the combined softmax + cross-entropy gradient simplifies beautifully
to `(probabilities - one_hot) / m` — derive it before you look it up).

Train it on `sklearn.datasets.load_digits` (8×8 handwritten digits, 10 classes) and reach
≥95% test accuracy. Then display the learned first-layer weights as 8×8 images: you will see
stroke and edge detectors appear. That is representation learning, visible.
:::

## Interview Questions

:::interview
1. Why do neural networks need non-linear activations?
2. What does backpropagation actually compute, and how?
3. Why is ReLU preferred over sigmoid in hidden layers?
4. What happens if you initialise all weights to zero?
5. How would you debug a network whose loss becomes NaN?
:::

## Cheat Sheet

```python
forward:   Z = A_prev @ W + b ;  A = activation(Z)
backward:  dW = A_prev.T @ dZ ;  db = dZ.sum(0) ;  dA_prev = dZ @ W.T
           dZ_hidden = dA * activation_grad(Z)
update:    W -= lr * dW

init:      He   N(0, sqrt(2/fan_in))   for ReLU
           Xavier N(0, sqrt(1/fan_in)) for tanh/sigmoid
loss:      MSE (regression) · BCE (binary) · categorical CE (multi-class)
sigmoid+BCE and softmax+CE both give dZ = (pred - target)/m

debug:     overfit 10 examples · initial loss ≈ ln(k) · gradient check
           activation stats per layer · plot the loss curve
```

```quiz
[
  {
    "question": "You stack three linear layers with no activation functions. What have you built?",
    "options": [
      "A deep network with more capacity",
      "A single linear model - composing linear maps gives a linear map",
      "An unstable network",
      "A convolutional network"
    ],
    "answer": 1,
    "explanation": "Non-linearities are what make depth meaningful. Without them the three weight matrices collapse into one effective matrix."
  },
  {
    "question": "The loss becomes NaN after a few steps. What do you try first?",
    "options": [
      "Add more layers",
      "Reduce the learning rate, and check for log(0) or division by zero",
      "Train for more epochs",
      "Increase the batch size"
    ],
    "answer": 1,
    "explanation": "NaN almost always means exploding gradients (learning rate too high) or an invalid operation such as log(0). Clip probabilities and halve the learning rate."
  },
  {
    "question": "Why initialise weights randomly rather than to zero?",
    "options": [
      "Zero weights make training slower",
      "With identical weights every neuron computes the same output and the same gradient, so they never differentiate",
      "Zero is not a valid float",
      "It changes the loss function"
    ],
    "answer": 1,
    "explanation": "Symmetry breaking. Random initialisation (He for ReLU) also keeps activation variance stable across layers."
  }
]
```

## Summary

- A layer is `XW + b` followed by a non-linearity; depth without non-linearity is pointless.
- Backprop is the chain rule applied backwards, reusing forward-pass values.
- Three lines per linear layer give every gradient you need.
- Debug by overfitting a tiny batch, checking the initial loss, and gradient checking.

## Next Step

PyTorch — the same loop with autograd, GPU support and the layer library, plus a real
training project.

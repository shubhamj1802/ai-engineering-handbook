---
title: "PyTorch — Tensors, Autograd and a Real Training Loop"
order: 2
difficulty: Advanced
duration: 20
badges: ["Hands-on", "Project"]
summary: "Tensors, autograd, Modules, Datasets and DataLoaders — then a complete text-classification project with checkpoints, early stopping and inference."
prereqs: ["Neural Networks from Scratch"]
keyConcepts: ["tensor", "autograd", "nn.Module", "DataLoader", "optimizer", "checkpoint"]
---

:::note In one line
**Every training loop is the same five lines.** Forward, loss, zero the gradients, backward, step. Everything else is scaffolding.
:::

## Why this matters

PyTorch is the language the field writes in. Every model on Hugging Face, every fine-tuning
script, every embedding model you will run locally in Phase 11 is PyTorch. You do not need
to train foundation models to need fluency here — you need it to read code, fine-tune a
small classifier, and understand what a framework is doing on your behalf.

## Mental Model

Every training loop you will ever read is the same five steps, in the same order.
<figure class="lesson-figure">
<svg viewBox="0 0 660 230" role="img" aria-label="Diagram of the five steps in a training loop: forward pass to predict, compute the loss, zero the gradients, backward pass, then the optimiser step, repeating for each batch.">
  <defs>
    <marker id="tr-a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--accent)"/>
    </marker>
  </defs>
  <rect x="14" y="52" width="112" height="56" rx="9" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.7"/>
  <text class="dg-label" x="70" y="74" text-anchor="middle" fill="var(--accent)">1 · forward</text>
  <text class="dg-sub"   x="70" y="92" text-anchor="middle">predict</text>
  <rect x="146" y="52" width="112" height="56" rx="9" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.7"/>
  <text class="dg-label" x="202" y="74" text-anchor="middle" fill="var(--accent)">2 · loss</text>
  <text class="dg-sub"   x="202" y="92" text-anchor="middle">how wrong?</text>
  <rect x="278" y="52" width="112" height="56" rx="9" fill="var(--panel-2)" stroke="var(--warn)" stroke-width="1.8"/>
  <text class="dg-label" x="334" y="74" text-anchor="middle" fill="var(--warn)">3 · zero</text>
  <text class="dg-sub"   x="334" y="92" text-anchor="middle">clear old grads</text>
  <rect x="410" y="52" width="112" height="56" rx="9" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.7"/>
  <text class="dg-label" x="466" y="74" text-anchor="middle" fill="var(--accent)">4 · backward</text>
  <text class="dg-sub"   x="466" y="92" text-anchor="middle">who to blame</text>
  <rect x="542" y="52" width="104" height="56" rx="9" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.8"/>
  <text class="dg-label" x="594" y="74" text-anchor="middle" fill="var(--ok)">5 · step</text>
  <text class="dg-sub"   x="594" y="92" text-anchor="middle">nudge weights</text>
  <path class="dg-arrow" d="M126,80 L142,80" marker-end="url(#tr-a)"/>
  <path class="dg-arrow" d="M258,80 L274,80" marker-end="url(#tr-a)"/>
  <path class="dg-arrow" d="M390,80 L406,80" marker-end="url(#tr-a)"/>
  <path class="dg-arrow" d="M522,80 L538,80" marker-end="url(#tr-a)"/>
  <path d="M594,108 Q594,146 330,146 L70,146 L70,114" stroke="var(--accent)" stroke-width="1.8" fill="none" stroke-dasharray="5 4" marker-end="url(#tr-a)"/>
  <text class="dg-sub" x="330" y="164" text-anchor="middle" fill="var(--accent)">repeat, once per batch</text>
  <rect x="14" y="182" width="632" height="40" rx="8" fill="var(--panel)" stroke="var(--danger)" stroke-width="1.4" stroke-dasharray="5 4"/>
  <text class="dg-sub" x="330" y="206" text-anchor="middle" fill="var(--danger)">Forget step 3 and gradients pile up across batches. Training quietly goes wrong with no error message.</text>
</svg>
<figcaption>
<strong>Five lines, every time.</strong> Once you recognise this shape, any PyTorch training
script becomes readable — everything around it is data loading and logging.
</figcaption>
</figure>

```text
NumPy array        + gradient tracking  + GPU support   =  torch.Tensor

your NumPy loop                     PyTorch equivalent
─────────────────────────────────────────────────────────────
forward pass by hand         →      model(x)
backward pass by hand        →      loss.backward()      (autograd)
w -= lr * dw by hand         →      optimizer.step()
zero the gradients           →      optimizer.zero_grad()
```

Autograd records every operation on tensors that require gradients into a graph, then walks
it backwards. You write only the forward pass.

## Prerequisites

```bash
uv add torch --index https://download.pytorch.org/whl/cpu     # CPU build, small and fast to install
uv add scikit-learn numpy
```

## Core Concepts

### Tensors

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
torch.zeros(3, 4)  torch.ones(2, 2)  torch.randn(32, 10)  torch.arange(10)

x.shape          # torch.Size([3])
x.dtype          # torch.float32   ← PyTorch's default, unlike NumPy's float64
x.device         # device(type='cpu')

x.reshape(3, 1)  x.unsqueeze(0)  x.squeeze()  x.permute(1, 0)  x.T
a @ b            a * b            a.sum(dim=0)  a.mean(dim=1, keepdim=True)

torch.from_numpy(array)          # shares memory with the NumPy array
tensor.numpy()                   # back again (CPU tensors only)
tensor.to("cuda")                # move to GPU
tensor.detach()                  # same values, no gradient history
```

Device handling that works everywhere:

```python
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()      # Apple Silicon
    else "cpu"
)
model = model.to(device)
x = x.to(device)          # model and data must be on the same device
```

### Autograd

```python
w = torch.tensor([2.0], requires_grad=True)
x = torch.tensor([3.0])

y = w * x
loss = (y - 10) ** 2

loss.backward()           # walks the graph backwards
w.grad                    # tensor([-24.])  = d(loss)/dw
```

Three rules that prevent the usual bugs:

```python
optimizer.zero_grad()     # gradients ACCUMULATE by default - clear them every step

with torch.no_grad():     # inference: no graph, less memory, faster
    predictions = model(x)

loss_value = loss.item()  # detach a scalar for logging; keeping the tensor keeps the graph
```

### nn.Module

```python
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, n_features: int, n_hidden: int = 128, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()                       # always first
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden // 2, n_classes),  # raw logits; the loss applies softmax
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = Classifier(n_features=300)
sum(p.numel() for p in model.parameters() if p.requires_grad)     # parameter count
```

:::warning `model.train()` and `model.eval()` are not optional
Dropout and BatchNorm behave differently in training and inference. Forgetting `model.eval()`
before validation gives you noisy, pessimistic numbers; forgetting `model.train()` after it
silently disables dropout for the rest of training.
:::

### Datasets and DataLoaders

```python
from torch.utils.data import DataLoader, Dataset, TensorDataset


class TextDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.as_tensor(features, dtype=torch.float32)
        self.y = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


loader = DataLoader(
    TextDataset(X_train, y_train),
    batch_size=64,
    shuffle=True,          # True for training, False for validation and test
    num_workers=0,         # >0 spawns worker processes; keep 0 on Windows for small jobs
    pin_memory=True,       # faster host→GPU transfer
    drop_last=False,
)
```

### Losses and optimisers

```python
criterion = nn.CrossEntropyLoss()          # expects raw logits + integer class labels
criterion = nn.BCEWithLogitsLoss()         # binary; numerically safer than sigmoid + BCE
criterion = nn.MSELoss()                   # regression

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

AdamW is the sensible default: adaptive per-parameter learning rates plus correctly
implemented weight decay.

### The training loop

```python
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()            # 1. clear old gradients
        logits = model(xb)               # 2. forward
        loss = criterion(logits, yb)     # 3. loss
        loss.backward()                  # 4. backward
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 5. clip (optional but wise)
        optimizer.step()                 # 6. update

    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            ...
```

Those six lines are every PyTorch training script ever written.

## Real-World Example

A complete project: classify support tickets into eight teams, with checkpointing, early
stopping, class weighting and an inference path.

```python title="src/ticketnet/train.py"
"""Ticket classification with PyTorch.

Production-shaped: reproducible seeds, class weighting, gradient clipping,
LR scheduling, early stopping on validation F1, checkpointed artifacts, and a
final evaluation on an untouched test set.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainingConfig:
    n_features: int = 5_000
    hidden: int = 256
    dropout: float = 0.3
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 40
    patience: int = 6
    max_grad_norm: float = 1.0
    seed: int = 42


@dataclass
class TrainingHistory:
    rows: list[dict] = field(default_factory=list)

    def add(self, **row) -> None:
        self.rows.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in row.items()))

    def best(self, metric: str = "val_f1") -> dict:
        return max(self.rows, key=lambda r: r[metric])


class TicketClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int, *, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),                 # LayerNorm works with any batch size
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate(model: nn.Module, loader: DataLoader, criterion, device) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, n = 0.0, 0
    all_predictions, all_labels = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * len(yb)
            n += len(yb)
            all_predictions.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)
    return total_loss / n, float(f1_score(labels, predictions, average="macro")), predictions, labels


def train_model(
    X_train, y_train, X_val, y_val, *, n_classes: int, config: TrainingConfig, out_dir: Path
) -> tuple[nn.Module, TrainingHistory]:
    set_seed(config.seed)
    device = get_device()
    print(f"device: {device}")

    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(X_train, dtype=torch.float32),
                      torch.as_tensor(y_train, dtype=torch.long)),
        batch_size=config.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.as_tensor(X_val, dtype=torch.float32),
                      torch.as_tensor(y_val, dtype=torch.long)),
        batch_size=256, shuffle=False,
    )

    model = TicketClassifier(config.n_features, n_classes,
                             hidden=config.hidden, dropout=config.dropout).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # class weights: rare teams should not be ignored
    counts = np.bincount(y_train, minlength=n_classes)
    weights = torch.tensor(len(y_train) / (n_classes * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=2)

    history = TrainingHistory()
    best_f1, epochs_without_improvement = -1.0, 0
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "best.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        started = time.perf_counter()
        epoch_loss, seen = 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)         # set_to_none is slightly faster
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item() * len(yb)
            seen += len(yb)

        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_f1)

        history.add(epoch=epoch, train_loss=epoch_loss / seen, val_loss=val_loss,
                    val_f1=val_f1, lr=optimizer.param_groups[0]["lr"],
                    seconds=time.perf_counter() - started)

        if val_f1 > best_f1:
            best_f1, epochs_without_improvement = val_f1, 0
            torch.save({
                "model_state": model.state_dict(),
                "config": asdict(config),
                "n_classes": n_classes,
                "epoch": epoch,
                "val_f1": val_f1,
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"early stopping at epoch {epoch} (best val_f1 {best_f1:.4f})")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state"])
    return model, history


def main() -> int:
    from sklearn.datasets import fetch_20newsgroups

    categories = ["comp.graphics", "rec.autos", "sci.med", "sci.space",
                  "talk.politics.guns", "rec.sport.hockey", "sci.crypt", "misc.forsale"]
    data = fetch_20newsgroups(subset="all", categories=categories,
                              remove=("headers", "footers", "quotes"), random_state=42)

    texts, labels = data.data, np.asarray(data.target)
    X_temp, X_test_text, y_temp, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels)
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)

    config = TrainingConfig()
    vectorizer = TfidfVectorizer(max_features=config.n_features, ngram_range=(1, 2),
                                 min_df=3, sublinear_tf=True, stop_words="english")
    X_train = vectorizer.fit_transform(X_train_text).toarray()      # fit on TRAIN only
    X_val = vectorizer.transform(X_val_text).toarray()
    X_test = vectorizer.transform(X_test_text).toarray()

    out_dir = Path("models/ticketnet")
    model, history = train_model(X_train, y_train, X_val, y_val,
                                 n_classes=len(categories), config=config, out_dir=out_dir)

    device = get_device()
    test_loader = DataLoader(
        TensorDataset(torch.as_tensor(X_test, dtype=torch.float32),
                      torch.as_tensor(y_test, dtype=torch.long)),
        batch_size=256)
    _, test_f1, predictions, truth = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)

    print(f"\nbest epoch: {history.best()['epoch']}  test macro-F1: {test_f1:.4f}\n")
    print(classification_report(truth, predictions, target_names=categories, digits=3))

    import joblib
    joblib.dump(vectorizer, out_dir / "vectorizer.joblib")
    (out_dir / "metrics.json").write_text(
        json.dumps({"test_macro_f1": test_f1, "history": history.rows}, indent=2), encoding="utf-8")

    # A linear baseline, for honesty
    from sklearn.linear_model import LogisticRegression
    baseline = LogisticRegression(max_iter=1_000).fit(X_train, y_train)
    print(f"logistic regression baseline macro-F1: "
          f"{f1_score(y_test, baseline.predict(X_test), average='macro'):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```text
device: cpu
parameters: 1,313,544
epoch=1 train_loss=1.9214 val_loss=1.4021 val_f1=0.5612 lr=0.0010 seconds=2.1
epoch=5 train_loss=0.5124 val_loss=0.6841 val_f1=0.7834 lr=0.0010 seconds=2.0
epoch=12 train_loss=0.2018 val_loss=0.6212 val_f1=0.8241 lr=0.0005 seconds=2.0
epoch=18 train_loss=0.1244 val_loss=0.6488 val_f1=0.8253 lr=0.0003 seconds=2.0
early stopping at epoch 21 (best val_f1 0.8253)

best epoch: 18  test macro-F1: 0.8189

                    precision    recall  f1-score   support
     comp.graphics      0.784     0.812     0.798       195
         rec.autos      0.841     0.858     0.849       198
           sci.med      0.812     0.796     0.804       198
...
logistic regression baseline macro-F1: 0.8106
```

**Read the last line.** A 1.3-million-parameter network beat logistic regression by 0.008
macro-F1 on TF-IDF features. That is not a failure of PyTorch — it is the honest result:
deep learning wins on raw, high-dimensional, structured input (text as tokens, images as
pixels), not on features that are already linearly separable. The real gain comes from
replacing TF-IDF with a pre-trained transformer encoder, which is the next step.

### Inference

```python title="src/ticketnet/predict.py"
"""Load the checkpoint and score new tickets."""
from __future__ import annotations

from pathlib import Path

import joblib
import torch
import torch.nn.functional as F

from .train import TicketClassifier, get_device


class TicketPredictor:
    def __init__(self, model_dir: Path, labels: list[str]) -> None:
        self.device = get_device()
        checkpoint = torch.load(model_dir / "best.pt", map_location=self.device)
        config = checkpoint["config"]

        self.model = TicketClassifier(
            config["n_features"], checkpoint["n_classes"],
            hidden=config["hidden"], dropout=config["dropout"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()                              # ← critical: disables dropout

        self.vectorizer = joblib.load(model_dir / "vectorizer.joblib")
        self.labels = labels

    @torch.no_grad()                                   # no gradient graph during inference
    def predict(self, texts: list[str], *, top_k: int = 3) -> list[list[dict]]:
        features = torch.as_tensor(
            self.vectorizer.transform(texts).toarray(), dtype=torch.float32
        ).to(self.device)

        probabilities = F.softmax(self.model(features), dim=1).cpu().numpy()

        results = []
        for row in probabilities:
            top = row.argsort()[::-1][:top_k]
            results.append([
                {"team": self.labels[i], "probability": round(float(row[i]), 4)} for i in top
            ])
        return results


if __name__ == "__main__":
    predictor = TicketPredictor(Path("models/ticketnet"),
                                ["comp.graphics", "rec.autos", "sci.med", "sci.space",
                                 "talk.politics.guns", "rec.sport.hockey", "sci.crypt", "misc.forsale"])
    for prediction in predictor.predict([
        "My GPU driver crashes when rendering large scenes with ray tracing enabled",
        "Looking for information about encryption key exchange protocols",
    ]):
        print(prediction)
```

```text
[{'team': 'comp.graphics', 'probability': 0.8412}, {'team': 'sci.space', 'probability': 0.0521}, ...]
[{'team': 'sci.crypt', 'probability': 0.9104}, {'team': 'comp.graphics', 'probability': 0.0312}, ...]
```

## Common Mistakes

:::mistake
```python
# 1. Forgetting zero_grad()
loss.backward(); optimizer.step()          # gradients accumulate across batches

# 2. Forgetting model.eval() / model.train()
# dropout stays active during validation → noisy, pessimistic metrics

# 3. No torch.no_grad() at inference
# builds a graph, wastes memory, can OOM on large batches

# 4. Accumulating loss tensors
total += loss                               # keeps the whole graph alive → OOM
total += loss.item()                        # correct

# 5. Model and data on different devices
# RuntimeError: Expected all tensors to be on the same device

# 6. Softmax before CrossEntropyLoss
nn.CrossEntropyLoss()(F.softmax(logits), y)   # double softmax; pass raw logits

# 7. Shuffling validation data
DataLoader(val_ds, shuffle=True)             # harmless but makes results non-comparable

# 8. Fitting the vectorizer on all data
vectorizer.fit(all_texts)                    # leakage; fit on train only
```
:::

## Debugging

```python
# 1. Can it overfit 20 examples?
tiny = TensorDataset(X[:20], y[:20])
# train 200 epochs; loss must reach ~0. If not, the model or loss is wrong.

# 2. Are gradients flowing?
for name, parameter in model.named_parameters():
    if parameter.grad is None:
        print(f"{name}: NO GRADIENT")
    else:
        print(f"{name}: grad_norm={parameter.grad.norm():.6f}")
# all zeros → dead ReLUs or a detached graph; huge → clip and lower the learning rate

# 3. Shape errors
print(x.shape, logits.shape, y.shape)
# CrossEntropyLoss wants logits (batch, classes) and targets (batch,) as int64

# 4. Anomaly detection for NaN
torch.autograd.set_detect_anomaly(True)      # slow; use only while debugging
```

## Performance Considerations

| Technique | Effect |
| --- | --- |
| `batch_size` ↑ | better hardware use; watch memory and generalisation |
| `num_workers=4` | parallel data loading; the usual bottleneck on GPU training |
| `pin_memory=True` | faster host→device copies |
| mixed precision (`torch.autocast`) | ~2× speed and half the memory on modern GPUs |
| `torch.compile(model)` | 1.3–2× on PyTorch 2.x, one line |
| gradient accumulation | simulate a large batch on a small GPU |
| `set_to_none=True` in `zero_grad` | marginally faster, less memory |

```python
model = torch.compile(model)                        # one line, real speed-up

scaler = torch.amp.GradScaler("cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = criterion(model(xb), yb)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Hands-on Exercise

:::exercise Beat the baseline properly
The TF-IDF network barely beat logistic regression. Fix that by changing the *input
representation* rather than the architecture:

1. `uv add sentence-transformers` and embed the tickets with a small pre-trained model
   (`all-MiniLM-L6-v2`, 384 dimensions).
2. Train the same classifier head on those embeddings.
3. Compare macro-F1 against both TF-IDF models, and note training time and feature
   dimensionality.
4. Add a confusion matrix and identify the two most-confused classes.
5. Report the parameter count of each approach.

Expect the 384-dimensional embeddings to beat 5,000-dimensional TF-IDF with fewer
parameters. That is transfer learning, and it is the bridge to Phases 10–11.
:::

:::solution Expected result
```text
representation          features  params      macro_f1  train_s
tfidf + logistic           5,000   40,008       0.8106      3.2
tfidf + mlp                5,000  1,313,544     0.8189     42.1
minilm + logistic            384    3,080       0.8421      1.8
minilm + mlp                 384  116,552       0.8657     11.4

most confused: sci.space ↔ comp.graphics (rendering/simulation vocabulary overlaps)
```

Fewer features, 11× fewer parameters, better accuracy. The pre-trained encoder has already
learned what "similar meaning" means, which no amount of MLP capacity on top of TF-IDF can
recover. This is precisely why Phase 11 spends its time on embeddings.
:::

## Challenge

:::challenge Fine-tune a transformer encoder
Fine-tune `distilbert-base-uncased` on the same task with Hugging Face `transformers`:
tokenise with padding and truncation, use `AutoModelForSequenceClassification`, a learning
rate around 2e-5, 3 epochs, and linear warmup.

Compare macro-F1, training time and parameter count against all four approaches above, then
write a short recommendation: given 2,000 labelled tickets and a latency budget of 50 ms,
which would you actually ship? The answer is usually "the frozen-embedding classifier" — and
knowing *why* is the point of the exercise.
:::

## Interview Questions

:::interview
1. What does `loss.backward()` do?
2. Why must you call `optimizer.zero_grad()` every step?
3. What changes between `model.train()` and `model.eval()`?
4. Why pass raw logits to `CrossEntropyLoss` rather than softmax probabilities?
5. How do you train with a batch size larger than your GPU memory allows?
:::

## Cheat Sheet

```python
torch.tensor(data) torch.zeros(s) torch.randn(s) torch.as_tensor(np_array)
t.shape t.dtype t.device t.to(device) t.detach() t.item() t.cpu().numpy()
t.reshape() t.unsqueeze(0) t.squeeze() t.permute(1,0) a @ b a.sum(dim=0)

class Net(nn.Module):
    def __init__(self): super().__init__(); self.fc = nn.Linear(i, o)
    def forward(self, x): return self.fc(x)

nn.Linear Conv1d LayerNorm BatchNorm1d ReLU GELU Dropout Sequential Embedding
nn.CrossEntropyLoss(weight=..., label_smoothing=0.05)   # raw logits + int64 targets
nn.BCEWithLogitsLoss()  nn.MSELoss()
torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
lr_scheduler.ReduceLROnPlateau / CosineAnnealingLR / OneCycleLR

loop: model.train() → zero_grad(set_to_none=True) → forward → loss → backward
      → clip_grad_norm_ → step ;  model.eval() + torch.no_grad() for validation

torch.save({"model_state": model.state_dict(), ...}, path)
model.load_state_dict(torch.load(path, map_location=device)["model_state"])
torch.compile(model)   torch.autocast("cuda", dtype=torch.bfloat16)
```

```quiz
[
  {
    "question": "You forget optimizer.zero_grad(). What happens?",
    "options": [
      "An error is raised",
      "Gradients accumulate across batches, so each update uses the sum of all previous gradients",
      "The model trains faster",
      "Nothing; PyTorch clears them automatically"
    ],
    "answer": 1,
    "explanation": "Accumulation is deliberate (it enables gradient accumulation for large effective batches), so you must clear gradients yourself each step."
  },
  {
    "question": "Validation loss is unstable and worse than training loss by a lot, even early in training. What should you check first?",
    "options": [
      "The learning rate",
      "That model.eval() is called before validation, so dropout and batchnorm behave correctly",
      "The batch size",
      "The optimiser"
    ],
    "answer": 1,
    "explanation": "Leaving the model in training mode keeps dropout active during evaluation, injecting noise into every validation number."
  },
  {
    "question": "Your MLP on TF-IDF features barely beats logistic regression. What is the most promising change?",
    "options": [
      "Add more layers",
      "Change the input representation - use pre-trained embeddings instead of TF-IDF",
      "Train for more epochs",
      "Increase the hidden size"
    ],
    "answer": 1,
    "explanation": "Depth helps when the model must learn representations from raw input. On top of already-engineered features, extra capacity mostly adds overfitting; transfer learning from a pre-trained encoder is the real lever."
  }
]
```

## Summary

- PyTorch = NumPy + autograd + GPU; you write the forward pass and the framework does the
  rest.
- The loop is always: `zero_grad → forward → loss → backward → step`, with `train()`/`eval()`
  and `no_grad()` in the right places.
- Checkpoint on the best validation metric, stop early, and always compare against a linear
  baseline.
- Deep learning's advantage comes from learned representations — which is why pre-trained
  embeddings beat a bigger MLP on engineered features.

## Next Step

Phase 10: transformers and attention in detail, then production-grade LLM API engineering.

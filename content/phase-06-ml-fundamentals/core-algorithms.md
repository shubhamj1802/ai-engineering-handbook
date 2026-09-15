---
title: The Core Algorithms
order: 2
difficulty: Intermediate
duration: 18
badges: ["Hands-on", "Reference"]
summary: "Linear and logistic regression, decision trees, random forests, gradient boosting, KNN, SVM and k-means — how each works, what it assumes, and when to reach for it."
prereqs: ["What Machine Learning Actually Is"]
keyConcepts: ["linear regression", "logistic regression", "decision tree", "ensemble", "k-means"]
---

:::note In one line
**Start with the simple model.** Linear and tree-based models are fast, explainable, and beat a neural network on tabular data more often than people expect.
:::

## Why this matters

You need roughly seven algorithms to solve most tabular problems, and knowing *why* each one
fails is more useful than knowing twenty. The same intuitions carry into deep learning:
gradient descent, regularisation, and the bias–variance trade-off are the same ideas at a
different scale.

## Mental Model

```text
Is there a label?
├── no  → UNSUPERVISED
│         k-means (groups), PCA (compress), DBSCAN (density + outliers)
└── yes → SUPERVISED
          ├── number  → REGRESSION
          │     Linear/Ridge (interpretable, fast)  ·  Gradient boosting (accuracy)
          └── category → CLASSIFICATION
                Logistic regression (baseline, calibrated probabilities)
                Random forest / Gradient boosting (tabular default)
                KNN (tiny data, local structure)  ·  SVM (small, high-dimensional)
```

Default recipe for a tabular problem: **logistic/linear regression as the baseline, gradient
boosting as the contender.** If neither works, the problem is the features, not the
algorithm.

## Core Concepts

### Linear regression — the foundation

Fit a line (or hyperplane) that minimises squared error:

```text
ŷ = w₁x₁ + w₂x₂ + … + wₙxₙ + b
loss = mean((y - ŷ)²)                   ← minimise this by gradient descent
```

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso

model = LinearRegression().fit(X_train, y_train)
model.coef_          # one weight per feature - directly interpretable
model.intercept_

Ridge(alpha=1.0)     # L2: shrinks weights, handles correlated features
Lasso(alpha=0.1)     # L1: drives weights to exactly zero → feature selection
```

Assumes a roughly linear relationship and roughly constant error variance. Sensitive to
outliers (squared error punishes them heavily) and to feature scale when regularised —
always scale before Ridge/Lasso.

### Logistic regression — classification's baseline

Same linear combination, squashed into a probability:

```text
p = sigmoid(w·x + b) = 1 / (1 + e^-(w·x + b))
loss = log loss (cross-entropy)
```

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1_000,
    C=1.0,                      # inverse regularisation strength: smaller = more regularised
    class_weight="balanced",    # for imbalanced classes
).fit(X_train, y_train)

model.predict(X_test)               # hard labels at a 0.5 threshold
model.predict_proba(X_test)[:, 1]   # probabilities - usually what you actually want
```

It is not "just a baseline": it gives well-calibrated probabilities out of the box, trains in
milliseconds, and its coefficients are auditable — which matters when a decision affects a
customer and someone asks why.

### Decision trees

Recursively split on the feature/threshold that best separates the target.

```text
                 top_score < 0.55?
                 ├── yes → tokens > 3000?
                 │         ├── yes → ESCALATE (91% of 340 samples)
                 │         └── no  → ANSWER
                 └── no  → ANSWER (96% of 2,100 samples)
```

```python
from sklearn.tree import DecisionTreeClassifier, export_text

tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=42)
tree.fit(X_train, y_train)
print(export_text(tree, feature_names=list(X.columns)))     # readable rules
```

No scaling needed, handles non-linearity and interactions, fully interpretable — and
overfits wildly if unconstrained. A tree with no depth limit will memorise the training set
perfectly.

### Random forest — many decorrelated trees

Train hundreds of trees, each on a bootstrap sample of rows and a random subset of features,
then average.

```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",       # decorrelates the trees - the key idea
    n_jobs=-1,
    random_state=42,
).fit(X_train, y_train)

importances = pd.Series(forest.feature_importances_, index=X.columns).sort_values(ascending=False)
```

Strong out of the box, hard to overfit badly, parallel. Larger and slower at inference than a
linear model, and its built-in feature importances are biased toward high-cardinality
features — prefer permutation importance.

### Gradient boosting — the tabular champion

Trees built **sequentially**, each correcting the previous ensemble's errors.

```python
from sklearn.ensemble import HistGradientBoostingClassifier

model = HistGradientBoostingClassifier(
    max_iter=400,
    learning_rate=0.06,
    max_depth=None,
    max_leaf_nodes=31,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
).fit(X_train, y_train)
```

`HistGradientBoosting*` is scikit-learn's fast histogram implementation; it handles NaNs
natively and needs no scaling. XGBoost, LightGBM and CatBoost are the specialised
alternatives — all four win most tabular benchmarks.

The trade-off: sensitive to hyperparameters (especially `learning_rate` × `max_iter`), and
sequential, so slower to train than a forest.

### K-nearest neighbours

No training: predict by looking at the `k` closest training points.

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=15, weights="distance").fit(X_train, y_train)
```

**Scaling is mandatory** — distances are meaningless if one feature is in thousands and
another in fractions. Prediction is O(n) per query, so it does not scale. Conceptually
important though: KNN over embeddings *is* vector search (Phase 11).

### Support vector machines

Find the boundary with the widest margin; the kernel trick handles non-linearity.

```python
from sklearn.svm import SVC

model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True).fit(X_train_scaled, y_train)
```

Excellent on small, high-dimensional, clean data (a few thousand rows). Training is roughly
O(n²)–O(n³), so it is impractical beyond ~50k rows, and probabilities require an extra
calibration step.

### k-means — unsupervised grouping

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, n_init=10, random_state=42).fit(X_scaled)
kmeans.labels_          # cluster per row
kmeans.cluster_centers_
kmeans.inertia_         # within-cluster sum of squares - the elbow method uses this
```

Assumes roughly spherical, similar-sized clusters; you must pick `k`; scaling matters. Use
silhouette score to evaluate, and DBSCAN when clusters are irregular or you need outlier
detection.

### The comparison table

| Algorithm | Scale needed | Handles non-linear | Interpretable | Train speed | Predict speed | Good for |
| --- | --- | --- | --- | --- | --- | --- |
| Linear/Ridge | yes | no | high | very fast | very fast | baselines, extrapolation |
| Logistic | yes | no | high | very fast | very fast | calibrated probabilities |
| Decision tree | no | yes | very high | fast | very fast | explaining rules |
| Random forest | no | yes | medium | medium | medium | robust default |
| Gradient boosting | no | yes | medium | slow | fast | best tabular accuracy |
| KNN | **yes** | yes | low | none | slow | small data, similarity |
| SVM (rbf) | **yes** | yes | low | slow | medium | small, high-dimensional |
| k-means | **yes** | — | medium | fast | fast | segmentation |

## Minimal Example

```python title="compare_algorithms.py"
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=3_000, n_features=20, n_informative=8,
                           n_redundant=4, weights=[0.7, 0.3], random_state=42)

models = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1_000)),
    "tree (depth 4)": DecisionTreeClassifier(max_depth=4, random_state=42),
    "tree (unlimited)": DecisionTreeClassifier(random_state=42),
    "random forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "gradient boosting": HistGradientBoostingClassifier(random_state=42),
    "knn (k=15)": make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
}

rows = []
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
    rows.append({"model": name, "roc_auc": scores.mean().round(4), "std": scores.std().round(4)})

print(pd.DataFrame(rows).sort_values("roc_auc", ascending=False).to_string(index=False))
```

```text
            model  roc_auc     std
gradient boosting   0.9634  0.0073
    random forest   0.9581  0.0081
       knn (k=15)   0.9234  0.0101
         logistic   0.9126  0.0092
   tree (depth 4)   0.8917  0.0143
 tree (unlimited)   0.8104  0.0166
```

The unlimited tree is the worst model on the list — a clean demonstration that capacity
without constraint is not skill.

## Real-World Example

A model-selection harness with proper preprocessing, permutation importance and
interpretation — what you actually run when choosing a model for a real problem.

```python title="src/ml/model_selection.py"
"""Compare candidate models honestly and explain the winner.

Preprocessing lives inside each Pipeline, so cross-validation cannot leak, and
every model sees exactly the same folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Impute + scale numerics, impute + one-hot categoricals. Fitted per fold."""
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                                         sparse_output=False)),
            ]), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def candidate_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "baseline": Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
        "logistic": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=2_000, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=3, max_features="sqrt",
                class_weight="balanced_subsample", n_jobs=-1, random_state=42)),
        ]),
        "gradient_boosting": Pipeline([
            ("prep", preprocessor),
            ("clf", HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.06, early_stopping=True,
                validation_fraction=0.1, random_state=42)),
        ]),
    }


def evaluate_all(
    X: pd.DataFrame, y: pd.Series, models: dict[str, Pipeline], *, folds: int = 5
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    rows = []

    for name, model in models.items():
        scores = cross_validate(
            model, X, y, cv=cv,
            scoring=["roc_auc", "average_precision", "f1", "precision", "recall"],
            n_jobs=-1, return_train_score=True, error_score="raise",
        )
        rows.append({
            "model": name,
            "roc_auc": scores["test_roc_auc"].mean(),
            "pr_auc": scores["test_average_precision"].mean(),
            "f1": scores["test_f1"].mean(),
            "precision": scores["test_precision"].mean(),
            "recall": scores["test_recall"].mean(),
            "overfit_gap": scores["train_f1"].mean() - scores["test_f1"].mean(),
            "fit_seconds": scores["fit_time"].mean(),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("pr_auc", ascending=False)
        .round(4)
        .reset_index(drop=True)
    )


def explain(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, *, top: int = 10) -> pd.DataFrame:
    """Permutation importance: shuffle a column, measure the damage.

    Unlike tree feature_importances_, this is model-agnostic, computed on held-out
    data, and not biased toward high-cardinality features.
    """
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc", n_jobs=-1
    )
    return (
        pd.DataFrame({
            "feature": X_test.columns,
            "importance": result.importances_mean,
            "std": result.importances_std,
        })
        .sort_values("importance", ascending=False)
        .head(top)
        .round(4)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 6_000

    df = pd.DataFrame({
        "tokens": rng.integers(300, 6_000, n),
        "top_score": rng.beta(5, 2, n),
        "n_chunks": rng.integers(1, 12, n),
        "question_length": rng.integers(20, 600, n),
        "prior_escalations": rng.poisson(0.4, n),
        "hour": rng.integers(0, 24, n),
        "channel": rng.choice(["slack", "web", "email"], n, p=[0.5, 0.35, 0.15]),
        "plan": rng.choice(["free", "pro", "enterprise"], n, p=[0.5, 0.35, 0.15]),
    })
    df.loc[rng.choice(n, 400, replace=False), "top_score"] = np.nan       # realistic gaps

    logit = (
        -2.4
        + 3.0 * (1 - df["top_score"].fillna(0.5))
        + 0.0002 * df["tokens"]
        + 0.5 * df["prior_escalations"]
        + 0.7 * (df["plan"] == "enterprise")
        + 0.4 * df["hour"].isin([2, 3, 4]).astype(float)
        + rng.normal(0, 0.55, n)
    )
    df["escalated"] = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)

    numeric = ["tokens", "top_score", "n_chunks", "question_length", "prior_escalations", "hour"]
    categorical = ["channel", "plan"]
    X, y = df[numeric + categorical], df["escalated"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = candidate_models(build_preprocessor(numeric, categorical))
    print(f"positive class rate: {y.mean():.1%}\n")
    print(evaluate_all(X_train, y_train, models).to_string(index=False))

    best = models["gradient_boosting"].fit(X_train, y_train)
    print("\npermutation importance (held-out):")
    print(explain(best, X_test, y_test).to_string(index=False))
```

```text
positive class rate: 31.4%

            model  roc_auc  pr_auc      f1  precision  recall  overfit_gap  fit_seconds
gradient_boosting   0.9021  0.8104  0.7391     0.7628  0.7169       0.0918       0.9412
    random_forest   0.8934  0.7965  0.7288     0.6903  0.7719       0.2641       1.1038
         logistic   0.8876  0.7842  0.7201     0.6684  0.7806       0.0104       0.0521
         baseline   0.5000  0.3140  0.0000     0.0000  0.0000       0.0000       0.0033

permutation importance (held-out):
          feature  importance     std
        top_score      0.2418  0.0104
           tokens      0.0421  0.0038
prior_escalations      0.0316  0.0029
             plan      0.0104  0.0018
             hour      0.0061  0.0014
```

Gradient boosting wins, but only by 0.03 PR-AUC over logistic regression, which trains 18×
faster and is fully auditable. On a real project that trade-off often favours the simpler
model — and permutation importance confirms the model learned the true drivers rather than
an artefact.

## Common Mistakes

:::mistake
```python
# 1. Not scaling for distance/regularised models
KNeighborsClassifier().fit(X_raw, y)       # a feature in thousands dominates every distance
SVC().fit(X_raw, y)                         # same
# trees and boosting do NOT need scaling

# 2. An unconstrained decision tree
DecisionTreeClassifier()                    # memorises the training set
DecisionTreeClassifier(max_depth=6, min_samples_leaf=20)

# 3. Trusting tree feature_importances_
# biased toward high-cardinality features; use permutation_importance on held-out data

# 4. Ignoring class_weight on imbalanced data
LogisticRegression()                        # learns to predict the majority
LogisticRegression(class_weight="balanced")

# 5. predict() when you need predict_proba()
# thresholding at 0.5 is a business decision, not a default

# 6. Starting with a complicated model
# fit logistic regression first: it takes 50ms and tells you whether the signal exists
```
:::

## Debugging

```python
# Is the model learning anything at all?
from sklearn.dummy import DummyClassifier          # compare against this first

# Where does it fail?
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, predictions, digits=3))

# Is it confident when wrong?
probabilities = model.predict_proba(X_test)[:, 1]
wrong = (predictions != y_test)
print("mean confidence when wrong:", probabilities[wrong].round(2).mean())

# Are the probabilities meaningful?
from sklearn.calibration import calibration_curve
true_rate, predicted_rate = calibration_curve(y_test, probabilities, n_bins=10)
```

Calibration matters in AI engineering: if you route on "confidence > 0.8", those numbers had
better mean something. Tree ensembles are typically over-confident; wrap them in
`CalibratedClassifierCV` when the probability itself drives a decision.

## Performance Considerations

| Model | 10k rows | 1M rows | Notes |
| --- | --- | --- | --- |
| Logistic | ms | seconds | scales to millions with SGD |
| Random forest | seconds | minutes | parallel, memory-hungry |
| HistGradientBoosting | seconds | minutes | histogram binning; the practical default |
| SVM (rbf) | seconds | hours | O(n²)+ — impractical at scale |
| KNN | instant to fit | slow per query | no training, expensive inference |

For inference latency: linear models are microseconds, tree ensembles are tens to hundreds
of microseconds. Both are negligible next to a single LLM call, which is worth remembering
when you are deciding whether to use a model or a prompt.

## Hands-on Exercise

:::exercise Choose a model with evidence
Using a dataset of your own or `sklearn.datasets.fetch_openml("credit-g")`:

1. Establish the baseline with `DummyClassifier`.
2. Build a `ColumnTransformer` handling numeric and categorical columns.
3. Cross-validate five candidates: logistic, tree (depth 4), random forest, gradient
   boosting, KNN.
4. Report ROC-AUC, PR-AUC, F1 and the train/test gap for each.
5. Produce permutation importance for the best model.
6. Write a three-sentence recommendation that names the model, the metric it wins on, and
   the trade-off you are accepting.

The write-up is the deliverable — a table of numbers without a recommendation is not a
decision.
:::

:::solution Reference recommendation
```text
model            roc_auc  pr_auc      f1  overfit_gap  fit_s
gradient_boosting 0.7912  0.6428  0.6104       0.1832   0.71
random_forest     0.7843  0.6312  0.5991       0.3104   0.88
logistic          0.7791  0.6201  0.6018       0.0211   0.04
tree_depth_4      0.7102  0.5488  0.5402       0.0604   0.02
knn_k15           0.6854  0.5011  0.4887       0.1442   0.02

Recommendation: ship logistic regression. Gradient boosting leads by 0.02 PR-AUC,
which is inside the fold-to-fold variation, while logistic trains 18x faster, has a
train/test gap of 0.02 versus 0.18, and produces calibrated probabilities we can
threshold for the escalation decision and explain to a customer. If PR-AUC becomes
the binding constraint later, gradient boosting is the upgrade path - but we should
spend the next week on features, not on models, because the top three are within
0.01 of each other.
```

That last sentence is the most valuable one: when several algorithms cluster together, the
ceiling is set by the features, and more model-tuning is wasted effort.
:::

## Challenge

:::challenge Implement gradient boosting from scratch
Write a small gradient booster for regression: start with the mean as the prediction, then
for `n_estimators` rounds fit a shallow `DecisionTreeRegressor` to the residuals and add
`learning_rate × tree.predict(X)` to the running prediction.

Compare its RMSE to `sklearn.ensemble.GradientBoostingRegressor` on the same data, and plot
training RMSE against the number of rounds for three learning rates. Seeing the residuals
shrink round by round makes boosting permanently intuitive — and the learning-rate/rounds
trade-off you observe is exactly the one you will tune in every real project.
:::

## Interview Questions

:::interview
1. When would you choose logistic regression over gradient boosting?
2. How does a random forest reduce the variance of a single tree?
3. What does the learning rate do in gradient boosting, and how does it interact with the
   number of estimators?
4. Which algorithms require feature scaling and why?
5. What is the difference between L1 and L2 regularisation?
:::

## Cheat Sheet

```python
# regression
LinearRegression() Ridge(alpha=1.0) Lasso(alpha=0.1) HistGradientBoostingRegressor()
# classification
LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
DecisionTreeClassifier(max_depth=6, min_samples_leaf=20)
RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=-1)
HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, early_stopping=True)
KNeighborsClassifier(15, weights="distance")        # scale first
SVC(kernel="rbf", C=1.0, probability=True)           # scale first, small data
# unsupervised
KMeans(n_clusters=5, n_init=10)  DBSCAN(eps=0.5, min_samples=5)  PCA(n_components=0.95)

model.predict_proba(X)[:, 1]                          # probabilities, not labels
permutation_importance(model, X_test, y_test, n_repeats=10, scoring="roc_auc")
CalibratedClassifierCV(model, method="isotonic", cv=5)
```

```quiz
[
  {
    "question": "Which model needs feature scaling?",
    "options": ["Random forest", "Gradient boosting", "K-nearest neighbours", "Decision tree"],
    "answer": 2,
    "explanation": "KNN (and SVM, and regularised linear models) compare distances, so a feature measured in thousands would dominate one measured in fractions. Tree-based models split on thresholds and are scale-invariant."
  },
  {
    "question": "A single decision tree scores 1.00 on train and 0.68 on test. What is the fix?",
    "options": [
      "More trees will not help a single tree; constrain it with max_depth/min_samples_leaf, or use an ensemble",
      "Increase the learning rate",
      "Scale the features",
      "Use a smaller test set"
    ],
    "answer": 0,
    "explanation": "An unconstrained tree memorises. Limiting depth and leaf size, or averaging many decorrelated trees (random forest), reduces the variance."
  },
  {
    "question": "You need to explain each decision to a regulator. Which do you choose?",
    "options": [
      "Random forest with 500 trees",
      "Logistic regression with scaled, named features",
      "An RBF-kernel SVM",
      "Gradient boosting with 1000 rounds"
    ],
    "answer": 1,
    "explanation": "Logistic regression gives one signed, inspectable coefficient per feature and calibrated probabilities - auditable by construction rather than through post-hoc approximations."
  }
]
```

## Summary

- Logistic/linear regression is the baseline that tells you whether signal exists.
- Trees capture non-linearity and interactions but overfit unless constrained.
- Random forests reduce variance by averaging decorrelated trees; boosting reduces bias by
  correcting errors sequentially and usually wins on tabular data.
- KNN and SVM need scaling and do not scale to large datasets.
- Choose with evidence: baseline, cross-validation, permutation importance, and an explicit
  trade-off statement.

## Next Step

Evaluation done properly — the metrics that reveal what accuracy hides, plus pipelines,
cross-validation and hyperparameter tuning.

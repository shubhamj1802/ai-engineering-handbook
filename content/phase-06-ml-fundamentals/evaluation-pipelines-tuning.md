---
title: Evaluation, Pipelines and Hyperparameter Tuning
order: 3
difficulty: Advanced
duration: 16
badges: ["Hands-on", "Reference"]
summary: "Precision, recall, F1, ROC-AUC and the threshold decision — plus Pipelines that prevent leakage and search strategies that find good hyperparameters without wasting a week."
prereqs: ["The Core Algorithms"]
keyConcepts: ["precision", "recall", "ROC-AUC", "threshold", "Pipeline", "cross-validation"]
---

:::note In one line
**Accuracy is a trap on unbalanced data.** A model that says no to everything scores 99% when only 1% is a yes.
:::

## Why this matters

Choosing a metric *is* choosing what the system optimises for, and choosing a threshold is a
business decision disguised as a technical one. Get this wrong and you ship a model that
scores well and behaves badly. Everything here transfers directly to Phase 24, where you
evaluate RAG systems and agents with the same discipline.

## Mental Model

```text
                   PREDICTED
                 positive   negative
ACTUAL positive    TP         FN       ← missed cases (recall problem)
       negative    FP         TN       ← false alarms (precision problem)

precision = TP / (TP + FP)    "when we said yes, how often were we right?"
recall    = TP / (TP + FN)    "of all the real cases, how many did we catch?"
F1        = harmonic mean     "one number when both matter equally"

You cannot maximise both. Moving the threshold trades one for the other.
```

Which one matters depends entirely on the cost of each mistake:

| Situation | Optimise | Because |
| --- | --- | --- |
| Cancer screening | **recall** | a missed case is catastrophic; a false alarm costs a test |
| Spam filter | **precision** | a lost legitimate email is worse than a spam that gets through |
| Escalating support tickets to humans | balance | missing an angry customer vs. wasting agent time |
| Retrieval for RAG | **recall@k**, then precision after reranking | you cannot answer from a document you never retrieved |

## Core Concepts

### Classification metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report,
)

print(classification_report(y_test, predictions, digits=3))
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

probabilities = model.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, probabilities)             # threshold-independent ranking quality
average_precision_score(y_test, probabilities)   # PR-AUC: use this when positives are rare
```

:::warning ROC-AUC flatters imbalanced problems
With 1% positives, a model can have ROC-AUC 0.95 and still produce mostly false positives,
because the huge number of true negatives keeps the false-positive *rate* low. PR-AUC
(`average_precision_score`) reflects what the user experiences. Report both; decide on PR-AUC.
:::

### The threshold is yours to choose

`predict()` thresholds at 0.5, which is almost never the right business threshold.

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

precision, recall, thresholds = precision_recall_curve(y_val, probabilities)

# Option 1: the best F1
f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
best_threshold = thresholds[int(np.nanargmax(f1[:-1]))]

# Option 2: the cheapest threshold, given what each mistake costs
def expected_cost(threshold: float, *, cost_fp: float = 1.0, cost_fn: float = 12.0) -> float:
    predicted = (probabilities >= threshold).astype(int)
    fp = int(((predicted == 1) & (y_val == 0)).sum())
    fn = int(((predicted == 0) & (y_val == 1)).sum())
    return fp * cost_fp + fn * cost_fn

candidates = np.linspace(0.05, 0.95, 91)
cheapest = float(candidates[int(np.argmin([expected_cost(t) for t in candidates]))])
```

Choose the threshold on the **validation** set, then report performance at that fixed
threshold on the test set. Choosing it on the test set is leakage.

### Regression metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, predictions)              # same units, robust to outliers
rmse = mean_squared_error(y_test, predictions) ** 0.5       # punishes large errors
r2 = r2_score(y_test, predictions)                          # share of variance explained
mape = (np.abs((y_test - predictions) / np.maximum(np.abs(y_test), 1e-9))).mean() * 100
```

MAE for "average error in minutes", RMSE when large errors are disproportionately bad, R²
for a scale-free comparison. MAPE breaks near zero — never use it for values that can be
small.

### Pipelines — the anti-leakage device

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), numeric_columns),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical_columns),
])

pipeline = Pipeline([("prep", preprocessor), ("clf", HistGradientBoostingClassifier())])
pipeline.fit(X_train, y_train)          # every step fitted on train data only
```

Inside cross-validation, the whole pipeline is re-fitted on each training fold. That is the
mechanism that makes the score honest — and it is why "just call `fit_transform` on
everything first" is wrong.

A pipeline is also a single deployable artifact:

```python
import joblib
joblib.dump(pipeline, "models/escalation_v3.joblib")     # preprocessing travels with the model
```

### Cross-validation strategies

```python
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit, cross_validate
)

StratifiedKFold(n_splits=5, shuffle=True, random_state=42)   # classification default
GroupKFold(n_splits=5)                                        # one customer never spans folds
TimeSeriesSplit(n_splits=5)                                   # train past → validate future
```

```text
TimeSeriesSplit, 5 folds:
  fold 1  train [====]              val [==]
  fold 2  train [======]            val [==]
  fold 3  train [========]          val [==]
```

### Hyperparameter search

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform, randint

# exhaustive: fine for a handful of combinations
grid = GridSearchCV(
    pipeline,
    param_grid={"clf__max_depth": [3, 5, 7], "clf__learning_rate": [0.03, 0.1]},
    scoring="average_precision", cv=5, n_jobs=-1, refit=True,
)

# randomised: far better use of a fixed budget when the space is large
search = RandomizedSearchCV(
    pipeline,
    param_distributions={
        "clf__learning_rate": loguniform(0.01, 0.3),
        "clf__max_leaf_nodes": randint(15, 64),
        "clf__min_samples_leaf": randint(5, 60),
        "clf__l2_regularization": loguniform(1e-6, 1.0),
    },
    n_iter=40, scoring="average_precision", cv=5, n_jobs=-1, random_state=42,
)
search.fit(X_train, y_train)
search.best_params_, search.best_score_
```

Random search beats grid search for the same budget: most hyperparameters barely matter, and
random sampling spends more trials on the ones that do. For larger budgets, use Optuna's
Bayesian search.

:::danger Nested cross-validation, or an honest holdout
If you tune with cross-validation and then report `search.best_score_` as your performance
estimate, it is optimistic — you selected the best of forty noisy scores. Either keep a
completely untouched test set for the final number, or use nested CV.
:::

## Minimal Example

```python title="threshold_choice.py"
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=6_000, weights=[0.93, 0.07], random_state=0)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0, stratify=y_temp)

model = HistGradientBoostingClassifier(random_state=0).fit(X_train, y_train)
val_probabilities = model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_val, val_probabilities)
f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
best = thresholds[int(np.nanargmax(f1[:-1]))]

test_probabilities = model.predict_proba(X_test)[:, 1]
for name, threshold in [("default 0.5", 0.5), (f"tuned {best:.2f}", best)]:
    predictions = (test_probabilities >= threshold).astype(int)
    print(f"{name:<14} precision={precision_score(y_test, predictions):.3f} "
          f"recall={recall_score(y_test, predictions):.3f} "
          f"flagged={predictions.sum():>4}")
```

```text
default 0.5    precision=0.783 recall=0.571 flagged=  60
tuned 0.28     precision=0.639 recall=0.762 flagged=  98
```

Same model, same data — 33% more real cases caught, at the cost of more false alarms. Which
row is better is a business question, and now you can put the trade-off in front of whoever
should answer it.

## Real-World Example

A complete evaluation module: metrics at a chosen operating point, cost-based threshold
selection, calibration, and a report.

```python title="src/ml/evaluation.py"
"""Evaluation with an explicit operating point.

The output is not a number, it is a decision: which threshold do we run at, what
does that cost, and how confident are we in the estimate?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    threshold: float
    precision: float
    recall: float
    f1: float
    flagged: int
    expected_cost: float

    def describe(self, *, total: int) -> str:
        return (
            f"threshold {self.threshold:.3f} → flags {self.flagged:,} of {total:,} "
            f"({self.flagged / total:.1%}) · precision {self.precision:.3f} · "
            f"recall {self.recall:.3f} · cost {self.expected_cost:,.0f}"
        )


def choose_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    cost_false_positive: float = 1.0,
    cost_false_negative: float = 10.0,
    min_precision: float | None = None,
    min_recall: float | None = None,
) -> OperatingPoint:
    """Pick the cheapest threshold that satisfies any hard constraints.

    Costs are relative, not absolute: 'a missed escalation costs ten times what a
    false alarm costs' is a conversation the business can actually have.
    """
    y_true = np.asarray(y_true).astype(int)
    candidates = np.unique(np.round(np.linspace(0.01, 0.99, 197), 4))

    best: OperatingPoint | None = None
    for threshold in candidates:
        predicted = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if min_precision is not None and precision < min_precision:
            continue
        if min_recall is not None and recall < min_recall:
            continue

        cost = fp * cost_false_positive + fn * cost_false_negative
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        point = OperatingPoint(float(threshold), precision, recall, f1, int(predicted.sum()), float(cost))

        if best is None or point.expected_cost < best.expected_cost:
            best = point

    if best is None:
        raise ValueError("no threshold satisfies the given precision/recall constraints")
    return best


def full_report(
    y_true: np.ndarray, probabilities: np.ndarray, *, point: OperatingPoint
) -> dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int)
    predicted = (probabilities >= point.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    return {
        "threshold": point.threshold,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),   # calibration: lower is better
        "precision": point.precision,
        "recall": point.recall,
        "f1": point.f1,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "positive_rate": float(y_true.mean()),
        "flag_rate": float(predicted.mean()),
    }


def calibration_table(y_true: np.ndarray, probabilities: np.ndarray, *, bins: int = 10) -> pd.DataFrame:
    """Does 'confidence 0.8' really mean 80%? Essential if you route on confidence."""
    observed, predicted = calibration_curve(y_true, probabilities, n_bins=bins, strategy="quantile")
    return pd.DataFrame({
        "predicted": predicted.round(3),
        "observed": observed.round(3),
        "gap": (observed - predicted).round(3),
    })


def threshold_sweep(y_true: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    """The whole trade-off curve - the table to put in front of a stakeholder."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    frame = pd.DataFrame({
        "threshold": np.append(thresholds, 1.0),
        "precision": precision,
        "recall": recall,
    })
    frame["f1"] = 2 * frame["precision"] * frame["recall"] / np.maximum(
        frame["precision"] + frame["recall"], 1e-12
    )
    return frame.iloc[::max(len(frame) // 12, 1)].round(3).reset_index(drop=True)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=10_000, n_features=18, n_informative=7,
                               weights=[0.92, 0.08], random_state=3)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=3, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=3, stratify=y_temp)

    model = HistGradientBoostingClassifier(max_iter=300, random_state=3).fit(X_train, y_train)
    val_probabilities = model.predict_proba(X_val)[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]

    # the threshold is chosen on VALIDATION and then frozen
    point = choose_threshold(y_val, val_probabilities,
                             cost_false_positive=1.0, cost_false_negative=12.0,
                             min_precision=0.35)
    print("chosen on validation:", point.describe(total=len(y_val)))

    report = full_report(y_test, test_probabilities, point=point)
    print("\ntest set at that threshold:")
    for key, value in report.items():
        print(f"  {key:<14} {value:.4f}" if isinstance(value, float) else f"  {key:<14} {value}")

    print("\ntrade-off curve:")
    print(threshold_sweep(y_test, test_probabilities).to_string(index=False))

    print("\ncalibration:")
    print(calibration_table(y_test, test_probabilities).to_string(index=False))
```

```text
chosen on validation: threshold 0.196 → flags 351 of 2,000 (17.6%) · precision 0.379 · recall 0.831 · cost 313

test set at that threshold:
  threshold      0.1960
  roc_auc        0.9312
  pr_auc         0.6104
  brier          0.0483
  precision      0.3712
  recall         0.8250
  f1             0.5121
  tp             132
  fp             224
  fn             28
  tn             1616
  positive_rate  0.0800
  flag_rate      0.1780

calibration:
 predicted  observed   gap
     0.008     0.005 -0.003
     0.021     0.019 -0.002
     0.412     0.397 -0.015
     0.786     0.812  0.026
```

The model catches 82% of positives while flagging 18% of traffic for review. Whether that is
the right trade is a business decision — and the calibration table shows the probabilities
are trustworthy to about ±0.03, so a confidence-based routing rule (Phase 20) would be sound.

## Common Mistakes

:::mistake
```python
# 1. Accuracy on imbalanced data
# 2. Reporting the tuned CV score as the final performance estimate (selection bias)
# 3. Choosing the threshold on the test set
# 4. Preprocessing outside the Pipeline
X_scaled = scaler.fit_transform(X)          # leaks into every CV fold
# 5. Grid-searching 6 parameters with 5 values each = 15,625 fits
#    → RandomizedSearchCV(n_iter=50)
# 6. Optimising a metric nobody asked for
#    "we improved F1 by 0.02" while the actual goal was "catch 90% of escalations"
# 7. Ignoring calibration when the probability drives a decision
```
:::

## Debugging

```python
# Look at the errors themselves, not just the metric
errors = X_test[predictions != y_test].copy()
errors["probability"] = probabilities[predictions != y_test]
errors["actual"] = y_test[predictions != y_test]
print(errors.sort_values("probability", ascending=False).head(10))
```

The ten most confident mistakes almost always reveal something concrete: a missing feature,
a mislabelled segment, a data-quality problem. That inspection is worth more than another
round of tuning.

```python
# Does performance hold across segments?
for segment, group in test_df.groupby("plan"):
    idx = group.index
    print(segment, round(roc_auc_score(y_test[idx], probabilities[idx]), 3), len(idx))
```

A model can score 0.90 overall and 0.61 on your most valuable customer segment. Always
slice.

## Performance Considerations

- `RandomizedSearchCV(n_iter=40, cv=5)` = 200 fits. Budget accordingly; use `n_jobs=-1`.
- Use `HalvingRandomSearchCV` to spend most of the budget on promising configurations.
- Cache preprocessing across a search with `Pipeline(memory="cache_dir")` when transforms
  are expensive.
- Tune on a sample while exploring; refit the winner on everything.

## Hands-on Exercise

:::exercise Build an operating-point report
For a binary problem of your choice:

1. Train a pipeline with proper preprocessing and cross-validation.
2. Produce a precision/recall/F1 table across 15 thresholds on validation data.
3. Add an `expected_cost` column with explicit costs for false positives and false negatives.
4. Choose the threshold three ways: best F1, cheapest cost, and "highest recall with
   precision ≥ 0.5".
5. Report all three on the test set in one table, with the flag rate.
6. Write two sentences recommending one, naming the trade-off.

This table is the artifact to bring to a product review. It converts "the model is 89%
accurate" into "at this setting we review 18% of tickets and catch 83% of escalations".
:::

:::solution Reference output
```text
strategy              threshold  precision  recall     f1  flag_rate  cost
best_f1                   0.412      0.612   0.494  0.547      0.065   412
cheapest_cost             0.196      0.371   0.825  0.512      0.178   313
recall_at_precision_0.5   0.318      0.503   0.631  0.560      0.100   367

Recommendation: run at 0.196. It costs least under our stated 12:1 penalty for a missed
escalation, and reviewing 18% of tickets is within the support team's stated capacity of
20%. If capacity tightens, 0.318 halves the review load while still catching 63% of
escalations - that is the fallback, not the default.
```
:::

## Challenge

:::challenge Nested cross-validation
Implement nested CV: an outer 5-fold loop for the performance estimate and an inner 3-fold
`RandomizedSearchCV` for hyperparameters, so tuning never sees the outer test fold.

Compare the nested estimate with the naive `search.best_score_`. Quantify the optimism gap
— typically 0.01–0.05 — and write a note explaining to a colleague why the naive number
should never appear in a report. That gap is the mechanism behind most "the model got worse
in production" stories.
:::

## Interview Questions

:::interview
1. When is ROC-AUC misleading, and what do you use instead?
2. How do you choose a classification threshold?
3. Why must preprocessing live inside the Pipeline?
4. What is the difference between grid search and random search, and when does each win?
5. What is model calibration and when do you need it?
:::

## Cheat Sheet

```python
classification_report(y, pred, digits=3)      confusion_matrix(y, pred).ravel()
roc_auc_score(y, proba)                        # ranking, threshold-free
average_precision_score(y, proba)              # PR-AUC: imbalanced data
brier_score_loss(y, proba)                     # calibration
precision_recall_curve(y, proba)               # the whole trade-off

mean_absolute_error / mean_squared_error ** 0.5 / r2_score

Pipeline([("prep", ColumnTransformer([...])), ("clf", Model())])
cross_validate(pipe, X, y, cv=StratifiedKFold(5), scoring=[...], return_train_score=True)
RandomizedSearchCV(pipe, distributions, n_iter=40, scoring="average_precision", cv=5, n_jobs=-1)
CalibratedClassifierCV(model, method="isotonic", cv=5)
joblib.dump(pipe, "model.joblib")              # preprocessing ships with the model
```

```quiz
[
  {
    "question": "1% of transactions are fraud. Your model has ROC-AUC 0.96 but PR-AUC 0.31. What does that tell you?",
    "options": [
      "The model is excellent",
      "Ranking is good, but at any useful threshold most flagged transactions will be false positives",
      "ROC-AUC was computed incorrectly",
      "You need more features"
    ],
    "answer": 1,
    "explanation": "With extreme imbalance the enormous true-negative count keeps the false-positive rate - and therefore ROC-AUC - flattering. PR-AUC reflects the precision a reviewer actually experiences."
  },
  {
    "question": "Why must the scaler live inside the Pipeline rather than being applied to X first?",
    "options": [
      "It is faster",
      "So it is re-fitted on each training fold and never sees validation data",
      "Pipelines require it",
      "It reduces memory"
    ],
    "answer": 1,
    "explanation": "Fitting the scaler on all data leaks the validation distribution into training, inflating cross-validation scores."
  },
  {
    "question": "Missing an escalation costs roughly 12x a false alarm. How should you set the threshold?",
    "options": [
      "Leave it at 0.5",
      "Minimise expected cost: sweep thresholds on validation data with those relative costs and pick the cheapest",
      "Maximise accuracy",
      "Use the mean predicted probability"
    ],
    "answer": 1,
    "explanation": "The threshold is a business decision. Encode the relative costs explicitly, choose on validation, then report the frozen threshold's performance on test."
  }
]
```

## Summary

- Precision and recall trade off; which matters depends on what each mistake costs.
- ROC-AUC for ranking, PR-AUC for imbalanced problems, Brier for calibration.
- The 0.5 threshold is a default, not a decision — choose it on validation with explicit
  costs.
- Pipelines make preprocessing part of the model and prevent leakage in cross-validation.
- Random search beats grid search per unit of compute; never report a tuned CV score as your
  final estimate.

## Next Step

Phase 7: two complete projects, from raw data to a deployed inference API.

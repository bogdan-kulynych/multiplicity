## multiplicity

[![CI](https://github.com/bogdan-kulynych/multiplicity/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bogdan-kulynych/multiplicity/actions/workflows/ci.yml)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Library for evaluating [predictive multiplicity](https://arxiv.org/abs/1909.06677) of deep leearning models.

### Setup

```
pip install multiplicity
```

### Quickstart

The library provides a method to estimate a [viable prediction range](https://arxiv.org/abs/2206.01131) ---the minimum and maximum possible predictions--- within the Rashomon set ---a set of models that have epsilon-similar loss on some reference dataset.

```
import multiplicity

# Train binary classifier with torch.
x = ...
train_loader = ...
model = ...
model(x)  # e.g., 0.75

# Specify how similar is the loss for models in the Rashomon set.
epsilon = 0.01

# Specify the loss function that defines the Rashomon set.
stopping_criterion = multiplicity.ZeroOneLossStoppingCriterion(train_loader)

# Compute viable prediction range.
lb, pred, ub = multiplicity.viable_prediction_range(
    model=model,
    target_example=x,
    stopping_criterion=stopping_criterion,
    criterion_thresholds=epsilon,
)
# e.g., lb=0.71, pred=0.75, ub=0.88
```

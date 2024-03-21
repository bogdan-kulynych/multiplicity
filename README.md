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

The library provides a method to estimate [viable prediction intervals](https://arxiv.org/abs/2206.01131): prediction intervals that are robust to a small change in model's loss at training or evaluation time.

Import the library:

```
from multiplicity import torch as multiplicity
```

Suppose we have a trained torch binary classifier which outputs softmax probabilities:
```
model(x)  # 0.75
```

Specify to the deviation of which metric we want to be robust to, and on which dataset:
```
robustness_criterion = multiplicity.ZeroOneLossCriterion(train_loader)
```

Then, we can compute the viable prediction range for a given example x like so:
```
lb, pred, ub = multiplicity.viable_prediction_range(
    model=model,
    target_example=x,
    robustness_criterion=robustness_criterion,
    criterion_thresholds=epsilon,
)
# lb=0.71, pred=0.75, ub=0.88
```

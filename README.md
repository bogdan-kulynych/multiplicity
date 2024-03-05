## multiplicity

Library for evaluating [predictive multiplicity](https://arxiv.org/abs/1909.06677) of deep leearning models.

### Setup

```
pip install multiplicity
```

### Quickstart

The library provides a method to estimate a [viable prediction range](https://arxiv.org/abs/2206.01131) ---the minimum and maximum possible predictions--- within the Rashomon set ---a set of models that have epsilon-similar loss on some reference dataset.

```
import multiplicity

# Train binary classifier in torch.
x = ...
train_loader = ...
model = ...
model(x)  # e.g., 0.75

# Specify how similar is the loss for models in the Rashomon set.
epsilon = 0.01

# Specify the loss function that defines the Rashomon set.
stopping_criterion = multiplicity.ZeroOneLossStoppingCriterion(train_loader)

# Compute viable prediction range.
lb, pred, ub = multiplicity.binary_viable_prediction_range(
    model=model,
    target_example=x,
    stopping_criterion=stopping_criterion,
    criterion_thresholds=epsilon,
)
# e.g., lb=0.71, pred=0.75, ub=0.88
```

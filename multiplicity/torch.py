import copy
from typing import Callable, Iterable, Optional, Union

import torch
import numpy as np


score_function_map = {
    "output": lambda model, x: model(x).squeeze(),
    "sigmoid": lambda model, x: torch.sigmoid(model(x)).squeeze(),
}


predict_function_map = {
    "output": score_function_map["output"],
    "sigmoid": score_function_map["sigmoid"],
}


def get_func_by_name(request: Union[str, Callable], func_map: dict, func_type: str):
    if isinstance(request, str):
        try:
            return func_map[request]
        except KeyError:
            raise ValueError(f"Unknown {func_type}: {request}")
    else:
        return request


class ZeroOneErrorCriterion:
    """
    Robustness criterion defined by zero-one error.

    Params:
        ref_loader: Dataloader of the dataset on which the criterion will be computed.
        decision_threshold: Threshold of the model prediction at which the decision is positive.
        predict_func: Function to obtain a prediction from a given model and an example
    """

    def __init__(
        self,
        ref_loader: torch.utils.data.DataLoader,
        decision_threshold: float = 0.5,
        predict_func: Union[str, Callable] = "output",
    ):
        self.ref_loader = ref_loader
        self.decision_threshold = decision_threshold
        self.predict_func = get_func_by_name(
            predict_func, predict_function_map, "predict_func"
        )

    def __call__(self, model: torch.nn.Module):
        num_correct = 0
        num_total = 0
        with torch.no_grad():
            for inputs, labels in self.ref_loader:
                preds = self.predict_func(model, inputs) >= self.decision_threshold
                num_correct += float(torch.sum(preds != labels).cpu())
                num_total += len(preds)

        return num_correct / num_total


class LossCriterion:
    """
    Robustness criterion defined by an arbitrary torch loss function.

    Params:
        ref_loader: Dataloader of the dataset on which the criterion will be computed.
        loss_func: The loss function
        predict_func: Function to obtain a prediction from a given model and an example
    """

    def __init__(
        self,
        ref_loader: torch.utils.data.DataLoader,
        loss_func: Callable,
        predict_func: Union[str, Callable] = "output",
    ):
        self.ref_loader = ref_loader
        self.loss_func = loss_func
        self.predict_func = get_func_by_name(
            predict_func, predict_function_map, "predict_func"
        )

    def __call__(self, model: torch.nn.Module):
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for inputs, targets in self.ref_loader:
                # Forward pass
                outputs = self.predict_func(model, inputs)

                # Calculate the loss
                loss = self.loss_func(outputs, targets)

                # Accumulate the total loss
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches


def _constrained_greedy_opt(
    *,
    model: torch.nn.Module,
    target_example: torch.Tensor,
    criterion_thresholds: Iterable[float],
    robustness_criterion: Callable,
    score_func: Callable,
    optimizer_class: Callable,
    direction: int = +1,
    step_size: float = 10e-5,
    max_steps: int = 100,
    weight_decay: float = 0.0,
    verbose: bool = False,
):
    num_thresholds = len(criterion_thresholds)

    with torch.no_grad():
        init_criterion = cur_criterion = robustness_criterion(model=model)
        init_score = best_score = cur_score = score_func(model, target_example)

    model.train()
    for submodule in model.modules():
        # Freeze BatchNorm layers, because here our batch size is 1.
        if isinstance(submodule, torch.nn.BatchNorm2d):
            submodule.eval()
        # Freeze Dropout.
        if isinstance(submodule, torch.nn.Dropout):
            submodule.eval()

    optimizer = optimizer_class(
        model.parameters(), lr=step_size, weight_decay=weight_decay
    )

    latest_threshold_index = 0
    results = [init_score] * len(criterion_thresholds)

    if verbose:
        print("beginning greedy optimization")
        print(f"{init_criterion=:.4f} {init_score=:.4f}")

    for step in range(max_steps):
        # Check the constraints.
        with torch.no_grad():
            cur_criterion = robustness_criterion(model=model)
            if verbose:
                print(f"{step=}. {cur_criterion=:.4f}, {cur_score=:.4f}")

            for i in range(latest_threshold_index, num_thresholds):
                threshold = criterion_thresholds[i]
                if abs(cur_criterion - init_criterion) >= threshold:
                    results[i] = best_score.cpu().numpy()
                    if verbose:
                        print(
                            f"reached threshold. {threshold=:.4f}, {best_score=:.4f}",
                        )
                    latest_threshold_index = i + 1

            if latest_threshold_index == num_thresholds and verbose:
                print("done.")
                break

            cur_score = cur_score.detach()
            if direction == -1:
                if cur_score > best_score:
                    best_score = cur_score
            else:
                if cur_score < best_score:
                    best_score = cur_score

            # Propagate the best score.
            for j in range(latest_threshold_index, len(criterion_thresholds)):
                results[j] = float(best_score.cpu().numpy())

        # Make gradient steps.
        optimizer.zero_grad()
        cur_score = score_func(model, target_example)
        target_loss = direction * cur_score
        target_loss.backward()
        optimizer.step()

    return float(init_score.cpu().numpy()), np.array(results)


def viable_prediction_range(
    *,
    model: torch.nn.Module,
    target_example: Union[np.ndarray, torch.Tensor],
    robustness_criterion: Union[str, Callable],
    criterion_thresholds: Union[Iterable[float], float],
    score_func: Union[str, Callable] = "output",
    optimizer_class=None,
    step_size: float = 10e-5,
    max_steps: int = 100,
    weight_decay: float = 0.0,
    verbose: bool = False,
):
    """
    Viable prediction range at given maximum loss difference.

    Args:
        model: Torch module.
        target_example: Target example for which we compute the range.
        robustness_criterion: Criterion which decides until when the range is computed.
        criterion_thresholds: List of strictly increasing thresholds for the stopping criterion.
        score_func: Scoring function to be optimized in the Rashomon set.
        optimizer_class: Optimizer.
        step_size: Optimizer step size.
        max_steps: Maximum optimization steps.
        weight_decay: Optimizer weight_decay.
        verbose: Whether to output progress.
    """
    score_func = get_func_by_name(score_func, score_function_map, "score_func")
    _criterion_thresholds = criterion_thresholds

    if isinstance(criterion_thresholds, float):
        _criterion_thresholds = [criterion_thresholds]

    if not all(
        _criterion_thresholds[i] < _criterion_thresholds[i + 1]
        for i in range(len(_criterion_thresholds) - 1)
    ):
        raise ValueError("criterion_thresholds must be strictly increasing.")

    if optimizer_class is None:
        optimizer_class = torch.optim.Adam

    model_prime = copy.deepcopy(model)
    pred1, ubs = _constrained_greedy_opt(
        direction=-1,
        model=model_prime,
        target_example=target_example,
        criterion_thresholds=_criterion_thresholds,
        robustness_criterion=robustness_criterion,
        score_func=score_func,
        optimizer_class=optimizer_class,
        step_size=step_size,
        max_steps=max_steps,
        weight_decay=weight_decay,
        verbose=verbose,
    )

    model_prime = copy.deepcopy(model)
    pred2, lbs = _constrained_greedy_opt(
        direction=+1,
        model=model_prime,
        target_example=target_example,
        criterion_thresholds=_criterion_thresholds,
        robustness_criterion=robustness_criterion,
        score_func=score_func,
        optimizer_class=optimizer_class,
        step_size=step_size,
        max_steps=max_steps,
        weight_decay=weight_decay,
        verbose=verbose,
    )

    assert pred1 == pred2
    assert all(lbs <= pred1)
    assert all(ubs >= pred1)

    if isinstance(criterion_thresholds, float):
        lbs = lbs[0]
        ubs = ubs[0]

    return lbs, pred1, ubs

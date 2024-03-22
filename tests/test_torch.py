import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from multiplicity.torch import (
    LossCriterion,
    ZeroOneErrorCriterion,
    viable_prediction_range,
)


@pytest.fixture
def binary_classification_setup():
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Select two classes for binary classification (e.g., class 0 and class 1)
    X_binary = X[y != 2]
    y_binary = y[y != 2]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_binary, y_binary, test_size=0.2, random_state=42
    )

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    # Define the neural network model
    class TwoLayerNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(TwoLayerNN, self).__init__()
            self.layer1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            y_prob = self.sigmoid(x)
            return y_prob

    # Set the input, hidden, and output sizes
    input_size = X_train.shape[1]
    hidden_size = 10
    output_size = 1

    # Instantiate the model, loss function, and optimizer
    model = TwoLayerNN(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.1)

    # Training the model
    num_epochs = 30
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs.squeeze(), labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, train_loader, X_test_tensor, y_test_tensor


@pytest.mark.parametrize("robustness_criterion", ["loss", "zero_one_error"])
@pytest.mark.parametrize("criterion_thresholds", [0.01, [0.001, 0.01, 0.1]])
def test_viable_prediction_range(
    binary_classification_setup, robustness_criterion, criterion_thresholds
):
    model, train_loader, X_test, y_test = binary_classification_setup
    target_example = X_test[0]

    if robustness_criterion == "zero_one_error":
        robustness_criterion = ZeroOneErrorCriterion(train_loader)
    elif robustness_criterion == "loss":
        robustness_criterion = LossCriterion(train_loader, nn.BCELoss())

    lbs, pred, ubs = viable_prediction_range(
        model=model,
        target_example=target_example,
        criterion_thresholds=criterion_thresholds,
        robustness_criterion=robustness_criterion,
        step_size=1e-3,
        max_steps=50,
        verbose=True,
    )

    if isinstance(criterion_thresholds, float):
        lbs = [lbs]
        ubs = [ubs]
        criterion_thresholds = [criterion_thresholds]

    assert len(lbs) == len(ubs) == len(criterion_thresholds)
    for lb, ub in zip(lbs, ubs):
        assert lb <= pred <= ub

    assert all(lbs[i] >= lbs[i + 1] for i in range(len(lbs) - 1))
    assert all(ubs[i] <= ubs[i + 1] for i in range(len(ubs) - 1))


def test_viable_prediction_range_with_zero(binary_classification_setup):
    model, train_loader, X_test, y_test = binary_classification_setup
    target_example = X_test[0]

    robustness_criterion = ZeroOneErrorCriterion(train_loader)

    lbs, pred, ubs = viable_prediction_range(
        model=model,
        target_example=target_example,
        criterion_thresholds=[0, 0.1, 0.2],
        robustness_criterion=robustness_criterion,
        step_size=1e-3,
        max_steps=50,
        verbose=True,
    )

    # Because the first threshold is zero, the first (lb, ub) should
    # be have width zero too.
    assert lbs[0] == pred == ubs[0]

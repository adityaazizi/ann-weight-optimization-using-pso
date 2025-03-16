import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from typing import Callable, Optional, Dict, List


def create_fitness_function(
    model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, criterion: nn.Module
) -> Callable[[np.ndarray], float]:
    """
    Create a fitness function for PSO based on model loss

    Args:
        model: Neural network model
        X_train: Training features
        y_train: Training labels
        criterion: Loss function

    Returns:
        Fitness function that takes weights and returns loss
    """

    def fitness_function(weights: np.ndarray) -> float:
        model.set_weights(weights)
        with torch.no_grad():
            outputs = model(X_train)
            loss = criterion(outputs, y_train.unsqueeze(1))
            loss_value = loss.item()
        return loss_value

    return fitness_function


def evaluate_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """
    Evaluate model accuracy

    Args:
        model: Neural network model
        X: Features
        y: Labels

    Returns:
        Accuracy score
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        predicted = (outputs >= 0.5).float()
        accuracy = accuracy_score(y.numpy(), predicted.numpy())
    return accuracy


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.01,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Train the neural network model

    Args:
        model: Neural network model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        epochs: Number of training epochs
        lr: Learning rate
        verbose: Whether to print progress

    Returns:
        Dictionary with training history
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train.unsqueeze(1))

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Evaluate
        train_acc = evaluate_model(model, X_train, y_train)
        test_acc = evaluate_model(model, X_test, y_test)

        history["train_loss"].append(loss.item())
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if verbose and epoch % 10 == 0:
            print(
                f"Epoch {epoch}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
            )

    end_time = time.time()
    if verbose:
        print(f"Training completed in {end_time - start_time:.2f} seconds")

    return history


def plot_results(
    pso_history: List[float],
    standard_history: Dict[str, List[float]],
    pso_history_train: Dict[str, List[float]],
    title: str = "PSO vs Standard Initialization",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the results of the experiment

    Args:
        pso_history: PSO convergence history
        standard_history: Training history for standard initialization
        pso_history_train: Training history for PSO initialization
        title: Plot title
        save_path: Path to save the figure, or None to display
    """
    plt.figure(figsize=(15, 10))

    # Plot PSO convergence
    plt.subplot(2, 2, 1)
    plt.plot(list(range(len(pso_history))), pso_history, "g-")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness (Loss)")
    plt.title("PSO Convergence")

    # Plot learning curve (loss)
    plt.subplot(2, 2, 2)
    plt.plot(standard_history["train_loss"], "b-", label="Standard Init")
    plt.plot(pso_history_train["train_loss"], "r-", label="PSO Init")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Plot learning curve (train accuracy)
    plt.subplot(2, 2, 3)
    plt.plot(standard_history["train_acc"], "b-", label="Standard Init")
    plt.plot(pso_history_train["train_acc"], "r-", label="PSO Init")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()

    # Plot learning curve (test accuracy)
    plt.subplot(2, 2, 4)
    plt.plot(standard_history["test_acc"], "b-", label="Standard Init")
    plt.plot(pso_history_train["test_acc"], "r-", label="PSO Init")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

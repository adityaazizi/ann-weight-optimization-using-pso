import torch
import numpy as np
from typing import Tuple
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_breast_cancer_data(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and preprocess Wisconsin Breast Cancer dataset

    Args:
        test_size: Proportion of dataset to include in test split
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as PyTorch tensors
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Load dataset
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

    return X_train, X_test, y_train, y_test

def get_dataset_info(X_train, X_test, y_train, y_test):
    """
    Get basic information about a dataset
    """
    info = {
        "n_train_samples": X_train.shape[0],
        "n_test_samples": X_test.shape[0],
        "n_features": X_train.shape[1],
        "is_binary": len(torch.unique(y_train)) <= 2,
        "classes": torch.unique(y_train).numpy().tolist(),
    }
    
    if info["is_binary"]:
        # Get class distribution for binary classification
        info["class_distribution_train"] = {
            "positive": (y_train == 1).float().mean().item(),
            "negative": (y_train == 0).float().mean().item()
        }
        info["class_distribution_test"] = {
            "positive": (y_test == 1).float().mean().item(),
            "negative": (y_test == 0).float().mean().item()
        }
    
    return info

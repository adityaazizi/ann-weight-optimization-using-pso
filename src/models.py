import time
import torch
import numpy as np
import torch.nn as nn
from typing import Callable, Tuple, List

class SimpleANN(nn.Module):
    """Simple Artificial Neural Network with customizable architecture."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def set_weights(self, weights: np.ndarray) -> None:
        """Set model weights from a flat array (for PSO integration)"""
        with torch.no_grad():
            # Weights and bias for fc1
            fc1_weight_size = self.fc1.weight.data.numel()
            fc1_bias_size = self.fc1.bias.data.numel()

            fc1_weight = weights[:fc1_weight_size].reshape(self.fc1.weight.shape)
            fc1_bias = weights[fc1_weight_size : fc1_weight_size + fc1_bias_size]

            # Weights and bias for fc2
            fc2_weight_size = self.fc2.weight.data.numel()
            fc2_weight_start = fc1_weight_size + fc1_bias_size
            fc2_weight = weights[
                fc2_weight_start : fc2_weight_start + fc2_weight_size
            ].reshape(self.fc2.weight.shape)
            fc2_bias = weights[fc2_weight_start + fc2_weight_size :]

            # Set weights to model
            self.fc1.weight.data = torch.FloatTensor(fc1_weight)
            self.fc1.bias.data = torch.FloatTensor(fc1_bias)
            self.fc2.weight.data = torch.FloatTensor(fc2_weight)
            self.fc2.bias.data = torch.FloatTensor(fc2_bias)

    def get_total_params(self) -> int:
        """Return the total number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())


class PSO:
    """Particle Swarm Optimization implementation for neural network weight optimization."""

    def __init__(
        self,
        n_particles: int,
        n_dimensions: int,
        fitness_function: Callable[[np.ndarray], float],
        bounds: Tuple[float, float] = (-1, 1),
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
    ):
        """
        Initialize PSO optimizer

        Args:
            n_particles: Number of particles in the swarm
            n_dimensions: Dimensionality of the search space (total parameters)
            fitness_function: Function to evaluate particle fitness
            bounds: Tuple of (min, max) for particle position limits
            w: Inertia weight
            c1: Cognitive weight
            c2: Social weight
        """
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.fitness_function = fitness_function
        self.bounds = bounds

        # PSO parameters
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Initialize positions and velocities
        self.positions = np.random.uniform(
            bounds[0], bounds[1], (n_particles, n_dimensions)
        )
        self.velocities = np.random.uniform(-0.1, 0.1, (n_particles, n_dimensions))

        # Initialize best positions
        self.p_best_positions = self.positions.copy()
        self.p_best_scores = np.array(
            [self.fitness_function(p) for p in self.positions]
        )

        # Global best
        self.g_best_index = np.argmin(self.p_best_scores)
        self.g_best_position = self.p_best_positions[self.g_best_index].copy()
        self.g_best_score = self.p_best_scores[self.g_best_index]

    def update(self) -> None:
        """Update particle positions and velocities for one iteration"""
        # Generate random coefficients
        r1, r2 = np.random.random(2)

        # Update velocities
        self.velocities = (
            self.w * self.velocities
            + self.c1 * r1 * (self.p_best_positions - self.positions)
            + self.c2 * r2 * (self.g_best_position - self.positions)
        )

        # Update positions
        self.positions += self.velocities

        # Apply position bounds
        self.positions = np.clip(self.positions, self.bounds[0], self.bounds[1])

        # Evaluate current positions
        current_scores = np.array([self.fitness_function(p) for p in self.positions])

        # Update personal bests
        for i in range(self.n_particles):
            if current_scores[i] < self.p_best_scores[i]:
                self.p_best_scores[i] = current_scores[i]
                self.p_best_positions[i] = self.positions[i].copy()

        # Update global best
        min_index = np.argmin(self.p_best_scores)
        if self.p_best_scores[min_index] < self.g_best_score:
            self.g_best_index = min_index
            self.g_best_position = self.p_best_positions[min_index].copy()
            self.g_best_score = self.p_best_scores[min_index]

    def optimize(
        self, n_iterations: int, verbose: bool = True
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run PSO optimization for specified number of iterations

        Args:
            n_iterations: Number of iterations to run
            verbose: Whether to print progress information

        Returns:
            Tuple of (best_position, best_score, history)
        """
        history = []  # List to store best_score for each iteration
        start_time = time.time()

        for i in range(n_iterations):
            self.update()
            history.append(float(self.g_best_score))

            if verbose and i % 10 == 0:
                print(f"Iteration {i}, Best Score: {self.g_best_score:.6f}")

        end_time = time.time()
        if verbose:
            print(f"PSO optimization completed in {end_time - start_time:.2f} seconds")

        return self.g_best_position, self.g_best_score, history

# ANN Weight Optimization Using PSO

Implementation of an Artificial Neural Network (ANN) with weight initialization optimized using Particle Swarm Optimization (PSO) for an Expert System Course in my 7th Semester, created with my two university buddies Abrar Dwi Fairuz Nadhif and Abid Ammar Mahdy.

## Overview

This project explores using Particle Swarm Optimization (PSO) as an alternative method for initializing neural network weights. By comparing PSO-initialized networks with standard random initialization, we investigate whether meta-heuristic algorithms can provide better starting points for gradient-based training.

## Dataset

The experiments use the Wisconsin Breast Cancer dataset, a binary classification problem with 30 features extracted from digitized images of breast mass cell nuclei. The task is to classify tumors as malignant or benign.

## Project Structure

```
ann-weight-optimization-using-pso/
├── src/
│   ├── __init__.py
│   ├── models.py      # Neural network and PSO implementation
│   ├── data.py        # Dataset loading and preprocessing
│   └── utils.py       # Training, evaluation, and visualization utilities
├── notebooks/
│   ├── experiment.ipynb   # Main experiment notebook
│   └── comparison.png  # Results and visualizations
├── .python-version   # Python version specification for pyenv
├── poetry.lock       # Lock file for dependencies
├── pyproject.toml    # Project configuration and dependencies
└── README.md         # Project documentation
```

## Key Findings

1. **Faster Initial Convergence**: PSO-initialized networks achieve significantly lower loss values in early training epochs compared to standard random initialization.

2. **Lower Starting Loss**: Models with PSO-initialized weights begin training with approximately 5x lower loss values than randomly initialized models.

3. **Convergence vs. Generalization**: While PSO initialization leads to faster training convergence, standard initialization eventually achieves comparable or sometimes better test accuracy.

4. **Computational Trade-off**: PSO initialization adds computational overhead as a pre-training step, creating a two-stage training process that may be a bottleneck for larger models.

5. **Application-Specific Benefits**: PSO initialization shows the most promise for applications where rapid initial model deployment is valuable or when training resources are limited.

## Conclusion

This project, originally developed for an Expert Systems course, demonstrates that meta-heuristic approaches like PSO can successfully complement gradient-based neural network training. While not a universal improvement over standard practices, PSO initialization offers distinct advantages in specific scenarios, particularly for accelerating early training performance.

The results highlight the continued value of exploring alternative optimization techniques at the intersection of evolutionary computation and deep learning.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ann-weight-optimization-using-pso.git
cd ann-weight-optimization-using-pso

# Install dependencies using Poetry
poetry install

# Activate the virtual environment
poetry env activate

# Run the Jupyter notebook
jupyter notebook notebooks/experiment.ipynb
```

## Contributor

- Aditya Aulia Al Azizi
- Abrar Dwi Fairuz Nadhif
- Abid Ammar Mahdy

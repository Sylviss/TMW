import torch
import numpy as np


def cost_matrix_numpy(x, y):
    """
    Compute L2 squared distance matrix between two sets of points using NumPy.
    
    Args:
        x: numpy array of shape (m, d) - source points
        y: numpy array of shape (n, d) - target points
        
    Returns:
        numpy array of shape (m, n) - squared L2 distance matrix
    """
    Cxy = np.expand_dims((x**2).sum(axis=1), 1) + np.expand_dims((y**2).sum(axis=1), 0) - 2 * x @ y.T
    return Cxy


def cost_matrix_torch(x, y):
    """
    Compute L2 squared distance matrix between two sets of points using PyTorch.
    
    Args:
        x: torch tensor of shape (m, d) - source points
        y: torch tensor of shape (n, d) - target points
        
    Returns:
        torch tensor of shape (m, n) - squared L2 distance matrix
    """
    return torch.cdist(x, y, p=2).pow(2)

"""
Distance Wrapper Module for Time Series
========================================

A wrapper class for various time series distance functions.
Allows switching between distances by name and editing hyperparameters via a dictionary.

Supported distances:
- opw: Order-Preserving Wasserstein distance
- taot: Time-Aware Optimal Transport
- tcot: Time-Coupled Optimal Transport
- awswd: Auto-Weighted Sinkhorn Wasserstein Distance
- tmw: Temporal Masked Wasserstein distance
- otw: Optimal Transport Warping
- sdtw: Soft Dynamic Time Warping
- dtw: Dynamic Time Warping (using tslearn library)
- euclidean: Euclidean distance
"""

import torch
import torch.nn as nn
import numpy as np
from numba import jit
from tslearn.metrics import dtw as tslearn_dtw

from dtw_mine.opw import OrderPreservingWasserstein, TAOT, TCOT, AWSWD
from dtw_mine.sDTW import sDTW
from dtw_mine.gow import GeneralizedOrderedWassersteinKNN
from dtw_mine.pow import PartialOrderedWassersteinKNN
from dtw_mine.otw import otw_distance_pytorch
from tmw_package.temporal_masked_OT_dist import TMWDist


class TimeSeriesDistance(nn.Module):
    """
    A wrapper class for various Time Series distance functions.
    Allows switching between distances by name and editing hyperparameters via a dictionary.
    
    All distances expect inputs of shape (batch_size, seq_len, dims) and return
    a tensor of shape (batch_size,) containing pairwise distances.
    """
    
    def __init__(self, distance_name, distance_params=None, device=None):
        """
        Initialize the distance wrapper.
        
        Args:
            distance_name: str - Name of the distance function
            distance_params: dict - Optional parameters to override defaults
            device: torch.device - Device to use for computation
        """
        super().__init__()
        self.distance_name = distance_name.lower()
        self.distance_params = distance_params if distance_params is not None else {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default configurations for each distance
        self.default_params = {
            'opw': {
                'lambda1': 1.0,
                'lambda2': 1.0,
                'sigma': 1.0,
                'num_iter': 20
            },
            'taot': {
                'reg_lambda': 50.0,
                'time_weight': 1.0,
                'num_iter': 100
            },
            'tcot': {
                'reg_lambda': 10.0,
                'num_iter': 100
            },
            'awswd': {
                'reg_lambda': 50.0,
                'l_window': 5,
                'k_steep': 0.1,
                'num_sinkhorn': 50,
                'num_outer': 5
            },
            'tmw': {
                'cost_function': "L2",
                'mask_type': 1,
                'reg': 0.01,
                'max_iterations': 1000,
                'thres': 1e-5,
                'eps_threshold': 0.1,
                'masked': True,
                'rescale': True
            },
            'otw': {
                'm_cost': 1.0,
                's_window': -1,  # -1 means global (s=n)
                'beta_smooth_l1': 1.0,
                'strategy_neg': 'direct'  # 'direct' or 'split_pos_neg'
            },
            'sdtw': {
                'use_cuda': True,
                'gamma': 1.0,
                'normalize': False,
                'bandwidth': None
            },
            'gow': {
                'lambda1': 5.0,
                'lambda2': 0.1,
                'max_iter': 5,
                'sinkhorn_iter': 20,
                'fw_iter': 10
            },
            'pow': {
                'order_reg': 1.0,
                'sinkhorn_reg': 0.05,
                'm_mass': 0.8,
                'num_iter': 20
            },
            'dtw': {
                'global_constraint': None,  # None, 'itakura', or 'sakoe_chiba'
                'sakoe_chiba_radius': None,
                'itakura_max_slope': None
            },
            'euclidean': {}
        }
        
        if self.distance_name not in self.default_params:
            raise ValueError(
                f"Unknown distance name: {self.distance_name}. "
                f"Available: {list(self.default_params.keys())}"
            )
        
        # Initialize params with defaults and override with user provided
        self.params = self.default_params[self.distance_name].copy()
        self.params.update(self.distance_params)
        
        self.distance_module = self._build_distance_module()
    
    def _build_distance_module(self):
        """Build the underlying distance module based on distance_name."""
        p = self.params
        
        if self.distance_name == 'opw':
            return OrderPreservingWasserstein(**p)
        
        elif self.distance_name == 'taot':
            return TAOT(**p)
        
        elif self.distance_name == 'tcot':
            return TCOT(**p)
        
        elif self.distance_name == 'awswd':
            return AWSWD(**p)
        
        elif self.distance_name == 'tmw':
            return TMWDist(device=self.device)
        
        elif self.distance_name == 'sdtw':
            return sDTW(**p)
        
        elif self.distance_name == 'gow':
            return GeneralizedOrderedWassersteinKNN(**p)
        
        elif self.distance_name == 'pow':
            return PartialOrderedWassersteinKNN(**p)
        
        elif self.distance_name == 'otw':
            # OTW uses a function, not a module - we'll call it in forward
            return None
        
        elif self.distance_name == 'dtw':
            # DTW doesn't need a module - we'll use tslearn in forward
            return None
        
        elif self.distance_name == 'euclidean':
            # Simple Euclidean doesn't need a module
            return None
        
        else:
            raise ValueError(f"Unknown distance: {self.distance_name}")
    
    def _update_module_params(self):
        """Updates the internal module parameters from self.params dictionary."""
        p = self.params
        m = self.distance_module
        
        if m is None:
            return
        
        if self.distance_name == 'opw':
            m.lambda1 = p['lambda1']
            m.lambda2 = p['lambda2']
            m.sigma = p['sigma']
            m.num_iter = p['num_iter']
        
        elif self.distance_name == 'taot':
            m.reg_lambda = p['reg_lambda']
            m.w = p['time_weight']
            m.num_iter = p['num_iter']
        
        elif self.distance_name == 'tcot':
            m.reg_lambda = p['reg_lambda']
            m.num_iter = p['num_iter']
        
        elif self.distance_name == 'awswd':
            m.reg_lambda = p['reg_lambda']
            m.l = p['l_window']
            m.k = p['k_steep']
            m.num_sinkhorn = p['num_sinkhorn']
            m.num_outer = p['num_outer']
        
        elif self.distance_name == 'sdtw':
            m.gamma = p['gamma']
            m.normalize = p['normalize']
            m.use_cuda = p['use_cuda']
            bw = p['bandwidth']
            m.bandwidth = 0 if bw is None else float(bw)
        
        elif self.distance_name == 'gow':
            m.lambda1 = p['lambda1']
            m.lambda2 = p['lambda2']
            m.max_iter = p['max_iter']
            m.sinkhorn_iter = p['sinkhorn_iter']
            m.fw_iter = p['fw_iter']
        
        elif self.distance_name == 'pow':
            m.order_reg = p['order_reg']
            m.sinkhorn_reg = p['sinkhorn_reg']
            m.m_mass = p['m_mass']
            m.num_iter = p['num_iter']
    
    @torch.no_grad()
    def forward(self, x, y):
        """
        Computes the distance between x and y.
        
        Args:
            x, y: Input tensors of shape (batch_size, seq_len, dims)
            
        Returns:
            torch.Tensor: Distance for each pair, shape (batch_size,)
        """
        # Update module parameters in case self.params was modified
        self._update_module_params()
        
        # Ensure inputs are on correct device
        x = x.to(self.device)
        y = y.to(self.device)
        
        if self.distance_name == 'tmw':
            # TMW forward takes additional parameters
            forward_args = {k: v for k, v in self.params.items()}
            return self.distance_module(x, y, **forward_args)
        
        elif self.distance_name in ['opw', 'sdtw', 'gow', 'pow']:
            # These return just the distance
            return self.distance_module(x, y)
        
        elif self.distance_name in ['taot', 'tcot', 'awswd']:
            # These return (distance, transport_plan) tuple
            result = self.distance_module(x, y)
            if isinstance(result, tuple):
                return result[0]
            return result
        
        elif self.distance_name == 'dtw':
            return self._compute_dtw_batch(x, y)
        
        elif self.distance_name == 'otw':
            return self._compute_otw_batch(x, y)
        
        elif self.distance_name == 'euclidean':
            return self._compute_euclidean_batch(x, y)
        
        else:
            raise ValueError(f"Unknown distance: {self.distance_name}")
    
    def _compute_dtw_batch(self, x, y):
        """
        Compute DTW distance for batched inputs using tslearn.
        Falls back to a simple numpy implementation if tslearn unavailable.
        
        Args:
            x, y: Tensors of shape (batch_size, seq_len, dims)
            
        Returns:
            Tensor of shape (batch_size,) with DTW distances
        """
        batch_size = x.shape[0]
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        
        try:
            distances = []
            p = self.params
            
            for i in range(batch_size):
                d = tslearn_dtw(
                    x_np[i], y_np[i],
                    global_constraint=p.get('global_constraint'),
                    sakoe_chiba_radius=p.get('sakoe_chiba_radius'),
                    itakura_max_slope=p.get('itakura_max_slope')
                )
                distances.append(d)
            
            return torch.tensor(distances, device=self.device, dtype=x.dtype)
        
        except ImportError:
            # Fallback to simple DTW implementation
            print("Warning: tslearn not available, using simple DTW implementation")
            return self._simple_dtw_batch(x_np, y_np)
    
    def _simple_dtw_batch(self, x_np, y_np):
        """Simple DTW implementation as fallback."""
        @jit(nopython=True)
        def dtw_distance(s1, s2):
            n, m = len(s1), len(s2)
            dtw_matrix = np.full((n + 1, m + 1), np.inf)
            dtw_matrix[0, 0] = 0
            
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = np.sum((s1[i-1] - s2[j-1]) ** 2)
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],
                        dtw_matrix[i, j-1],
                        dtw_matrix[i-1, j-1]
                    )
            
            return np.sqrt(dtw_matrix[n, m])
        
        batch_size = x_np.shape[0]
        distances = np.zeros(batch_size)
        
        for i in range(batch_size):
            distances[i] = dtw_distance(x_np[i], y_np[i])
        
        return torch.tensor(distances, device=self.device)
    
    def _compute_euclidean_batch(self, x, y):
        """
        Compute Euclidean distance for batched inputs.
        
        Args:
            x, y: Tensors of shape (batch_size, seq_len, dims)
            
        Returns:
            Tensor of shape (batch_size,) with Euclidean distances
        """
        # Flatten seq_len and dims, then compute L2 norm
        diff = x - y
        diff_flat = diff.view(diff.shape[0], -1)
        return torch.norm(diff_flat, dim=1)
    
    def _compute_otw_batch(self, x, y):
        """
        Compute OTW (Optimal Transport Warping) distance for batched inputs.
        
        Args:
            x, y: Tensors of shape (batch_size, seq_len, dims)
            
        Returns:
            Tensor of shape (batch_size,) with OTW distances
        """
        batch_size = x.shape[0]
        p = self.params
        
        distances = []
        for i in range(batch_size):
            # OTW expects 1D tensors, so we need to flatten dims if > 1
            # For multivariate, we compute OTW per dimension and sum
            x_i = x[i]  # Shape: (seq_len, dims)
            y_i = y[i]  # Shape: (seq_len, dims)
            
            if x_i.shape[1] == 1:
                # Univariate case
                d = otw_distance_pytorch(
                    x_i.squeeze(-1), y_i.squeeze(-1),
                    m_cost=p['m_cost'],
                    s_window=p['s_window'],
                    beta_smooth_l1=p['beta_smooth_l1'],
                    strategy_neg=p['strategy_neg']
                )
            else:
                # Multivariate: sum OTW over each dimension
                d = torch.tensor(0.0, device=x.device, dtype=x.dtype)
                for dim in range(x_i.shape[1]):
                    d += otw_distance_pytorch(
                        x_i[:, dim], y_i[:, dim],
                        m_cost=p['m_cost'],
                        s_window=p['s_window'],
                        beta_smooth_l1=p['beta_smooth_l1'],
                        strategy_neg=p['strategy_neg']
                    )
            distances.append(d)
        
        return torch.stack(distances)
    
    def __repr__(self):
        return f"TimeSeriesDistance(name={self.distance_name}, params={self.params})"


# Convenience function for quick distance computation
def compute_distance(x, y, distance_name, distance_params=None, device=None):
    """
    Convenience function to compute distance without explicitly creating a wrapper.
    
    Args:
        x, y: Input tensors of shape (batch_size, seq_len, dims)
        distance_name: str - Name of the distance function
        distance_params: dict - Optional parameters
        device: torch.device - Device to use
        
    Returns:
        torch.Tensor: Distance for each pair
    """
    dist = TimeSeriesDistance(distance_name, distance_params, device)
    return dist(x, y)

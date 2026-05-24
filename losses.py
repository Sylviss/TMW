import torch
import torch.nn as nn
from dtw_mine.opw import OrderPreservingWasserstein, TAOT, TCOT, AWSWD
from dtw_mine.sDTW import sDTW
from dtw_mine.gow import GeneralizedOrderedWasserstein
from dtw_mine.pow import PartialOrderedWasserstein
from tmw_package.temporal_masked_OT_loss import TMWLoss

class TimeSeriesLoss(nn.Module):
    """
    A wrapper class for various Time Series loss functions.
    Allows switching between losses by name and editing hyperparameters via a dictionary.
    """
    def __init__(self, loss_name, loss_params=None):
        super().__init__()
        self.loss_name = loss_name.lower()
        self.loss_params = loss_params if loss_params is not None else {}
        
        # Default configurations for each loss
        self.default_params = {
            'opw': {'lambda1': 1.0, 'lambda2': 1.0, 'sigma': 1.0, 'num_iter': 20},
            'taot': {'reg_lambda': 50.0, 'time_weight': 1.0, 'num_iter': 100},
            'tcot': {'reg_lambda': 10.0, 'num_iter': 100},
            'awswd': {'reg_lambda': 50.0, 'l_window': 5, 'k_steep': 0.1, 'num_sinkhorn': 50, 'num_outer': 5},
            'tmw': {
                'cost_function': "L2", 'mask_type': 1, 'reg': 0.01, 'max_iterations': 1000, 
                'thres': 1e-5, 'eps_threshold': 0.1, 
                'masked': True, 'rescale': True, 'device': None
            },
            'sdtw': {'use_cuda': True, 'gamma': 1.0, 'normalize': False, 'bandwidth': None},
            'gow': {'lambda1': 5.0, 'lambda2': 0.1, 'max_iter': 5, 'sinkhorn_iter': 20, 'fw_iter': 10},
            'pow': {'order_reg': 1.0, 'sinkhorn_reg': 0.05, 'm_mass': 0.8, 'num_iter': 20}
        }
        
        if self.loss_name not in self.default_params:
             raise ValueError(f"Unknown loss name: {self.loss_name}. Available: {list(self.default_params.keys())}")

        # Initialize params with defaults and override with user provided
        self.params = self.default_params[self.loss_name].copy()
        self.params.update(self.loss_params)
        
        self.loss_module = self._build_loss_module()

    def _build_loss_module(self):
        p = self.params
        if self.loss_name == 'opw':
            return OrderPreservingWasserstein(**p)
        elif self.loss_name == 'taot':
            return TAOT(**p)
        elif self.loss_name == 'tcot':
            return TCOT(**p)
        elif self.loss_name == 'awswd':
            return AWSWD(**p)
        elif self.loss_name == 'tmw':
            device = p.get('device', None)
            return TMWLoss(device=device)
        elif self.loss_name == 'sdtw':
            return sDTW(**p)
        elif self.loss_name == 'gow':
            return GeneralizedOrderedWasserstein(**p)
        elif self.loss_name == 'pow':
            return PartialOrderedWasserstein(**p)
            
    def _update_module_params(self):
        """Updates the internal module parameters from self.params dictionary."""
        p = self.params
        m = self.loss_module
        
        if self.loss_name == 'opw':
            m.lambda1 = p['lambda1']
            m.lambda2 = p['lambda2']
            m.sigma = p['sigma']
            m.num_iter = p['num_iter']
        elif self.loss_name == 'taot':
            m.reg_lambda = p['reg_lambda']
            m.w = p['time_weight']
            m.num_iter = p['num_iter']
        elif self.loss_name == 'tcot':
            m.reg_lambda = p['reg_lambda']
            m.num_iter = p['num_iter']
        elif self.loss_name == 'awswd':
            m.reg_lambda = p['reg_lambda']
            m.l = p['l_window']
            m.k = p['k_steep']
            m.num_sinkhorn = p['num_sinkhorn']
            m.num_outer = p['num_outer']
        elif self.loss_name == 'sdtw':
            m.gamma = p['gamma']
            m.normalize = p['normalize']
            m.use_cuda = p['use_cuda']
            bw = p['bandwidth']
            m.bandwidth = 0 if bw is None else float(bw)
        elif self.loss_name == 'gow':
            m.lambda1 = p['lambda1']
            m.lambda2 = p['lambda2']
            m.max_iter = p['max_iter']
            m.sinkhorn_iter = p['sinkhorn_iter']
            m.fw_iter = p['fw_iter']
        elif self.loss_name == 'pow':
            m.order_reg = p['order_reg']
            m.sinkhorn_reg = p['sinkhorn_reg']
            m.m_mass = p['m_mass']
            m.num_iter = p['num_iter']
        # TMW parameters are passed in forward, so no update needed here (except device which is init only)

    def forward(self, x, y):
        """
        Computes the loss between x and y.
        x, y: Input tensors (batch_size, seq_len, dims)
        """
        # Update module parameters in case self.params was modified
        self._update_module_params()
        
        if self.loss_name == 'none':
            # Return zero loss (no regularization)
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        elif self.loss_name == 'tmw':
            # Filter out 'device' as it is not an argument for forward
            forward_args = {k: v for k, v in self.params.items() if k != 'device'}
            return self.loss_module(x, y, **forward_args)
        else:
            return self.loss_module(x, y)

import torch
import torch.nn as nn
import math


class PartialOrderedWasserstein(nn.Module):
    def __init__(self, order_reg=1.0, sinkhorn_reg=0.05, m_mass=0.8, num_iter=1000, eps=1e-8, stop_thr=5e-3):
        super(PartialOrderedWasserstein, self).__init__()
        self.order_reg = order_reg
        self.sinkhorn_reg = sinkhorn_reg
        self.m_mass = m_mass
        self.num_iter = num_iter
        self.eps = eps
        self.stop_thr = stop_thr

    def forward(self, x, y):
        batch_size, n, dim = x.shape
        _, m, _ = y.shape
        device, dtype = x.device, x.dtype

        D = torch.sum((x.unsqueeze(2) - y.unsqueeze(1))**2, dim=-1)
        med = torch.median(D.view(batch_size, -1), dim=-1).values
        D_norm = D / torch.clamp(med, min=self.eps).view(batch_size, 1, 1)

        i_grid = torch.arange(n, device=device, dtype=dtype).view(n, 1) / n
        j_grid = torch.arange(m, device=device, dtype=dtype).view(1, m) / m
        I = (i_grid - j_grid) ** 2
        
        M = D_norm + self.order_reg * I.unsqueeze(0)

        a = torch.ones(batch_size, n, device=device, dtype=dtype) / n
        b = torch.ones(batch_size, m, device=device, dtype=dtype) / m
        
        dummy_val = 1.0 - self.m_mass
        a_ext = torch.cat([a, torch.full((batch_size, 1), dummy_val, device=device, dtype=dtype)], dim=1)
        b_ext = torch.cat([b, torch.full((batch_size, 1), dummy_val, device=device, dtype=dtype)], dim=1)

        M_ext = torch.zeros((batch_size, n + 1, m + 1), device=device, dtype=dtype)
        M_ext[:, :n, :m] = M
        M_ext[:, n, m] = M.view(batch_size, -1).max(dim=-1).values * 100

        log_K = -M_ext / self.sinkhorn_reg
        log_u = torch.zeros_like(a_ext)
        log_v = torch.zeros_like(b_ext)
        log_a = torch.log(a_ext + 1e-12)
        log_b = torch.log(b_ext + 1e-12)

        for _ in range(self.num_iter):
            prev_log_u = log_u.clone()
            
            log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)
            log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
            
            if torch.max(torch.abs(log_u - prev_log_u)) < self.stop_thr:
                break

        log_T_ext = log_u.unsqueeze(2) + log_v.unsqueeze(1) + log_K
        T_partial = torch.exp(log_T_ext[:, :n, :m])

        pow_dist = torch.sum(T_partial * D, dim=(1, 2))
        
        return pow_dist
    

class PartialOrderedWassersteinKNN(nn.Module):
    def __init__(self, order_reg=1.0, sinkhorn_reg=0.01, m_mass=0.8, num_iter=2000, eps=1e-8, stop_thr=1e-5):
        super(PartialOrderedWassersteinKNN, self).__init__()
        self.order_reg = order_reg
        self.sinkhorn_reg = sinkhorn_reg
        self.m_mass = m_mass
        self.num_iter = num_iter
        self.eps = eps
        self.stop_thr = stop_thr

    def forward(self, x, y):
        batch_size, n, dim = x.shape
        _, m, _ = y.shape
        device, dtype = x.device, x.dtype

        # 1. Compute Cost Matrix (REMOVED MEDIAN NORMALIZATION)
        D = torch.sum((x.unsqueeze(2) - y.unsqueeze(1))**2, dim=-1)
        # Optional: scale D globally by feature dimension if numbers get too large
        D = D / dim 

        # 2. Order Regularization
        i_grid = torch.arange(n, device=device, dtype=dtype).view(n, 1) / n
        j_grid = torch.arange(m, device=device, dtype=dtype).view(1, m) / m
        I = (i_grid - j_grid) ** 2
        
        M = D + self.order_reg * I.unsqueeze(0)

        # 3. Marginals
        a = torch.ones(batch_size, n, device=device, dtype=dtype) / n
        b = torch.ones(batch_size, m, device=device, dtype=dtype) / m
        
        dummy_val = 1.0 - self.m_mass
        a_ext = torch.cat([a, torch.full((batch_size, 1), dummy_val, device=device, dtype=dtype)], dim=1)
        b_ext = torch.cat([b, torch.full((batch_size, 1), dummy_val, device=device, dtype=dtype)], dim=1)

        # 4. Extended Cost Matrix
        M_ext = torch.zeros((batch_size, n + 1, m + 1), device=device, dtype=dtype)
        M_ext[:, :n, :m] = M
        # High penalty for dummy-to-dummy matching
        max_cost = M.view(batch_size, -1).max(dim=-1).values * 100
        M_ext[:, n, m] = max_cost

        # 5. Log-domain Sinkhorn
        log_K = -M_ext / self.sinkhorn_reg
        log_u = torch.zeros_like(a_ext)
        log_v = torch.zeros_like(b_ext)
        log_a = torch.log(a_ext + 1e-12)
        log_b = torch.log(b_ext + 1e-12)

        for _ in range(self.num_iter):
            prev_log_u = log_u.clone()
            
            log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)
            log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
            
            if torch.max(torch.abs(log_u - prev_log_u)) < self.stop_thr:
                break

        # 6. Extract valid transport plan and compute distance
        log_T_ext = log_u.unsqueeze(2) + log_v.unsqueeze(1) + log_K
        T_partial = torch.exp(log_T_ext[:, :n, :m])

        # Compute the final POW distance
        pow_dist = torch.sum(T_partial * D, dim=(1, 2))
        
        return pow_dist
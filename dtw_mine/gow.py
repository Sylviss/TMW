import torch
import torch.nn as nn
import math

class GeneralizedOrderedWasserstein(nn.Module):
    def __init__(self, lambda1=10.0, lambda2=5.0, max_iter=15, sinkhorn_iter=100, fw_iter=100, stop_thr=5e-3):
        super(GeneralizedOrderedWasserstein, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.sinkhorn_iter = sinkhorn_iter
        self.fw_iter = fw_iter
        self.stop_thr = stop_thr

    def _get_basis_functions(self, n, m, device, dtype):
        j_norm = torch.arange(m, device=device, dtype=dtype) / (m - 1 if m > 1 else 1.0)
        f1 = torch.pow(j_norm, 0.05)
        f2 = torch.pow(j_norm, 0.28)
        f3 = j_norm 
        f4 = torch.pow(j_norm, 3.2)
        f5 = torch.pow(j_norm, 20.0)
        basis = torch.stack([f1, f2, f3, f4, f5], dim=1)
        return (n - 1) * basis

    def _compute_warped_cost(self, D_norm, w, F_scaled, n):
        batch_size = w.shape[0]
        m = F_scaled.shape[0]
        F_batch = F_scaled.t().unsqueeze(0).expand(batch_size, -1, -1)
        proj_f = torch.bmm(w.transpose(1, 2), F_batch) 
        i_grid = torch.arange(n, device=D_norm.device, dtype=D_norm.dtype).view(1, n, 1)
        warping_term = ((i_grid - proj_f)**2) / (n**2)
        return D_norm + self.lambda1 * warping_term

    def forward(self, x, y):
        batch_size, n, dim = x.shape
        _, m, _ = y.shape
        device, dtype = x.device, x.dtype

        D = torch.sum((x.unsqueeze(2) - y.unsqueeze(1))**2, dim=-1)
        med = torch.median(D.view(batch_size, -1), dim=-1).values
        D_norm = D / torch.clamp(med, min=1e-8).view(batch_size, 1, 1)

        F_scaled = self._get_basis_functions(n, m, device, dtype)
        num_fn = F_scaled.shape[1]
        
        w = torch.zeros(batch_size, num_fn, 1, device=device, dtype=dtype)
        w[:, 2, :] = 1.0
        
        log_alpha = -math.log(n)
        log_beta = -math.log(m)

        T = None
        for _ in range(self.max_iter):
            D_gow = self._compute_warped_cost(D_norm, w, F_scaled, n)
            log_K = -D_gow * (1.0 / self.lambda2)
            
            log_u = torch.zeros(batch_size, n, device=device, dtype=dtype)
            log_v = torch.zeros(batch_size, m, device=device, dtype=dtype)
            
            # Inner Sinkhorn Loop with Early Stopping
            for _s in range(self.sinkhorn_iter):
                prev_log_u = log_u.clone()
                
                log_v = log_beta - torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)
                log_u = log_alpha - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
                
                if torch.max(torch.abs(log_u - prev_log_u)) < self.stop_thr:
                    break
                    
            T = torch.exp(log_K + log_u.unsqueeze(2) + log_v.unsqueeze(1))

            with torch.no_grad():
                sqrt_T = torch.sqrt(T + 1e-9)
                i_idx = torch.arange(n, device=device, dtype=dtype).view(1, n, 1)
                Y = (sqrt_T * i_idx).view(batch_size, -1, 1)
                V = (sqrt_T.unsqueeze(-1) * F_scaled.view(1, 1, m, num_fn)).view(batch_size, -1, num_fn)
                
                for _f in range(self.fw_iter):
                    res = Y - torch.bmm(V, w)
                    grad = -2 * torch.bmm(V.transpose(1, 2), res)
                    best_idx = torch.argmin(grad.squeeze(-1), dim=1)
                    s = torch.zeros_like(w)
                    s[torch.arange(batch_size), best_idx, 0] = 1.0
                    gamma = 2.0 / (_f + 2.0)
                    w = w + gamma * (s - w)

        gow_dist = torch.sum(T * D, dim=(1, 2))
        return gow_dist
    


class GeneralizedOrderedWassersteinKNN(nn.Module):
    def __init__(self, lambda1=10.0, lambda2=10.0, max_iter=15, sinkhorn_iter=100, fw_iter=100, stop_thr=1e-4):
        super(GeneralizedOrderedWassersteinKNN, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2  # Higher lambda2 = sharper matching (better for K-NN)
        self.max_iter = max_iter
        self.sinkhorn_iter = sinkhorn_iter
        self.fw_iter = fw_iter
        self.stop_thr = stop_thr

    def _get_basis_functions(self, n, m, device, dtype):
        j_norm = torch.arange(m, device=device, dtype=dtype) / (m - 1 if m > 1 else 1.0)
        f1 = torch.pow(j_norm, 0.05)
        f2 = torch.pow(j_norm, 0.28)
        f3 = j_norm 
        f4 = torch.pow(j_norm, 3.2)
        f5 = torch.pow(j_norm, 20.0)
        basis = torch.stack([f1, f2, f3, f4, f5], dim=1)
        return (n - 1) * basis

    def _compute_warped_cost(self, D_base, w, F_scaled, n):
        batch_size = w.shape[0]
        m = F_scaled.shape[0]
        F_batch = F_scaled.t().unsqueeze(0).expand(batch_size, -1, -1)
        proj_f = torch.bmm(w.transpose(1, 2), F_batch) 
        i_grid = torch.arange(n, device=D_base.device, dtype=D_base.dtype).view(1, n, 1)
        
        warping_term = ((i_grid - proj_f)**2) / (n**2)
        return D_base + self.lambda1 * warping_term

    # We use no_grad because K-NN does not require backpropagation.
    # This prevents massive memory leaks during evaluation.
    @torch.no_grad() 
    def forward(self, x, y):
        batch_size, n, dim = x.shape
        _, m, _ = y.shape
        device, dtype = x.device, x.dtype

        # 1. Compute Cost Matrix (REMOVED Median Normalization for exact absolute distances)
        D = torch.sum((x.unsqueeze(2) - y.unsqueeze(1))**2, dim=-1)
        
        # Globally scale by feature dim so lambda1 remains balanced with the data scale.
        D_base = D / dim 

        F_scaled = self._get_basis_functions(n, m, device, dtype)
        num_fn = F_scaled.shape[1]
        
        w = torch.zeros(batch_size, num_fn, 1, device=device, dtype=dtype)
        w[:, 2, :] = 1.0 # Initialize with linear function (func3)
        
        log_alpha = -math.log(n)
        log_beta = -math.log(m)

        T = None
        for _ in range(self.max_iter):
            # Compute cost with current Frank-Wolfe weights
            D_gow = self._compute_warped_cost(D_base, w, F_scaled, n)
            
            # Apply lambda2 as originally formulated
            log_K = -D_gow * (1.0 / self.lambda2)
            
            log_u = torch.zeros(batch_size, n, device=device, dtype=dtype)
            log_v = torch.zeros(batch_size, m, device=device, dtype=dtype)
            
            # Inner Sinkhorn Loop
            for _s in range(self.sinkhorn_iter):
                prev_log_u = log_u.clone()
                
                log_v = log_beta - torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)
                log_u = log_alpha - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
                
                if torch.max(torch.abs(log_u - prev_log_u)) < self.stop_thr:
                    break
                    
            T = torch.exp(log_K + log_u.unsqueeze(2) + log_v.unsqueeze(1))

            # Frank-Wolfe step to update function weights `w`
            sqrt_T = torch.sqrt(T + 1e-9)
            i_idx = torch.arange(n, device=device, dtype=dtype).view(1, n, 1)
            Y = (sqrt_T * i_idx).view(batch_size, -1, 1)
            V = (sqrt_T.unsqueeze(-1) * F_scaled.view(1, 1, m, num_fn)).view(batch_size, -1, num_fn)
            
            for _f in range(self.fw_iter):
                res = Y - torch.bmm(V, w)
                grad = -2 * torch.bmm(V.transpose(1, 2), res)
                best_idx = torch.argmin(grad.squeeze(-1), dim=1)
                s = torch.zeros_like(w)
                s[torch.arange(batch_size), best_idx, 0] = 1.0
                gamma = 2.0 / (_f + 2.0)
                w = w + gamma * (s - w)

        # 2. Compute final absolute GOW distance using the unscaled original D 
        gow_dist = torch.sum(T * D, dim=(1, 2))
        return gow_dist
import torch
import torch.nn as nn
import math

class OrderPreservingWasserstein(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=1.0, sigma=1.0, num_iter=100, stop_thr=5e-3):
        super(OrderPreservingWasserstein, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.sigma = sigma
        self.num_iter = num_iter
        self.stop_thr = stop_thr

    def forward(self, x, y):
        batch_size, n, d = x.shape
        _, m, _ = y.shape

        x_expanded = x.unsqueeze(2)
        y_expanded = y.unsqueeze(1)
        D = torch.sum((x_expanded - y_expanded) ** 2, dim=-1)

        i_grid = torch.arange(n, device=x.device, dtype=x.dtype).view(n, 1)
        j_grid = torch.arange(m, device=x.device, dtype=x.dtype).view(1, m)
        
        temporal_diff_sq = (i_grid / n - j_grid / m) ** 2
        log_P = -temporal_diff_sq / (2 * self.sigma ** 2)
        S = self.lambda1 / (1 + temporal_diff_sq)

        log_K = -(D - S) / self.lambda2 + log_P

        alpha = torch.ones(batch_size, n, device=x.device, dtype=x.dtype) / n
        beta = torch.ones(batch_size, m, device=x.device, dtype=x.dtype) / m
        log_alpha = torch.log(alpha)
        log_beta = torch.log(beta)
        log_u = torch.zeros_like(log_alpha)
        log_v = torch.zeros_like(log_beta)

        # Log-space Sinkhorn with Early Stopping
        for _ in range(self.num_iter):
            prev_log_u = log_u.clone()
            
            log_v = log_beta - torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)
            log_u = log_alpha - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
            
            # Check convergence based on the max change in the dual variable
            if torch.max(torch.abs(log_u - prev_log_u)) < self.stop_thr:
                break

        log_T = log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1)
        T = torch.exp(log_T)
        opw_distance = torch.sum(T * D, dim=[1, 2])

        return opw_distance


class TAOT(nn.Module):
    def __init__(self, reg_lambda=50.0, time_weight=1.0, num_iter=1000, eps=1e-8, stop_thr=5e-3):
        super(TAOT, self).__init__()
        self.reg_lambda = reg_lambda
        self.w = time_weight
        self.num_iter = num_iter
        self.eps = eps
        self.stop_thr = stop_thr

    def forward(self, x, y):
        batch_size, n, d = x.shape
        _, m, _ = y.shape

        D_spatial = torch.sum((x.unsqueeze(2) - y.unsqueeze(1))**2, dim=-1)

        t_x = torch.linspace(1, n, n, device=x.device, dtype=x.dtype)
        t_y = torch.linspace(1, m, m, device=y.device, dtype=y.dtype)
        
        def torch_zscore(t):
            mu = t.mean()
            std = t.std()
            return (t - mu) / (std if std > 0 else 1.0)

        zt_x = torch_zscore(t_x)
        zt_y = torch_zscore(t_y)
        D_time = (zt_x.unsqueeze(1) - zt_y.unsqueeze(0))**2

        M = D_spatial + self.w * D_time
        M_flat = M.view(batch_size, -1)
        med = torch.median(M_flat, dim=-1).values
        med = torch.clamp(med, min=self.eps)
        C = M / med.view(batch_size, 1, 1)

        log_K = -C * self.reg_lambda
        log_u = torch.zeros(batch_size, n, device=x.device, dtype=x.dtype)
        log_v = torch.zeros(batch_size, m, device=y.device, dtype=y.dtype)
        log_alpha = -math.log(n)
        log_beta = -math.log(m)

        for _ in range(self.num_iter):
            prev_log_u = log_u.clone()
            
            log_v = log_beta - torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)
            log_u = log_alpha - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
            
            if torch.max(torch.abs(log_u - prev_log_u)) < self.stop_thr:
                break

        T = torch.exp(log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1))
        distance = torch.sum(T * M, dim=[1, 2])
        
        return distance, T


class TCOT(nn.Module):
    def __init__(self, reg_lambda=10.0, num_iter=1000, eps=1e-8, stop_thr=5e-3):
        super(TCOT, self).__init__()
        self.reg_lambda = reg_lambda
        self.num_iter = num_iter
        self.eps = eps
        self.stop_thr = stop_thr

    def forward(self, x, y):
        batch_size, n, d = x.shape
        _, m, _ = y.shape

        D = torch.sum((x.unsqueeze(2) - y.unsqueeze(1))**2, dim=-1)

        i_grid = torch.arange(n, device=x.device, dtype=x.dtype).view(n, 1) / (n - 1 if n > 1 else 1)
        j_grid = torch.arange(m, device=x.device, dtype=x.dtype).view(1, m) / (m - 1 if m > 1 else 1)
        Dt = torch.abs(i_grid - j_grid) + 1.0

        Dc = D * Dt
        medians = torch.median(Dc.view(batch_size, -1), dim=-1).values
        medians = torch.clamp(medians, min=self.eps)
        Dc_norm = Dc / medians.view(batch_size, 1, 1)

        log_K = -Dc_norm * self.reg_lambda
        log_alpha = -math.log(n)
        log_beta = -math.log(m)
        log_u = torch.zeros(batch_size, n, device=x.device, dtype=x.dtype)
        log_v = torch.zeros(batch_size, m, device=x.device, dtype=x.dtype)

        for _ in range(self.num_iter):
            prev_log_u = log_u.clone()
            
            log_v = log_beta - torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)
            log_u = log_alpha - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
            
            if torch.max(torch.abs(log_u - prev_log_u)) < self.stop_thr:
                break

        T = torch.exp(log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1))
        distance = torch.sum(T * Dc, dim=[1, 2])
        
        return distance, T


class AWSWD(nn.Module):
    def __init__(self, reg_lambda=10.0, l_window=3, k_steep=0.1, num_sinkhorn=100, num_outer=3, eps=1e-8, stop_thr=5e-3):
        super(AWSWD, self).__init__()
        self.reg_lambda = reg_lambda
        self.l = l_window
        self.k = k_steep
        self.num_sinkhorn = num_sinkhorn
        self.num_outer = num_outer 
        self.eps = eps
        self.stop_thr = stop_thr

    def forward(self, x, y):
        batch_size, n, d = x.shape
        _, m, _ = y.shape
        t0 = n / 4.0

        D1 = torch.sum((x.unsqueeze(2) - y.unsqueeze(1))**2, dim=-1)

        def get_derivative(seq):
            padded = torch.cat([seq[:, :1, :], seq, seq[:, -1:, :]], dim=1)
            return (padded[:, 2:, :] - padded[:, :-2, :]) / 2.0

        dx, dy = get_derivative(x), get_derivative(y)
        D2 = torch.zeros(batch_size, n, m, device=x.device, dtype=x.dtype)
        for k in range(-self.l, self.l + 1):
            dx_s = torch.roll(dx, shifts=k, dims=1)
            D2 += torch.sum((dx_s.unsqueeze(2) - dy.unsqueeze(1))**2, dim=-1)

        i_idx = torch.arange(n, device=x.device).view(n, 1)
        j_idx = torch.arange(m, device=x.device).view(1, m)
        D3 = 1.0 / (1.0 + torch.exp(-self.k * (torch.abs(i_idx - j_idx).float() - t0)))
        D3 = D3.unsqueeze(0).expand(batch_size, -1, -1)

        w = torch.ones(3, batch_size, device=x.device, dtype=x.dtype) / 3.0
        
        def norm_cost(D):
            med = torch.median(D.view(batch_size, -1), dim=-1).values
            return D / torch.clamp(med, min=self.eps).view(batch_size, 1, 1)

        D1_n, D2_n, D3_n = norm_cost(D1), norm_cost(D2), norm_cost(D3)

        T = None
        for _ in range(self.num_outer):
            D_combined = (w[0].view(-1, 1, 1) * D1_n + 
                          w[1].view(-1, 1, 1) * D2_n + 
                          w[2].view(-1, 1, 1) * D3_n)
            
            log_K = -D_combined * self.reg_lambda
            log_u = torch.zeros(batch_size, n, device=x.device)
            log_v = torch.zeros(batch_size, m, device=y.device)
            log_a, log_b = -math.log(n), -math.log(m)

            # Inner Sinkhorn Loop with Early Stopping
            for _ in range(self.num_sinkhorn):
                prev_log_u = log_u.clone()
                
                log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)
                log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
                
                if torch.max(torch.abs(log_u - prev_log_u)) < self.stop_thr:
                    break
            
            T = torch.exp(log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1))

            L1 = torch.sum(T * D1_n, dim=[1, 2])
            L2 = torch.sum(T * D2_n, dim=[1, 2])
            L3 = torch.sum(T * D3_n, dim=[1, 2])
            
            losses = torch.stack([L1, L2, L3])
            w = 1.0 / (2.0 * torch.sqrt(losses + 1e-6))
            w = w / w.sum(dim=0, keepdim=True)

        final_dist = torch.sum(T * (w[0].view(-1,1,1)*D1 + 
                                    w[1].view(-1,1,1)*D2 + 
                                    w[2].view(-1,1,1)*D3), dim=[1, 2])
        
        return final_dist, T
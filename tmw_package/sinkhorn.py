import torch
import numpy as np


def sinkhorn_log_domain_torch(p, q, C, Mask=None, reg=0.01, niter=10000, thresh=1e-5):
    """
    Sinkhorn algorithm in log domain using PyTorch for GPU acceleration.
    
    Args:
        p: torch tensor of shape (m,) - source distribution
        q: torch tensor of shape (n,) - target distribution
        C: torch tensor of shape (m, n) - cost matrix
        Mask: torch tensor of shape (m, n) - binary mask matrix (optional)
        reg: float - regularization parameter
        niter: int - maximum number of iterations
        thresh: float - convergence threshold
        
    Returns:
        torch tensor of shape (m, n) - optimal transport plan
    """
    C = C / C.max()
    
    def M(u, v):
        """Modified cost for logarithmic updates"""
        M = (-C + torch.unsqueeze(u, 1) + torch.unsqueeze(v, 0)) / reg
        if Mask is not None:
            M[Mask == 0] = -1e6
        return M

    def lse(A):
        """Log-sum-exp operation"""
        max_A, _ = torch.max(A, dim=1, keepdim=True)
        return torch.log(torch.exp(A - max_A).sum(1, keepdim=True) + 1e-10) + max_A

    # Sinkhorn iterations
    u, v, err = 0. * p, 0. * q, 0.
    actual_nits = 0

    for i in range(niter):
        u1 = u
        u = reg * (torch.log(p) - lse(M(u, v)).squeeze()) + u
        v = reg * (torch.log(q) - lse(M(u, v).T).squeeze()) + v
        
        
        err = torch.sum(torch.abs(u - u1))
        actual_nits += 1
        
        if err < thresh:
            break
            
    U, V = u, v
    pi = torch.exp(M(U, V))
    print(f"Iter used: {i}")
    return pi

def sinkhorn_log_domain(p, q, C, Mask=None, reg=0.01, niter=10000, thresh=1e-5):
    """
    Sinkhorn algorithm in log domain using NumPy.
    
    Args:
        p: numpy array of shape (m,) - source distribution
        q: numpy array of shape (n,) - target distribution
        C: numpy array of shape (m, n) - cost matrix
        Mask: numpy array of shape (m, n) - binary mask matrix (optional)
        reg: float - regularization parameter
        niter: int - maximum number of iterations
        thresh: float - convergence threshold
        
    Returns:
        numpy array of shape (m, n) - optimal transport plan
    """
    C /= C.max()
    
    def M(u, v):
        """Modified cost for logarithmic updates"""
        M = (-C + np.expand_dims(u, 1) + np.expand_dims(v, 0)) / reg
        if Mask is not None:
            M[Mask == 0] = -1e6
        return M

    def lse(A):
        """Log-sum-exp operation"""
        max_A = np.max(A, axis=1, keepdims=True)
        return np.log(np.exp(A - max_A).sum(1, keepdims=True) + 1e-10) + max_A

    # Sinkhorn iterations
    u, v, err = 0. * p, 0. * q, 0.
    actual_nits = 0

    for i in range(niter):
        u1 = u
        u = reg * (np.log(p) - lse(M(u, v)).squeeze()) + u
        v = reg * (np.log(q) - lse(M(u, v).T).squeeze()) + v
        err = np.linalg.norm(u - u1)

        actual_nits += 1
        if err < thresh:
            break
            
    U, V = u, v
    pi = np.exp(M(U, V))
    return pi

def sinkhorn_log_domain_refined(p, q, C, Mask=None, reg:float=0.01, niter:int=1000, thresh:float=1e-5):
    """Refined log-domain Sinkhorn with better numerical stability and efficiency."""
    # 1. Pre-computation and robust initialization
    C = C / C.max()
    K = -C / reg
    u = torch.zeros_like(p)
    v = torch.zeros_like(q)
    log_p = torch.log(p)
    log_q = torch.log(q)

    # Pre-calculate where the mask is zero for efficient application
    mask_zero_indices = None
    if Mask is not None:
        mask_zero_indices = (Mask == 0)

    for i in range(niter):
        u_prev = u

        # 2. Avoid redundant M computation
        # Update u
        M_uv = K + (u.unsqueeze(1) + v.unsqueeze(0))/reg
        if mask_zero_indices is not None:
            M_uv[mask_zero_indices] = -1e9 # Use a large negative number
        
        u = reg * (log_p - torch.logsumexp(M_uv.T, dim=0, keepdim=False)) + u

        # Update v
        # We need to re-evaluate M_uv as u has changed
        M_uv = K + (u.unsqueeze(1) + v.unsqueeze(0))/reg
        if mask_zero_indices is not None:
            M_uv[mask_zero_indices] = -1e9
        
        v = reg * (log_q - torch.logsumexp(M_uv, dim=0, keepdim=False)) + v

        # Check for convergence
        err = torch.sum(torch.abs(u - u_prev))
        if err < thresh:
            break

    # Final transport plan
    M_final = K + (u.unsqueeze(1) + v.unsqueeze(0))/reg
    if mask_zero_indices is not None:
        M_final[mask_zero_indices] = -torch.inf
    
    # print(i)
    pi = torch.exp(M_final)
    return pi


# def sinkhorn_log_domain_refined_batched(p, q, C, Mask=None, reg: float = 0.01, niter: int = 1000, thresh: float = 1e-5):
#     """
#     Refined log-domain Sinkhorn with better numerical stability and efficiency,
#     adapted for batch processing.

#     Args:
#         p (torch.Tensor): Batched source marginals, shape (batch_size, n).
#         q (torch.Tensor): Batched target marginals, shape (batch_size, m).
#         C (torch.Tensor): Batched cost matrix, shape (batch_size, n, m).
#         Mask (torch.Tensor, optional): Batched mask, shape (batch_size, n, m). Defaults to None.
#         reg (float, optional): Regularization parameter. Defaults to 0.01.
#         niter (int, optional): Number of iterations. Defaults to 1000.
#         thresh (float, optional): Convergence threshold. Defaults to 1e-5.

#     Returns:
#         torch.Tensor: The batched transport plan, shape (batch_size, n, m).
#     """
#     batch_size, n, m = C.shape

#     # 1. Pre-computation and robust initialization
#     C_max = torch.max(C.view(batch_size, -1), dim=1, keepdim=True)[0].unsqueeze(-1)
#     C = C / C_max
#     K = -C / reg
#     u = torch.zeros_like(p)
#     v = torch.zeros_like(q)
#     log_p = torch.log(p)
#     log_q = torch.log(q)

#     # Pre-calculate where the mask is zero for efficient application
#     mask_zero_indices = None
#     if Mask is not None:
#         mask_zero_indices = (Mask == 0)

#     for i in range(niter):
#         u_prev = u.clone()

#         # 2. Avoid redundant M computation
#         # Update u
#         M_uv = K + u.unsqueeze(2) + v.unsqueeze(1)
#         if mask_zero_indices is not None:
#             M_uv[mask_zero_indices] = -1e9  # Use a large negative number

#         u = reg * (log_p - torch.logsumexp(M_uv, dim=2)) + u

#         # Update v
#         # We need to re-evaluate M_uv as u has changed
#         M_uv = K + u.unsqueeze(2) + v.unsqueeze(1)
#         if mask_zero_indices is not None:
#             M_uv[mask_zero_indices] = -1e9

#         v = reg * (log_q - torch.logsumexp(M_uv, dim=1)) + v

#         # Check for convergence
#         err = torch.mean(torch.abs(u - u_prev), dim=1)
#         if torch.all(err < thresh):
#             break

#     # Final transport plan
#     M_final = K + u.unsqueeze(2) + v.unsqueeze(1)
#     if mask_zero_indices is not None:
#         M_final[mask_zero_indices] = -torch.inf

#     pi = torch.exp(M_final)
#     return pi

def sinkhorn_log_domain_refined_batched(p, q, C, Mask=None, reg: float = 0.01, niter: int = 1000, thresh: float = 1e-5):
    """
    Stable batched Log-domain Sinkhorn.
    
    Args:
        p: (B, N) source marginals
        q: (B, M) target marginals
        C: (B, N, M) cost matrix
        Mask: (B, N, M) binary mask (1 for valid, 0 for blocked)
        reg: Regularization strength (epsilon)
        niter: Max iterations
        thresh: Convergence threshold
    """
    batch_size, n, m = C.shape

    # 1. Normalize Cost to range [0, 1] for stability
    C_max = torch.max(C.view(batch_size, -1), dim=1, keepdim=True)[0].unsqueeze(-1)
    C_norm = C / (C_max + 1e-8)
    
    # Pre-compute K = -C/reg
    K = -C_norm / reg
    
    # Handle Masking (Log domain: 0 mask means -inf probability)
    if Mask is not None:
        K = K.masked_fill(Mask == 0, -1e12)

    # 2. Initialize Potentials
    # u (B, N), v (B, M)
    u = torch.zeros_like(p)
    v = torch.zeros_like(q)
    
    # Use small epsilon in log to avoid -inf if marginals contain zeros
    log_p = torch.log(p + 1e-12)
    log_q = torch.log(q + 1e-12)

    for i in range(niter):
        u_prev = u.clone()

        # Update u (corresponds to row-sum constraint)
        # u = reg * (log_p - logsumexp( (v - C)/reg ))
        # Using broadcasting: (K + v/reg) is (B, N, M)
        u = reg * (log_p - torch.logsumexp(K + v.unsqueeze(1) / reg, dim=2))

        # Update v (corresponds to col-sum constraint)
        # v = reg * (log_q - logsumexp( (u - C)/reg ))
        v = reg * (log_q - torch.logsumexp(K + u.unsqueeze(2) / reg, dim=1))

        # 3. Check Convergence
        # We check the average change in the potential u
        err = torch.mean(torch.abs(u - u_prev), dim=1)
        if torch.all(err < thresh):
            break

    # 4. Final Transport Plan: pi = exp((u + v - C) / reg)
    # Using broadcasting: u(B,N,1) + v(B,1,M) + K(B,N,M)
    pi = torch.exp(K + u.unsqueeze(2) / reg + v.unsqueeze(1) / reg)
    
    return pi
import torch
import torch.nn.functional as F
import numpy as np

def smooth_l1_loss(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Implements the smooth L1 loss (Huber loss) as defined in eq. (9) of the OTW paper.
    L_beta(x) = x^2 / (2*beta)      if |x| < beta
                |x| - beta / 2     if |x| >= beta
    """
    if beta < 1e-5: # Treat as standard L1 if beta is very small
        return torch.abs(x)
    
    abs_x = torch.abs(x)
    # Create masks for the two conditions
    mask_less_than_beta = abs_x < beta
    mask_geq_beta = abs_x >= beta
    
    # Calculate loss for each part
    loss_val = torch.zeros_like(x)
    loss_val[mask_less_than_beta] = (x[mask_less_than_beta] ** 2) / (2 * beta)
    loss_val[mask_geq_beta] = abs_x[mask_geq_beta] - beta / 2.0
    
    return loss_val

def otw_distance_pytorch(
    a: torch.Tensor, 
    b: torch.Tensor, 
    m_cost: float = 1.0, 
    s_window: int = -1, # -1 means global (s=n)
    beta_smooth_l1: float = 1.0,
    strategy_neg: str = "direct" # "direct" or "split_pos_neg"
    ) -> torch.Tensor:
    """
    Computes the Optimal Transport Warping (OTW) distance between two 
    1D time series tensors `a` and `b`.

    Args:
        a (torch.Tensor): The first time series (1D tensor of shape [N]).
        b (torch.Tensor): The second time series (1D tensor of shape [M]).
                          Currently assumes N=M for simplicity with windowed sums.
                          The paper implies N=M when using eq. (8) for OTW_m,s.
        m_cost (float): The cost 'm' for transporting to/from the sink
                        to balance total mass.
        s_window (int): The window size 's' for local cumulative sums.
                        If -1, uses global cumulative sums (s=n, as in eq. 6).
                        Must be >= 1 if not -1.
        beta_smooth_l1 (float): The beta parameter for the smooth L1 loss.
                                If 0, uses standard absolute value (L1 norm).
        strategy_neg (str): Strategy for handling negative values.
                            "direct": Apply OTW directly (using smooth L1 of differences).
                            "split_pos_neg": Split into positive/negative parts and sum OTWs.

    Returns:
        torch.Tensor: The OTW distance (scalar).
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Input time series 'a' and 'b' must be 1D tensors.")
    
    n = a.shape[0]
    m_ts = b.shape[0] # Length of series b

    if n != m_ts:
        # The paper's OTW_m,s formulation (eq. 8) seems to assume a common length 'n'
        # for the summation limit and for A_s(n), B_s(n).
        # For unequal lengths, the concept of A_s(n) vs B_s(n) (where n is length of 'a')
        # becomes ambiguous if 'b' has a different length M.
        # The original OT (eq. 1) handles different lengths via the transport plan T (NXM).
        # OTW's closed form (eq. 2, 6, 8) relies on cumulative sums up to a common 'n'.
        # For now, we'll require N=M. If not, one might need to resample/pad or
        # re-derive a version for unequal lengths based on the sink idea.
        raise ValueError(f"Current OTW implementation (eq. 8) assumes time series 'a' and 'b' have the same length. Got N={n}, M={m_ts}")

    # --- Strategy for negative values ---
    if strategy_neg == "split_pos_neg":
        a_pos = torch.relu(a)
        a_neg = torch.relu(-a)
        b_pos = torch.relu(b)
        b_neg = torch.relu(-b)
        
        # Recursively call with "direct" strategy for the non-negative parts
        dist_pos = otw_distance_pytorch(a_pos, b_pos, m_cost, s_window, beta_smooth_l1, strategy_neg="direct")
        dist_neg = otw_distance_pytorch(a_neg, b_neg, m_cost, s_window, beta_smooth_l1, strategy_neg="direct")
        return dist_pos + dist_neg
    
    elif strategy_neg != "direct":
        raise ValueError("Invalid strategy_neg. Choose 'direct' or 'split_pos_neg'.")

    # --- Calculate windowed cumulative sums A_s(i) and B_s(i) ---
    # (as per eq. 7 in the paper, adapted for 0-indexing)
    
    if s_window == -1 or s_window >= n: # Global cumulative sum (s=n)
        A_s = torch.cumsum(a, dim=0)
        B_s = torch.cumsum(b, dim=0)
    elif s_window < 1:
        raise ValueError("s_window must be >= 1 or -1 for global.")
    else:
        # For windowed sum, we can use 1D convolution with a kernel of ones
        # or a manual loop for clarity if preferred.
        # Using unfold for a sliding window sum:
        # This creates views, then sums over the window dimension.
        # Pad 'a' and 'b' at the beginning to handle windows at the start of the series.
        # The paper's A_s(i) = sum_{j=i-s+1 to i} a_j.
        # For 0-indexed i, this is sum_{j=i-s+1 to i} a_j
        # If i=0, window is from 1-s to 0.
        # Let's use a simpler cumsum trick for windowed sum:
        # S_window(i) = cumsum(i) - cumsum(i-window_size)
        
        # cumsum_a = torch.cumsum(a, dim=0)
        # cumsum_b = torch.cumsum(b, dim=0)
        
        # A_s = torch.zeros_like(a)
        # B_s = torch.zeros_like(b)
        
        # # A_s[k] = sum_{j=k-s+1 to k} a[j] (0-indexed)
        # for k_idx in range(n):
        #     start_idx = max(0, k_idx - s_window + 1)
        #     A_s[k_idx] = torch.sum(a[start_idx : k_idx + 1])
        #     B_s[k_idx] = torch.sum(b[start_idx : k_idx + 1])

        # More efficient: using convolution with a kernel of ones
        kernel = torch.ones(s_window, dtype=a.dtype, device=a.device)
        # Pad to ensure the output has length n and reflects sums *ending* at i
        # For 'valid' convolution, input length needs to be n + s_window - 1
        # To get sum up to index i using window s, effectively we need a convolution.
        # A simpler way that matches the paper's intention of A_s(i) = sum_{j from i-s (inclusive) to i (inclusive)} 
        # if 0-indexed, for a window of size s:
        # (or sum_{j from i-s+1 to i} for 1-indexed like paper).
        # Let's use 1D average pooling then multiply by s_window (or sum pooling if available directly)
        # Or, more directly:
        a_padded_for_conv = F.pad(a.unsqueeze(0).unsqueeze(0), (s_window - 1, 0), mode='constant', value=0)
        b_padded_for_conv = F.pad(b.unsqueeze(0).unsqueeze(0), (s_window - 1, 0), mode='constant', value=0)
        
        conv_kernel = torch.ones((1, 1, s_window), dtype=a.dtype, device=a.device)
        
        A_s_conv = F.conv1d(a_padded_for_conv, conv_kernel, stride=1, padding=0).squeeze()
        B_s_conv = F.conv1d(b_padded_for_conv, conv_kernel, stride=1, padding=0).squeeze()
        
        if A_s_conv.ndim == 0 and n == 1 : # Handle scalar case from squeeze if n=1
            A_s = A_s_conv.unsqueeze(0)
            B_s = B_s_conv.unsqueeze(0)
        else:
            A_s = A_s_conv
            B_s = B_s_conv

        if A_s.shape[0] != n or B_s.shape[0] != n:
             raise RuntimeError(f"Shape mismatch after conv for windowed sum. Expected {n}, got A_s:{A_s.shape}, B_s:{B_s.shape}")


    # --- Calculate OTW_m,s distance (eq. 8) using smooth L1 loss (eq. 10) ---
    # OTW_m,s(a,b) = m * L_beta(A_s(n) - B_s(n)) + sum_{i=1}^{n-1} L_beta(A_s(i) - B_s(i))
    # For 0-indexed tensors, A_s(n) is A_s[n-1], sum is from i=0 to n-2.
    
    diff_at_n = A_s[n-1] - B_s[n-1]
    term1 = m_cost * smooth_l1_loss(diff_at_n, beta=beta_smooth_l1)
    
    term2 = torch.tensor(0.0, device=a.device, dtype=a.dtype)
    if n > 1: # Summation term only exists if n > 1
        diff_up_to_n_minus_1 = A_s[:n-1] - B_s[:n-1]
        term2 = torch.sum(smooth_l1_loss(diff_up_to_n_minus_1, beta=beta_smooth_l1))
        
    distance = term1 + term2
    return distance

# --- Example Usage ---
if __name__ == '__main__':
    # Example time series
    ts_a_np = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.5], dtype=np.float32)
    ts_b_np = np.array([0.0, 1.0, 2.5, 3.0, 2.0, 1.0, 0.5], dtype=np.float32)
    ts_c_np = np.array([-1.0, -0.5, 0.0, 0.5, 0.0, -0.5, -1.0], dtype=np.float32)


    ts_a = torch.from_numpy(ts_a_np)
    ts_b = torch.from_numpy(ts_b_np)
    ts_c = torch.from_numpy(ts_c_np)

    print("--- Positive Time Series Examples ---")
    # Global OTW (s=n, equivalent to s_window=-1 or s_window >= n)
    dist_ab_global_abs = otw_distance_pytorch(ts_a, ts_b, m_cost=1.0, s_window=-1, beta_smooth_l1=0) # beta=0 -> abs
    dist_ab_global_smooth = otw_distance_pytorch(ts_a, ts_b, m_cost=1.0, s_window=-1, beta_smooth_l1=1.0)
    print(f"OTW(a, b) [Global, L1]: {dist_ab_global_abs.item()}")
    print(f"OTW(a, b) [Global, SmoothL1 beta=1]: {dist_ab_global_smooth.item()}")

    # Local OTW
    dist_ab_local_abs = otw_distance_pytorch(ts_a, ts_b, m_cost=1.0, s_window=3, beta_smooth_l1=0)
    dist_ab_local_smooth = otw_distance_pytorch(ts_a, ts_b, m_cost=1.0, s_window=3, beta_smooth_l1=1.0)
    print(f"OTW(a, b) [Local s=3, L1]: {dist_ab_local_abs.item()}")
    print(f"OTW(a, b) [Local s=3, SmoothL1 beta=1]: {dist_ab_local_smooth.item()}")

    # Identical series
    dist_aa_global_smooth = otw_distance_pytorch(ts_a, ts_a, m_cost=1.0, s_window=-1, beta_smooth_l1=1.0)
    print(f"OTW(a, a) [Global, SmoothL1 beta=1]: {dist_aa_global_smooth.item()}") # Should be 0

    print("\n--- Time Series with Negative Values ---")
    # Strategy 1: Direct application (eq. 10)
    dist_ac_direct = otw_distance_pytorch(ts_a, ts_c, m_cost=1.0, s_window=3, beta_smooth_l1=1.0, strategy_neg="direct")
    print(f"OTW(a, c) [Local s=3, SmoothL1 beta=1, Strategy: Direct]: {dist_ac_direct.item()}")

    # Strategy 2: Split positive and negative parts (eq. 11)
    dist_ac_split = otw_distance_pytorch(ts_a, ts_c, m_cost=1.0, s_window=3, beta_smooth_l1=1.0, strategy_neg="split_pos_neg")
    print(f"OTW(a, c) [Local s=3, SmoothL1 beta=1, Strategy: Split Pos/Neg]: {dist_ac_split.item()}")

    dist_cc_split = otw_distance_pytorch(ts_c, ts_c, m_cost=1.0, s_window=3, beta_smooth_l1=1.0, strategy_neg="split_pos_neg")
    print(f"OTW(c, c) [Local s=3, SmoothL1 beta=1, Strategy: Split Pos/Neg]: {dist_cc_split.item()}") # Should be 0

    print("\n--- Test with gradients ---")
    ts_x = torch.tensor([1.0, 2.0, 1.5], dtype=torch.float32, requires_grad=True)
    ts_y = torch.tensor([1.2, 1.8, 1.7], dtype=torch.float32)

    distance_xy = otw_distance_pytorch(ts_x, ts_y, m_cost=0.5, s_window=2, beta_smooth_l1=0.5)
    print(f"Distance (x,y) for grad test: {distance_xy.item()}")
    
    # Compute gradients
    distance_xy.backward()
    print(f"Gradient for ts_x: {ts_x.grad}")
    
    print("\n--- Test with s_window = 1 (should be like L1 norm of diffs + m_cost*L1(sum_a - sum_b)) ---")
    # When s=1, A_s(i) = a[i].
    # OTW_m,1(a,b) = m*L_beta(a[n-1]-b[n-1]) + sum_{i=0}^{n-2} L_beta(a[i]-b[i])
    # This is slightly different from pure L1(a-b) because of the m_cost on the last element's difference.
    # The paper states: "When s=1 then OTW_1,1(a,b) = ||a-b||_1". This implies m=1 for that specific statement.
    # Let's test with m_cost=1 and beta_smooth_l1=0 (for pure L1)
    
    # Recompute A_s and B_s for s=1, they are just 'a' and 'b'
    A_s_eq_1 = ts_a
    B_s_eq_1 = ts_b
    l1_term1 = 1.0 * torch.abs(A_s_eq_1[-1] - B_s_eq_1[-1]) # m_cost = 1
    l1_term2 = torch.sum(torch.abs(A_s_eq_1[:-1] - B_s_eq_1[:-1]))
    manual_otw_s1_m1_beta0 = l1_term1 + l1_term2
    
    dist_ab_s1_m1_beta0 = otw_distance_pytorch(ts_a, ts_b, m_cost=1.0, s_window=1, beta_smooth_l1=0)
    print(f"OTW(a, b) [Local s=1, m=1, L1]: {dist_ab_s1_m1_beta0.item()}")
    print(f"Manual calc for s=1,m=1,L1: {manual_otw_s1_m1_beta0.item()}") # Should be close
    
    # Check against paper's claim: OTW_1,1(a,b) = ||a-b||_1
    # This implies that m|A1(n)-B1(n)| + sum |A1(i)-B1(i)| where A1(i)=a_i
    # So, m|a_n - b_n| + sum_{i=1}^{n-1} |a_i - b_i|. If m=1, this is sum_{i=1 to n} |a_i-b_i| = L1_norm(a-b)
    l1_norm_a_b = torch.sum(torch.abs(ts_a - ts_b))
    print(f"L1 norm (a-b): {l1_norm_a_b.item()}")
    # They are indeed equal when m_cost=1 and beta_smooth_l1=0 and s_window=1
    
    # Example with single element series
    ts_single_a = torch.tensor([5.0])
    ts_single_b = torch.tensor([2.0])
    dist_single = otw_distance_pytorch(ts_single_a, ts_single_b, m_cost=2.0, s_window=1, beta_smooth_l1=0.5)
    # A_s[n-1] = A_s[0] = a[0]
    # term1 = m_cost * smooth_l1(a[0]-b[0], beta)
    # term2 = 0 (n=1)
    manual_single = 2.0 * smooth_l1_loss(torch.tensor(5.0-2.0), beta=0.5)
    print(f"OTW single element: {dist_single.item()}")
    print(f"Manual single element: {manual_single.item()}")
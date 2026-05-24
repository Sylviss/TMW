from . import utils, sinkhorn
import torch

class TMWDist(torch.nn.Module):
    """
    PyTorch loss function module for Temporal Masked Wasserstein distance.
    
    This module computes TMW loss for training neural networks on time series data.
    This version is optimized for batch processing to improve performance.
    """
    
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def cost_matrix_batch(self, xs_batch, xt_batch, cost_function="L2"):
        """
        Compute cost matrix between source and target points for a batch.
        
        Args:
            xs_batch: torch tensor of shape (b, m, d) - source points
            xt_batch: torch tensor of shape (b, n, d) - target points
            cost_function: str or callable - cost function type
            
        Returns:
            torch tensor of shape (b, m, n) - cost matrix
        """
        if cost_function == "L2":
            # Expand dimensions to use broadcasting for pairwise distance calculation
            xs_expanded = xs_batch.unsqueeze(2)  # Shape: (b, m, 1, d)
            xt_expanded = xt_batch.unsqueeze(1)  # Shape: (b, 1, n, d)
            # The following performs squared L2 distance, which is standard.
            return torch.sum((xs_expanded - xt_expanded)**2, dim=3)
        elif callable(cost_function):
            # The custom function must support batched inputs
            return cost_function(xs_batch, xt_batch)
        else:
            raise ValueError(f"Unsupported cost_function: {cost_function}")

    def mask_matrix_batch(self, xs_batch, xt_batch, mask_type, eps_threshold, rescale=False):
        """
        Compute mask matrix for temporal alignment for a batch using PyTorch.
        Supports mask types 1 and 2.
        
        Args:
            xs_batch: torch tensor of shape (b, m, d) - source time series
            xt_batch: torch tensor of shape (b, n, d) - target time series
            mask_type: int - type of masking (1 or 2)
            eps_threshold: float - threshold for binary mask
            rescale: bool - whether to rescale temporal coordinates
        Returns:
            torch tensor of shape (b, m, n) - binary mask matrix
        """
        batch_size, len_xs, _ = xs_batch.shape
        _, len_xt, _ = xt_batch.shape
        eps = 1e-10

        if mask_type == 1:
            fx_single = torch.arange(len_xs, device=self.device, dtype=torch.float32)
            fy_single = torch.arange(len_xt, device=self.device, dtype=torch.float32)
            if rescale:
                fx_single = fx_single / len_xs
                fy_single = fy_single / len_xt
            fx = fx_single.unsqueeze(0).expand(batch_size, -1)
            fy = fy_single.unsqueeze(0).expand(batch_size, -1)
                
        elif mask_type == 2:
            diff_xs = xs_batch[:, 1:] - xs_batch[:, :-1]
            diff_xt = xt_batch[:, 1:] - xt_batch[:, :-1]
            Cs = torch.linalg.norm(diff_xs, dim=2, ord=2)
            Ct = torch.linalg.norm(diff_xt, dim=2, ord=2)

            zeros = torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)
            fx = torch.cat((zeros, torch.cumsum(Cs, dim=1)), dim=1)
            fy = torch.cat((zeros, torch.cumsum(Ct, dim=1)), dim=1)

            fx_norm = fx[:, -1].unsqueeze(1)
            fy_norm = fy[:, -1].unsqueeze(1)
            fx = fx / (fx_norm + eps)
            fy = fy / (fy_norm + eps)
        else:
            raise ValueError(f"Unsupported mask_type: {mask_type}. Use 1 or 2.")
                
        diff_matrix = torch.abs(fx.unsqueeze(2) - fy.unsqueeze(1))
        
        M = (diff_matrix < eps_threshold).to(torch.float32)
        return M
    
    @torch.no_grad()
    def forward(self, y_pred_batch, y_true_batch, cost_function="L2", mask_type=1,
                reg=0.01, max_iterations=1000, thres=1e-5, eps_threshold=0.1, 
                masked=True, rescale=False):
        """
        Compute average TMW loss for a batch of predictions and ground truth.
        
        Args:
            y_pred_batch: torch tensor of shape (batch_size, sequence_length, feature_dim)
            y_true_batch: torch tensor of shape (batch_size, sequence_length, feature_dim)
            ... other parameters
            
        Returns:
            torch tensor - scalar average TMW loss
        """
        batch_size, len_pred, _ = y_pred_batch.shape
        _, len_true, _ = y_true_batch.shape
        
        # Create uniform marginal distributions for the batch
        p = torch.ones(batch_size, len_pred, device=self.device) / len_pred
        q = torch.ones(batch_size, len_true, device=self.device) / len_true
        
        # Compute cost and mask matrices for the entire batch
        C = self.cost_matrix_batch(y_pred_batch, y_true_batch, cost_function)
        
        if masked:
            M = self.mask_matrix_batch(y_pred_batch, y_true_batch, mask_type, eps_threshold, rescale)
        else:
            M = torch.ones(batch_size, len_pred, len_true, device=self.device)
            
        # Compute batched Sinkhorn
        pi = sinkhorn.sinkhorn_log_domain_refined_batched(p, q, C, M, reg, max_iterations, thres)
        
        # Compute TMW distance for the each pair in the batch and return it
        dist = torch.sum(pi * C * M, dim=(1, 2)) # Shape: (batch_size,)
        
        return dist
        
        
        


    # --- Original single-instance mask_matrix function is kept for reference or other uses ---
    def mask_matrix(self, xs, xt, mask_type, eps_threshold, rescale=False):
        """
        Compute mask matrix for temporal alignment using PyTorch.
        Supports mask types 1 and 2.
        
        Args:
            xs: torch tensor of shape (m, d) - source time series
            xt: torch tensor of shape (n, d) - target time series
            mask_type: int - type of masking (1 or 2)
            eps_threshold: float - threshold for binary mask
            rescale: bool - whether to rescale temporal coordinates
        Returns:
            torch tensor of shape (m, n) - binary mask matrix
        """
        len_xs = xs.shape[0]
        len_xt = xt.shape[0]

        if mask_type == 1:
            fx = torch.arange(len_xs, device=self.device, dtype=torch.float32)
            fy = torch.arange(len_xt, device=self.device, dtype=torch.float32)
            if rescale:
                fx = fx / len_xs
                fy = fy / len_xt
                
        elif mask_type == 2:
            diff_xs = xs[1:] - xs[:-1]
            diff_xt = xt[1:] - xt[:-1]

            Cs = torch.linalg.norm(diff_xs, dim=1, ord=2)
            Ct = torch.linalg.norm(diff_xt, dim=1, ord=2)

            zeros = torch.zeros(1, device=self.device, dtype=torch.float32)
            fx = torch.cat((zeros, torch.cumsum(Cs, dim=0)))
            fy = torch.cat((zeros, torch.cumsum(Ct, dim=0)))

            eps = 1e-10
            fx = fx / (fx[-1] + eps)
            fy = fy / (fy[-1] + eps)
        else:
            raise ValueError(f"Unsupported mask_type: {mask_type}. Use 1 or 2.")
                
        diff_matrix = torch.abs(fx.view(-1, 1) - fy)
        
        M = (diff_matrix < eps_threshold).to(torch.int)
        return M
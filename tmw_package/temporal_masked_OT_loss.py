from . import utils, sinkhorn
import torch

class TMWLoss(torch.nn.Module):
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

    def mask_matrix_batch(self, xs_batch, xt_batch, mask_type, eps_threshold, rescale=False, degree=3):
        """
        Compute mask matrix for temporal alignment for a batch using PyTorch.
        Supports multiple mask types including derivative-based masking.
        
        Args:
            xs_batch: torch tensor of shape (b, m, d) - source time series
            xt_batch: torch tensor of shape (b, n, d) - target time series
            mask_type: int or float - type of masking (1, 2, 2.1, 2.2)
            eps_threshold: float - threshold for binary mask
            rescale: bool - whether to rescale temporal coordinates
            degree: int - degree for averaging in mask type 2.2
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
                
        elif mask_type in [2, 2.1, 2.2, 3]:
            if mask_type in [2, 3]:
                diff_xs = xs_batch[:, 1:] - xs_batch[:, :-1]
                diff_xt = xt_batch[:, 1:] - xt_batch[:, :-1]
                Cs = torch.linalg.norm(diff_xs, dim=2, ord=2)
                Ct = torch.linalg.norm(diff_xt, dim=2, ord=2)
                
                zeros = torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)
                fx = torch.cat((zeros, torch.cumsum(Cs, dim=1)), dim=1)
                fy = torch.cat((zeros, torch.cumsum(Ct, dim=1)), dim=1)
            
            elif mask_type == 2.1: 
                diff_xs = xs_batch[:, 2:] - 2 * xs_batch[:, 1:-1] + xs_batch[:, :-2]
                diff_xt = xt_batch[:, 2:] - 2 * xt_batch[:, 1:-1] + xt_batch[:, :-2]
                Cs = torch.linalg.norm(diff_xs, dim=2, ord=2)
                Ct = torch.linalg.norm(diff_xt, dim=2, ord=2)
                
                zeros = torch.zeros((batch_size, 2), device=self.device, dtype=torch.float32)
                fx = torch.cat((zeros, torch.cumsum(Cs, dim=1)), dim=1)
                fy = torch.cat((zeros, torch.cumsum(Ct, dim=1)), dim=1)
            
            elif mask_type == 2.2:
                # This part is more complex but can also be vectorized
                diff_xs_sum = torch.zeros_like(xs_batch[:, 1:])
                diff_xt_sum = torch.zeros_like(xt_batch[:, 1:])
                diff_N_xs = torch.zeros_like(xs_batch[:, 1:, 0]) # Shape (b, m-1)
                diff_N_xt = torch.zeros_like(xt_batch[:, 1:, 0]) # Shape (b, n-1)

                for cur in range(1, degree + 1):
                    # Sum of previous points
                    diff_xs_sum += torch.nn.functional.pad(xs_batch, (0,0,cur,0))[:, :len_xs-1, :]
                    diff_xt_sum += torch.nn.functional.pad(xt_batch, (0,0,cur,0))[:, :len_xt-1, :]
                    # Count of points to average
                    diff_N_xs += torch.nn.functional.pad(torch.ones_like(xs_batch[:, :, 0]), (cur,0))[:, :len_xs-1]
                    diff_N_xt += torch.nn.functional.pad(torch.ones_like(xt_batch[:, :, 0]), (cur,0))[:, :len_xt-1]

                avg_prev_xs = diff_xs_sum / diff_N_xs.unsqueeze(-1)
                avg_prev_xt = diff_xt_sum / diff_N_xt.unsqueeze(-1)
                
                diff_xs = xs_batch[:, 1:] - avg_prev_xs
                diff_xt = xt_batch[:, 1:] - avg_prev_xt
                
                Cs = torch.linalg.norm(diff_xs, dim=2, ord=2)
                Ct = torch.linalg.norm(diff_xt, dim=2, ord=2)
            
                zeros = torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)
                fx = torch.cat((zeros, torch.cumsum(Cs, dim=1)), dim=1)
                fy = torch.cat((zeros, torch.cumsum(Ct, dim=1)), dim=1)

            fx_norm = fx[:, -1].unsqueeze(1)
            fy_norm = fy[:, -1].unsqueeze(1)
            fx = fx / (fx_norm + eps)
            fy = fy / (fy_norm + eps)
            
            # For mask_type 3: apply N/i weighting (emphasizes earlier points)
            if mask_type == 3:
                weight_xs = len_xs / torch.arange(1, len_xs + 1, device=self.device, dtype=torch.float32)
                weight_xt = len_xt / torch.arange(1, len_xt + 1, device=self.device, dtype=torch.float32)
                fx = fx * weight_xs.unsqueeze(0)
                fy = fy * weight_xt.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported mask_type: {mask_type}")
                
        diff_matrix = torch.abs(fx.unsqueeze(2) - fy.unsqueeze(1))
        
        M = (diff_matrix < eps_threshold).to(torch.float32)
        return M
            
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
        
        # Calculate loss for each item in the batch and average
        # Note: The original returned sum(pi * M * C). 
        # For a normalized distance, you might divide by total mass, but here we follow the original formula.
        loss_batch = torch.sum(pi * M * C, dim=[1, 2])
        
        return loss_batch.mean()


    # --- Original single-instance mask_matrix function is kept for reference or other uses ---
    def mask_matrix(self, xs, xt, mask_type, eps_threshold, rescale=False, degree=3):
        """
        Compute mask matrix for temporal alignment using PyTorch.
        Supports multiple mask types including derivative-based masking.
        
        Args:
            xs: torch tensor of shape (m, d) - source time series
            xt: torch tensor of shape (n, d) - target time series
            mask_type: int or float - type of masking (1, 2, 2.1, 2.2)
            eps_threshold: float - threshold for binary mask
            rescale: bool - whether to rescale temporal coordinates
            degree: int - degree for averaging in mask type 2.2
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
                
        elif mask_type in [2, 2.1, 3]:
            if mask_type in [2, 3]:
                diff_xs = xs[1:] - xs[:-1]
                diff_xt = xt[1:] - xt[:-1]
                start_zeros = 1
            else: # mask_type == 2.1
                diff_xs = xs[2:] - 2*xs[1:-1] + xs[:-2]
                diff_xt = xt[2:] - 2*xt[1:-1] + xt[:-2]
                start_zeros = 2

            Cs = torch.linalg.norm(diff_xs, dim=1, ord=2)
            Ct = torch.linalg.norm(diff_xt, dim=1, ord=2)

            zeros = torch.zeros(start_zeros, device=self.device, dtype=torch.float32)
            fx = torch.cat((zeros, torch.cumsum(Cs, dim=0)))
            fy = torch.cat((zeros, torch.cumsum(Ct, dim=0)))

            eps = 1e-10
            fx = fx / (fx[-1] + eps)
            fy = fy / (fy[-1] + eps)
            
            # For mask_type 3: apply N/i weighting (emphasizes earlier points)
            if mask_type == 3:
                weight_xs = len_xs / torch.arange(1, len_xs + 1, device=self.device, dtype=torch.float32)
                weight_xt = len_xt / torch.arange(1, len_xt + 1, device=self.device, dtype=torch.float32)
                fx = fx * weight_xs
                fy = fy * weight_xt
            
        elif mask_type == 2.2:
            # Averaged derivative-based coordinates
            diff_xs_sum = torch.zeros_like(xs[1:])
            diff_xt_sum = torch.zeros_like(xt[1:])
            diff_N_xs = torch.zeros_like(xs[1:, 0]) # Shape (m-1)
            diff_N_xt = torch.zeros_like(xt[1:, 0]) # Shape (n-1)

            for cur in range(1, degree + 1):
                diff_xs_sum += torch.nn.functional.pad(xs, (0,0,cur,0))[:len_xs-1, :]
                diff_xt_sum += torch.nn.functional.pad(xt, (0,0,cur,0))[:len_xt-1, :]
                diff_N_xs += torch.nn.functional.pad(torch.ones_like(xs[:,0]), (cur,0))[:len_xs-1]
                diff_N_xt += torch.nn.functional.pad(torch.ones_like(xt[:,0]), (cur,0))[:len_xt-1]
                
            avg_prev_xs = diff_xs_sum / diff_N_xs.unsqueeze(-1)
            avg_prev_xt = diff_xt_sum / diff_N_xt.unsqueeze(-1)

            diff_xs = xs[1:] - avg_prev_xs
            diff_xt = xt[1:] - avg_prev_xt
            Cs = torch.linalg.norm(diff_xs, dim=1, ord=2)
            Ct = torch.linalg.norm(diff_xt, dim=1, ord=2)
            
            zero = torch.tensor([0.0], device=self.device, dtype=torch.float32)
            fx = torch.cat((zero, torch.cumsum(Cs, dim=0)))
            fy = torch.cat((zero, torch.cumsum(Ct, dim=0)))                

            eps = 1e-10
            fx = fx / (fx[-1] + eps)
            fy = fy / (fy[-1] + eps)
        else:
            raise ValueError(f"Unsupported mask_type: {mask_type}")
                
        diff_matrix = torch.abs(fx.view(-1, 1) - fy)
        
        M = (diff_matrix < eps_threshold).to(torch.int)
        return M
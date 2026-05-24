from . import utils, linearprog, sinkhorn
import numpy as np

class TemporalMaskedWasserstein(object):
    """
    Temporal Masked Wasserstein (TMW) distance for time series data.
    
    This class implements the temporal masked optimal transport framework
    for comparing time series data with temporal alignment constraints.
    """
    
    def __init__(self):
        super().__init__()

    def cost_matrix(self, xs, xt, cost_function="L2", eps=1e-10):
        """
        Compute cost matrix between source and target points.
        
        Args:
            xs: numpy array of shape (m, d) - source points
            xt: numpy array of shape (n, d) - target points
            cost_function: str or callable - cost function type
            eps: float - small value to avoid numerical issues
            
        Returns:
            numpy array of shape (m, n) - cost matrix
        """
        if cost_function == "L2":
            return utils.cost_matrix_numpy(xs, xt)
        elif cost_function == "cosine":
            xs_norm = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + eps)
            xt_norm = xt / (np.linalg.norm(xt, axis=1, keepdims=True) + eps)
            return 0.5 * utils.cost_matrix_numpy(xs_norm, xt_norm)
        elif callable(cost_function):
            return cost_function(xs, xt)
        else:
            raise ValueError(f"Unsupported cost_function: {cost_function}")
    
    def mask_matrix(self, xs, xt, mask_type, eps_threshold, rescale=False):
        """
        Compute mask matrix for temporal alignment.
        
        Args:
            xs: numpy array of shape (m, d) - source time series
            xt: numpy array of shape (n, d) - target time series
            mask_type: int - type of masking (1 or 2)
            eps_threshold: float - threshold for binary mask
            rescale: bool - whether to rescale temporal coordinates
            
        Returns:
            numpy array of shape (m, n) - binary mask matrix
        """
        # mask matrix type 1: uniform temporal coordinates
        if mask_type == 1:
            fx = np.arange(len(xs), dtype=float)
            fy = np.arange(len(xt), dtype=float)
            if rescale:
                fx = fx / len(xs)
                fy = fy / len(xt)
        # mask matrix type 2: cumulative distance-based coordinates
        elif mask_type == 2:
            Cs = np.linalg.norm(xs[:-1] - xs[1:], axis=1)
            Ct = np.linalg.norm(xt[:-1] - xt[1:], axis=1)
            fx = np.concatenate(([0], np.cumsum(Cs)))
            fy = np.concatenate(([0], np.cumsum(Ct)))
            eps = 1e-10
            fx /= (fx[-1] + eps)
            fy /= (fy[-1] + eps)
        # mask matrix type 3: type 2 weighted by N/i (emphasizes earlier points)
        elif mask_type == 3:
            Cs = np.linalg.norm(xs[:-1] - xs[1:], axis=1)
            Ct = np.linalg.norm(xt[:-1] - xt[1:], axis=1)
            fx = np.concatenate(([0], np.cumsum(Cs)))
            fy = np.concatenate(([0], np.cumsum(Ct)))
            eps = 1e-10
            fx /= (fx[-1] + eps)
            fy /= (fy[-1] + eps)
            # Apply N/i weighting (i is 1-based index)
            N_xs = len(xs)
            N_xt = len(xt)
            weight_xs = N_xs / np.arange(1, N_xs + 1, dtype=float)
            weight_xt = N_xt / np.arange(1, N_xt + 1, dtype=float)
            fx = fx * weight_xs
            fy = fy * weight_xt
        else:
            raise ValueError(f"Unsupported mask_type: {mask_type}")
            
        diff_matrix = np.abs(fx.reshape(-1, 1) - fy)
        M = (diff_matrix < eps_threshold).astype(int)
        return M
            
        
    def tmw(self,p,q,xs,xt,cost_function="L2",mask_type=1,algorithm="linear_programming",normalized=True,
               reg=0.0001,max_iterations=100000,thres=1e-5,eps=1e-10,eps_threshold=0.1,masked=True,rescale=False):
        '''
        :param p: ndarray, (m,), Mass of source samples
        :param q: ndarray, (n,), Mass of target samples
        :param xs: ndarray, (m,d), d-dimensional source samples
        :param xt: ndarray, (n,d), d-dimensional target samples
        :param K: list of tuples, e.g., [(0,1),(10,20)]. Each tuple is an index pair of keypoints.
        :param cost_function: str or function, type of cost function. Default is "L2". Choices should be "L2", "cosine",
        and a pre-defined function.
        :param mask_type: type of masking to use with the distance. Default is 1.
        :param algorithm: str, algorithm to solve model. Default is "linear_programming". Choices should be
        "linear_programming" and "sinkhorn".
        :param tau_s: float, source temperature for computing the relation.
        :param tau_t: float, target temperature for computing the relation.
        :param normalized: bool, whether to normalize the distance
        :param reg: float, regularization coefficient in entropic model
        :param max_iterations: int, maximum number of iterations
        :param eps: float, a small number to avoid NaN
        :param thres: float, stop criterion for sinkhorn
        :return: transport plan, (m,n)
        :eps_threshold: float, threshold for temporal mask
        '''
        ## Cost matrix
        C = self.cost_matrix(xs,xt,cost_function,eps)
        if masked:
            M = self.mask_matrix(xs,xt,mask_type,eps_threshold,rescale)
        else:
            M = np.ones([xs.shape[0],xt.shape[0]])
        ## solving model
        if algorithm == "linear_programming":
            pi = linearprog.lp(p,q,C,M)
        elif algorithm == "sinkhorn":
            pi = sinkhorn.sinkhorn_log_domain(p,q,C,M,reg,max_iterations,thres)
        else:
            raise ValueError("algorithm must be 'linear_programming' or 'sinkhorn'!")
        return pi

    def tmwd(self,p,q,xs,xt,cost_function="L2",mask_type=1,algorithm="linear_programming",normalized=True,
               reg=0.0001,max_iterations=100000,thres=1e-5,eps=1e-10,eps_threshold=0.01,masked=True,rescale=True):
        '''
        :param p: ndarray, (m,), Mass of source samples
        :param q: ndarray, (n,), Mass of target samples
        :param xs: ndarray, (m,d), d-dimensional source samples
        :param xt: ndarray, (n,d), d-dimensional target samples
        :param K: list of tuples, e.g., [(0,1),(10,20)]. Each tuple is an index pair of keypoints.
        :param cost_function: str or function, type of cost function. Default is "L2". Choices should be "L2", "cosine",
        and a pre-defined function.
        :param mask_type: type of masking to use with the distance. Default is 1.
        :param algorithm: str, algorithm to solve model. Default is "linear_programming". Choices should be
        "linear_programming" and "sinkhorn".
        :param tau_s: float, source temperature for computing the relation.
        :param tau_t: float, target temperature for computing the relation.
        :param normalized: bool, whether to normalize the distance
        :param reg: float, regularization coefficient in entropic model
        :param max_iterations: int, maximum number of iterations
        :param eps: float, a small number to avoid NaN
        :param thres: float, stop criterion for sinkhorn
        :return: transport plan, (m,n)
        :eps_threshold: float, threshold for temporal mask
        '''
        C = self.cost_matrix(xs, xt, cost_function, eps)
        if masked:
            M = self.mask_matrix(xs, xt, mask_type, eps_threshold)
        else:
            M = np.ones([xs.shape[0], xt.shape[0]])
        if algorithm == "linear_programming":
            pi = linearprog.lp(p, q, C, M)
        elif algorithm == "sinkhorn":
            pi = sinkhorn.sinkhorn_log_domain(p, q, C, M, reg, max_iterations, thres)
        else:
            raise ValueError("algorithm must be 'linear_programming' or 'sinkhorn'!")
        return np.sum(np.multiply(np.multiply(pi, M), C))
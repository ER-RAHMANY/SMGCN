import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities.helpers import spmm

def compute_chebyshev_basis(x, edge_index, deg_inv_sqrt, num_nodes, K):
    """
    Compute the Chebyshev polynomials T_0(x), T_1(x), ..., T_{K-1}(x),
    where T_0(x) = x, T_1(x) = A_norm x,
    T_k(x) = 2 A_norm T_{k-1}(x) - T_{k-2}(x).
    """
    T = [x]
    if K == 1:
        return T

    T_1 = spmm(x, edge_index, deg_inv_sqrt, num_nodes)
    T.append(T_1)

    for k in range(2, K):
        T_k = 2.0 * spmm(T[k-1], edge_index, deg_inv_sqrt, num_nodes) - T[k-2]
        T.append(T_k)
    return T

class ChebConv(nn.Module):
    """
    Single Chebyshev layer from scratch:
    out = sum_{k=0 to K-1} T_k(x) * W_k
    where T_k are Chebyshev polynomials of A_norm.
    """
    def __init__(self, in_channels, out_channels, K):
        super().__init__()
    
        self.K = K
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_channels, out_channels) * 0.01)
            for _ in range(K)
        ])

    def forward(self, x, edge_index_with_loops, deg_inv_sqrt, num_nodes):
        # Compute Chebyshev basis
        T = compute_chebyshev_basis(x, edge_index_with_loops, deg_inv_sqrt, num_nodes, self.K)

        out = 0
        for k in range(self.K):
            out = out + T[k] @ self.weights[k]
        return out


class ChebNet(nn.Module):
    """
    2-layer Chebyshev Network
    """
    def __init__(self, in_channels, hidden_channels, out_channels, K=2):
        super().__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K=K)
        self.conv2 = ChebConv(hidden_channels, out_channels, K=K)

    def forward(self, x, edge_index_with_loops, deg_inv_sqrt, num_nodes):
        x = self.conv1(x, edge_index_with_loops, deg_inv_sqrt, num_nodes)
        x = F.relu(x)
        x = self.conv2(x, edge_index_with_loops, deg_inv_sqrt, num_nodes)
        return x
    
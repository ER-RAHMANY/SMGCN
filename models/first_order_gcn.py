import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities.helpers import spmm

class FirstOrderGCN(nn.Module):
    """
    2-layer version of:
    h = X W0 + A_norm X W1  (per layer)

    We'll do:
    layer1: H1 = ReLU( X W0_1 + A_norm X W1_1 )
    layer2: H2 = ( H1 W0_2 + A_norm H1 W1_2 )

    Then apply softmax or classification outside.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Layer 1:
        self.W0_1 = nn.Parameter(torch.randn(in_channels, hidden_channels) * 0.01)
        self.W1_1 = nn.Parameter(torch.randn(in_channels, hidden_channels) * 0.01)
        # Layer 2:
        self.W0_2 = nn.Parameter(torch.randn(hidden_channels, out_channels) * 0.01)
        self.W1_2 = nn.Parameter(torch.randn(hidden_channels, out_channels) * 0.01)

    def forward(self, x, edge_index_with_loops, deg_inv_sqrt, num_nodes):
        # layer 1
        A_norm_x = spmm(x, edge_index_with_loops, deg_inv_sqrt, num_nodes)
        out1 = x @ self.W0_1 + A_norm_x @ self.W1_1
        out1 = F.relu(out1)

        # layer 2
        A_norm_out1 = spmm(out1, edge_index_with_loops, deg_inv_sqrt, num_nodes)
        out2 = out1 @ self.W0_2 + A_norm_out1 @ self.W1_2
        return out2
    
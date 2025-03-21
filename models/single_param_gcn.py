import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities.helpers import spmm

class SingleParamConv(nn.Module):
    """
    Single layer of:
      (I + A_norm) * X * W
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_channels, out_channels) * 0.01)

    def forward(self, x, edge_index_no_loops, deg_inv_sqrt, num_nodes):
        # A_norm x
        a_x = spmm(x, edge_index_no_loops, deg_inv_sqrt, num_nodes)
        # (I + A_norm) x => x + a_x
        out = x + a_x
        # then multiply by W
        out = out @ self.W

        return out


class SingleParamGCN(nn.Module):
    """
    2-layer version of:
       (I + A_norm) X W1 => ReLU => (I + A_norm) H => W2
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SingleParamConv(in_channels, hidden_channels)
        self.conv2 = SingleParamConv(hidden_channels, out_channels)

    def forward(self, x, edge_index_no_loops, deg_inv_sqrt, num_nodes):
        out1 = self.conv1(x, edge_index_no_loops, deg_inv_sqrt, num_nodes)
        out1 = F.relu(out1)
        out2 = self.conv2(out1, edge_index_no_loops, deg_inv_sqrt, num_nodes)

        return out2
    
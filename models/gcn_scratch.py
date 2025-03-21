import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities.helpers import spmm

class GCNLayer(nn.Module):
    """
    Single layer of:
     A_norm * X * W
    """
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(in_channels, out_channels) * 0.01)
    
    def forward(self, x, edge_index_with_loops, deg_inv_sqrt, num_nodes):
        # A_norm x
        a_x = spmm(x, edge_index_with_loops, deg_inv_sqrt, num_nodes)
        # multiply by W
        out = a_x @ self.W

        return out

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(in_channels, hidden_channels)
        self.gc2 = GCNLayer(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index_with_loops, deg_inv_sqrt, num_nodes):
        x = self.gc1(x, edge_index_with_loops, deg_inv_sqrt, num_nodes)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gc2(x, edge_index_with_loops, deg_inv_sqrt, num_nodes)

        return x

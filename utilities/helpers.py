import torch
from torch_scatter import scatter_add
import matplotlib.pyplot as plt

########################################################################
# Utility Functions
########################################################################

def add_self_loops(edge_index, num_nodes):
    """
    Returns an edge_index with self-loops added to each node.
    """
    device = edge_index.device
    loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    return torch.cat([edge_index, loop_index], dim=1)

def normalize_edge_index(edge_index, num_nodes, with_self_loop):
    """
    Compute D^{-0.5} for each node (based on the adjacency with self-loops).
    Returns:
        edge_index_with_loops: [2, E + N]
        deg_inv_sqrt: [num_nodes]
    """
    # Add self-loops
    if with_self_loop == True:
        edge_index_prime = add_self_loops(edge_index, num_nodes)
    else:
        edge_index_prime = edge_index

    # Compute degree
    row, _ = edge_index_prime
    deg = scatter_add(torch.ones(edge_index_prime.size(1), device=row.device),
                      row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    return edge_index_prime, deg_inv_sqrt

def spmm(x, edge_index, deg_inv_sqrt, num_nodes):
    """
    Sparse matrix multiplication 'A_norm @ x'.
    Where A_norm = D^{-0.5} A D^{-0.5},

    x: [num_nodes, in_channels]
    edge_index_with_loops: [2, E]
    deg_inv_sqrt: [num_nodes]
    num_nodes: int

    Returns:
        out: [num_nodes, in_channels],  (A_norm @ x)
    """
    row, col = edge_index
    weighted_x_row = deg_inv_sqrt[row].unsqueeze(-1) * x[row]

    out = scatter_add(deg_inv_sqrt[col].unsqueeze(-1) * weighted_x_row,
                      col, dim=0, dim_size=num_nodes)
    return out

def plot_acc_barplot(acc_dict, dataset_name):
    models = list(acc_dict.keys())
    accuracies = list(acc_dict.values())

    plt.figure(figsize=(8, 5))
    plt.bar(models, accuracies, color='skyblue', edgecolor='black')
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title(f"Model Accuracy Comparison for {dataset_name} dataset")
    plt.ylim(0, 1)
    #plt.xticks(rotation=45)

    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=10)

    plt.show()

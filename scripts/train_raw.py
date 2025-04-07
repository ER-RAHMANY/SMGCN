import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from utilities.helpers import normalize_edge_index
from models.chebychevNet import ChebNet
from models.first_order_gcn import FirstOrderGCN
from models.single_param_gcn import SingleParamGCN
from models.gcn_scratch import GCN


def train_and_test(
    data,  # Pass the data object directly
    model_name="cheb",
    hidden_channels=16,
    epochs=200,
    lr=0.01,
    weight_decay=5e-4,
    save_model_path="trained_model.pt",
    dropout=0.2,
    split_idx=0  # Add a parameter to select the split index
):
    """
    Trains and evaluates a GCN model on the given data.

    Args:
        data: PyG Data object containing the graph.
        model_name: Name of the model to use.
        hidden_channels: Number of hidden channels.
        epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: Weight decay for optimizer.
        save_model_path: Path to save the trained model.
        dropout: Dropout rate.
        split_idx: Index of the split to use for training/validation/testing.

    Returns:
        model, final_test_accuracy
    """
    x = data.x
    y = data.y
    train_mask = data.train_mask[:, split_idx]  # Select the specified split
    val_mask = data.val_mask[:, split_idx]  # Select the specified split
    test_mask = data.test_mask[:, split_idx]  # Select the specified split
    num_nodes = x.size(0)
    in_channels = x.size(1)
    # Infer number of classes from the dataset
    out_channels = int(data.y.max()) + 1

    # Debug prints
    print("Number of classes:", out_channels)
    print("Shape of y:", y.shape)
    print("Shape of train_mask:", train_mask.shape)

    # Normalize adjacency once (for A_norm)
    edge_index_with_loops, deg_inv_sqrt_with_loops = normalize_edge_index(
        data.edge_index, num_nodes, with_self_loop=True)
    edge_index_no_loops, deg_inv_sqrt_no_loops = normalize_edge_index(
        data.edge_index, num_nodes, with_self_loop=False)

    # Build the chosen model
    if model_name == "chebychev_approx":
        model = ChebNet(in_channels, hidden_channels, out_channels, K=2)
        edge_index = edge_index_no_loops
        deg_inv_sqrt = deg_inv_sqrt_no_loops
    elif model_name == "first_order":
        model = FirstOrderGCN(in_channels, hidden_channels, out_channels)
        edge_index = edge_index_no_loops
        deg_inv_sqrt = deg_inv_sqrt_no_loops
    elif model_name == "single_param":
        model = SingleParamGCN(in_channels, hidden_channels, out_channels)
        edge_index = edge_index_no_loops
        deg_inv_sqrt = deg_inv_sqrt_no_loops
    elif model_name == "gcn_renormalized":
        model = GCN(in_channels, hidden_channels,
                    out_channels, dropout=dropout)
        edge_index = edge_index_with_loops
        deg_inv_sqrt = deg_inv_sqrt_with_loops
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Move everything to CPU or GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    edge_index = edge_index.to(device)
    deg_inv_sqrt = deg_inv_sqrt.to(device)
    model.to(device)

    # Set up optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(x, edge_index, deg_inv_sqrt, num_nodes)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(x, edge_index, deg_inv_sqrt, num_nodes)
                val_loss = F.cross_entropy(logits[val_mask], y[val_mask])
                val_pred = logits[val_mask].argmax(dim=1)
                val_acc = (val_pred == y[val_mask]).float().mean().item()
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
                  f"ValLoss: {val_loss.item():.4f} | ValAcc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_model_path)

    # Test
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, deg_inv_sqrt, num_nodes)
        test_pred = logits[test_mask].argmax(dim=1)
        test_acc = (test_pred == y[test_mask]).float().mean().item()
    print(f"Final Test Accuracy: {test_acc:.4f}")

    return model, test_acc

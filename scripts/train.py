import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from utilities.helpers import normalize_edge_index
from models.chebychevNet import ChebNet
from models.first_order_gcn import FirstOrderGCN
from models.single_param_gcn import SingleParamGCN
from models.gcn_scratch import GCN

def train_and_test(
    model_name="cheb",
    dataset_name="Cora",
    K=2,
    hidden_channels=16,
    epochs=200,
    lr=0.01,
    weight_decay=5e-4,
    save_model_path="trained_model.pt"
):
    """
    1) Loads a Planetoid dataset (Cora, CiteSeer, PubMed).
    2) Creates one of the GCN variants from scratch:
       - "cheb" => ChebNetScratch with K = 2 or 3 (etc.)
       - "first_order" => X W0 + A_norm X W1
       - "single_param" => (I + A_norm) X W
       - "gcn_renormalized" => A_hat_norm X W
    3) Trains and evaluates it, then saves the model to 'save_model_path'.

    Returns:
        model, final_test_accuracy
    """

    # Load dataset
    dataset = Planetoid(root=f"data/{dataset_name}", name=dataset_name)
    data = dataset[0]
    x = data.x
    y = data.y
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    num_nodes = x.size(0)
    in_channels = x.size(1)
    out_channels = dataset.num_classes

    # Normalize adjacency once (for A_norm)
    edge_index_with_loops, deg_inv_sqrt_with_loops = normalize_edge_index(data.edge_index, num_nodes, with_self_loop=True)
    edge_index_no_loops, deg_inv_sqrt_no_loops = normalize_edge_index(data.edge_index, num_nodes, with_self_loop=False)

    # Build the chosen model
    if model_name == "chebychev_approx":
        model = ChebNet(in_channels, hidden_channels, out_channels, K=K)
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
        model = GCN(in_channels, hidden_channels, out_channels)
        edge_index = edge_index_with_loops
        deg_inv_sqrt = deg_inv_sqrt_with_loops
    else:
        raise ValueError(f"Unknown model_name: {model_name}. choose one of:[chebychev_approx, first_order, single_param, gcn_renormalized]")

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

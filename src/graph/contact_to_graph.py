import numpy as np
import torch
from torch_geometric.data import Data


def contact_matrix_to_graph(
    contact_matrix,
    feature_dim=8,
    threshold=0.1
):
    """
    Convert a contact matrix into a graph representation
    suitable for a GNN.
    """

    if isinstance(contact_matrix, torch.Tensor):
        contact_matrix = contact_matrix.detach().cpu().numpy()

    n = contact_matrix.shape[0]

    # Use numpy to find edges efficiently
    edges = np.argwhere(contact_matrix > threshold)

    # Keep only upper triangle to avoid duplicates
    edges = edges[edges[:, 0] < edges[:, 1]]

    if len(edges) == 0:
        edges = np.array([[0, 1]])

    # Convert to PyG format
    edge_index = torch.tensor(
        edges.T,
        dtype=torch.long
    )

    # Make undirected
    edge_index = torch.cat([
        edge_index,
        edge_index.flip(0)
    ], dim=1)

    # Node features
    x = torch.zeros(
        (n, feature_dim),
        dtype=torch.float32
    )

    # Feature 0: normalized position
    x[:, 0] = torch.arange(n) / n

    # Feature 1: normalized degree
    degree = torch.tensor(
        np.sum(contact_matrix > threshold, axis=1)
    )
    x[:, 1] = degree / n

    # Feature 2-3: periodic encoding
    theta = 2 * np.pi * torch.arange(n) / n
    x[:, 2] = torch.sin(theta)
    x[:, 3] = torch.cos(theta)

    # Remaining features: zero (can be extended)
    # x[:, 4:] = 0.0

    return Data(
        x=x,
        edge_index=edge_index
    )

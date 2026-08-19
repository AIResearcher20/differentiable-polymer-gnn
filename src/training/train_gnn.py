import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch_geometric.loader import DataLoader

from src.models.physics_informed_gnn import PhysicsInformedGNN
from data.generate_simulation_dataset import generate_dataset


def train_gnn(
    n_samples=300,
    n_particles=50,
    epochs=120,
    batch_size=16,
    lr=1e-3
):
    graphs, targets = generate_dataset(
        n_samples=n_samples,
        n_particles=n_particles
    )

    train_idx, test_idx = train_test_split(
        range(len(graphs)),
        test_size=0.2,
        random_state=42
    )

    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]

    train_targets = targets[train_idx]
    test_targets = targets[test_idx]

    scaler = StandardScaler()

    train_y = torch.tensor(
        scaler.fit_transform(train_targets.numpy()),
        dtype=torch.float32
    )

    test_y = torch.tensor(
        scaler.transform(test_targets.numpy()),
        dtype=torch.float32
    )

    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True
    )

    model = PhysicsInformedGNN(
        input_dim=8,
        hidden_dim=64,
        output_dim=3
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    criterion = nn.MSELoss()

    for epoch in range(epochs):

        model.train()
        total_loss = 0.0

        for batch in train_loader:

            optimizer.zero_grad()

            prediction = model(batch)

            # Get the unique graph indices in this batch
            graph_indices = batch.batch.unique().sort()[0]

            # Select the corresponding target values
            loss = criterion(
                prediction,
                train_y[graph_indices]
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {total_loss / len(train_loader):.4f}"
            )

    return model, scaler, test_graphs, test_y


if __name__ == "__main__":
    model, scaler, test_graphs, test_y = train_gnn()
    print("Training complete.")

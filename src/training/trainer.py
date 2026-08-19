import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import DataLoader
from src.models.chromatin_gnn import ChromatinGNN

def train_gnn(X, y, epochs=80, batch_size=16, lr=0.001):
    """
    Train ChromatinGNN on simulation data.

    Args:
        X: List of PyG Data objects (graphs)
        y: Tensor of labels [spring, attraction, noise]
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate

    Returns:
        model: Trained ChromatinGNN model
        scaler: StandardScaler fitted to the training labels
    """
    # Split data
    train_idx, test_idx = train_test_split(range(len(X)), test_size=0.2, random_state=42)
    train_X = [X[i] for i in train_idx]
    test_X = [X[i] for i in test_idx]
    train_y = y[train_idx]
    test_y = y[test_idx]

    # Normalize labels
    scaler = StandardScaler()
    train_y_norm = scaler.fit_transform(train_y.numpy())
    test_y_norm = scaler.transform(test_y.numpy())

    # DataLoader - now yields (batch, batch_y)
    train_loader = DataLoader(list(zip(train_X, train_y_norm)), batch_size=batch_size, shuffle=True)

    # Model
    model = ChromatinGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(" Training ChromatinGNN...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_data, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_data)
            # Ensure batch_y is a tensor and matches pred size
            batch_y_tensor = torch.tensor(batch_y, dtype=torch.float32)
            loss = criterion(pred, batch_y_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: Loss = {epoch_loss/len(train_loader):.4f}")

    return model, scaler

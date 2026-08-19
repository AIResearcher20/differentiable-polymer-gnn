import numpy as np
import torch
import pickle
from pathlib import Path

from src.simulator.differentiable_polymer import DifferentiablePolymerSimulator
from src.graph.contact_to_graph import contact_matrix_to_graph


def generate_dataset(
    n_samples=300,
    n_particles=50,
    seed=42,
    save_path=None
):
    rng = np.random.default_rng(seed)

    graphs = []
    parameters = []

    for i in range(n_samples):

        spring = rng.uniform(0.3, 2.5)
        attraction = rng.uniform(0.1, 1.2)
        noise = rng.uniform(0.02, 0.25)

        simulator = DifferentiablePolymerSimulator(
            n_particles=n_particles,
            spring_constant=spring,
            attraction_force=attraction,
            noise=noise
        )

        with torch.no_grad():
            positions = simulator(n_steps=50)
            contact = simulator.compute_contact_matrix(positions)

        graph = contact_matrix_to_graph(contact, threshold=0.1)

        graphs.append(graph)
        parameters.append([spring, attraction, noise])

    targets = torch.tensor(parameters, dtype=torch.float32)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump((graphs, targets), f)

    return graphs, targets


if __name__ == "__main__":
    X, y = generate_dataset()
    print(f"Samples: {len(X)}")
    print(f"Targets: {y.shape}")
    print(f"Target range: spring={y[:,0].min():.2f}–{y[:,0].max():.2f}, "
          f"attraction={y[:,1].min():.2f}–{y[:,1].max():.2f}, "
          f"noise={y[:,2].min():.2f}–{y[:,2].max():.2f}")

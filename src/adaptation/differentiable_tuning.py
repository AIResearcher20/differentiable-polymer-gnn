import torch
import torch.nn.functional as F

from src.simulator.differentiable_polymer import DifferentiablePolymerSimulator


def fine_tune_simulator(
    contact_matrix_target,
    n_iterations=80,
    lr=0.01,
    n_steps=40,
    early_stopping_patience=20
):
    """
    Optimize differentiable polymer parameters
    against a target contact matrix.

    The target can be synthetic or realistic Hi-C-like data.

    Returns:
        simulator: fine-tuned simulator instance
        losses: list of MSE loss values
        parameter_history: list of parameter dicts
    """

    target = torch.tensor(contact_matrix_target, dtype=torch.float32)
    n_particles = target.shape[0]

    simulator = DifferentiablePolymerSimulator(
        n_particles=n_particles,
        spring_constant=1.0,
        attraction_force=0.5,
        noise=0.1
    )

    optimizer = torch.optim.Adam([
        simulator.spring_constant,
        simulator.attraction_force,
        simulator.noise
    ], lr=lr)

    losses = []
    parameter_history = []

    best_loss = float('inf')
    patience_counter = 0

    for iteration in range(n_iterations):

        optimizer.zero_grad()

        positions = simulator(n_steps=n_steps)
        simulated_contacts = simulator.compute_contact_matrix(positions)

        loss = F.mse_loss(simulated_contacts, target)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        losses.append(current_loss)

        parameter_history.append({
            "spring": simulator.spring_constant.item(),
            "attraction": simulator.attraction_force.item(),
            "noise": simulator.noise.item()
        })

        # Early stopping
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at iteration {iteration}")
            break

        if iteration % 20 == 0:
            print(f"Iteration {iteration:3d} | MSE={current_loss:.4f} | "
                  f"Spring={simulator.spring_constant.item():.3f} | "
                  f"Attraction={simulator.attraction_force.item():.3f} | "
                  f"Noise={simulator.noise.item():.3f}")

    return simulator, losses, parameter_history

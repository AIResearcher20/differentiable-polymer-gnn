import torch
import torch.nn as nn


class DifferentiablePolymerSimulator(nn.Module):
    """
    A proof-of-concept differentiable polymer simulator.

    The polymer is represented as a chain of particles in 3D space.
    The simulator includes differentiable parameters for:

    - Spring interactions between neighboring particles
    - Attraction between non-neighboring particles
    - Stochastic noise

    The resulting particle configuration can be converted into
    a differentiable contact matrix.
    """

    def __init__(
        self,
        n_particles,
        spring_constant=1.0,
        attraction_force=0.5,
        noise=0.1
    ):
        super().__init__()

        self.n_particles = n_particles

        # Learnable physical parameters
        self.spring_constant = nn.Parameter(
            torch.tensor(
                spring_constant,
                dtype=torch.float32
            )
        )

        self.attraction_force = nn.Parameter(
            torch.tensor(
                attraction_force,
                dtype=torch.float32
            )
        )

        self.noise = nn.Parameter(
            torch.tensor(
                noise,
                dtype=torch.float32
            )
        )

    def forward(self, n_steps=50):
        """
        Run the differentiable polymer simulation.

        Returns
        -------
        positions : torch.Tensor
            Final particle coordinates with shape
            (n_particles, 3).
        """

        # Initial 3D configuration
        positions = (
            torch.randn(
                self.n_particles,
                3
            ) * 0.5
        )

        for _ in range(n_steps):

            forces = torch.zeros_like(positions)

            # ------------------------------------------------
            # 1. Spring interaction between neighboring beads
            # ------------------------------------------------

            for i in range(self.n_particles - 1):

                diff = (
                    positions[i + 1]
                    - positions[i]
                )

                forces[i] += (
                    self.spring_constant * diff
                )

                forces[i + 1] -= (
                    self.spring_constant * diff
                )

            # ------------------------------------------------
            # 2. Attraction between non-neighboring beads
            # ------------------------------------------------

            for i in range(self.n_particles):

                for j in range(
                    i + 2,
                    self.n_particles
                ):

                    diff = (
                        positions[j]
                        - positions[i]
                    )

                    distance = (
                        torch.norm(diff)
                        + 1e-8
                    )

                    # Short-range attraction
                    if distance < 3.0:

                        force = (
                            self.attraction_force
                            * diff
                            / (distance**2 + 1e-8)
                        )

                        forces[i] += force
                        forces[j] -= force

            # ------------------------------------------------
            # 3. Stochastic perturbation
            # ------------------------------------------------

            stochastic_noise = (
                torch.randn_like(positions)
                * self.noise
            )

            # Differentiable position update
            positions = (
                positions
                + forces * 0.01
                + stochastic_noise
            )

        return positions

    def compute_contact_matrix(
        self,
        positions,
        threshold=2.0
    ):
        """
        Convert 3D polymer coordinates into a differentiable
        contact probability matrix.

        A sigmoid distance function provides a smooth
        approximation of binary contacts.
        """

        n = positions.shape[0]

        contact_matrix = torch.zeros(
            (n, n),
            dtype=positions.dtype,
            device=positions.device
        )

        for i in range(n):

            for j in range(i + 1, n):

                distance = torch.norm(
                    positions[i] - positions[j]
                )

                # Smooth contact probability
                contact = torch.sigmoid(
                    -(distance - threshold) * 10.0
                )

                contact_matrix[i, j] = contact
                contact_matrix[j, i] = contact

        return contact_matrix

    def get_parameters(self):
        """
        Return the current physical parameters.
        """

        return {
            "spring": self.spring_constant.item(),
            "attraction": self.attraction_force.item(),
            "noise": self.noise.item()
        }


# ============================================================
# Basic simulator test
# ============================================================

if __name__ == "__main__":

    simulator = DifferentiablePolymerSimulator(
        n_particles=40,
        spring_constant=1.0,
        attraction_force=0.5,
        noise=0.1
    )

    with torch.no_grad():

        positions = simulator(
            n_steps=50
        )

        contact_matrix = (
            simulator.compute_contact_matrix(
                positions
            )
        )

    print("Differentiable polymer simulator tested")
    print(
        f"Particles: {simulator.n_particles}"
    )
    print(
        f"Positions: {positions.shape}"
    )
    print(
        f"Contact matrix: {contact_matrix.shape}"
    )
    print(
        f"Parameters: {simulator.get_parameters()}"
    )

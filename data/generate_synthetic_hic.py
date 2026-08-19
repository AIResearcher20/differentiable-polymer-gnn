def generate_realistic_hic_data(
    n_bins=80,
    density=0.12,
    distance_scale=20.0,
    seed=42
):
    """
    Generate a synthetic Hi-C-like contact matrix.

    The simulation incorporates:
    - Distance-dependent contact decay
    - Sparse interactions
    - Stochastic contact strengths
    - Measurement-like noise

    Note:
        This function generates synthetic data.
        It does NOT represent experimental Hi-C measurements.
    """

    rng = np.random.default_rng(seed)

    contact_matrix = np.zeros((n_bins, n_bins), dtype=np.float32)

    for i in range(n_bins):
        for j in range(i + 1, n_bins):

            # Genomic-distance-dependent contact probability
            distance = abs(i - j)
            probability = np.exp(-distance / distance_scale) * density

            if rng.random() < probability:

                # Stochastic contact strength
                score = rng.exponential(10.0) + 1.0

                contact_matrix[i, j] = score
                contact_matrix[j, i] = score

    # Add observation-like noise
    noise = rng.normal(
        loc=0.0,
        scale=0.5,
        size=(n_bins, n_bins)
    )

    contact_matrix += noise

    # Remove negative values
    contact_matrix = np.maximum(contact_matrix, 0.0)

    # Enforce symmetry
    contact_matrix = (
        contact_matrix + contact_matrix.T
    ) / 2.0

    # Normalize
    max_value = contact_matrix.max()

    if max_value > 0:
        contact_matrix /= max_value

    return contact_matrix.astype(np.float32)


# ============================================================
# Generate synthetic target contact matrix
# ============================================================

N_BINS = 80

contact_matrix_real = generate_realistic_hic_data(
    n_bins=N_BINS,
    density=0.12,
    distance_scale=20.0,
    seed=SEED
)

print(" Synthetic Hi-C-like contact matrix generated")
print(f"Shape: {contact_matrix_real.shape}")
print(
    f"Density (>0.01): "
    f"{np.mean(contact_matrix_real > 0.01):.2%}"
)
print(f"Minimum: {contact_matrix_real.min():.4f}")
print(f"Maximum: {contact_matrix_real.max():.4f}")
print(f"Mean: {contact_matrix_real.mean():.4f}")
print(f"Std: {contact_matrix_real.std():.4f}")


# ============================================================
# Visualization
# ============================================================

plt.figure(figsize=(8, 6))

plt.imshow(
    contact_matrix_real,
    cmap="hot",
    interpolation="nearest"
)

plt.colorbar(label="Normalized Contact Frequency")

plt.title(
    "Synthetic Hi-C-like Contact Matrix"
)

plt.xlabel("Genomic Bin")
plt.ylabel("Genomic Bin")

plt.tight_layout()
plt.show()

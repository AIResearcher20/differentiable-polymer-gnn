import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_contact_matrix(
    matrix,
    title="Contact Matrix",
    max_bins=40,
    save_path=None
):
    matrix = np.asarray(matrix)

    plt.figure(figsize=(7, 6))
    plt.imshow(
        matrix[:max_bins, :max_bins],
        cmap="hot",
        interpolation="nearest"
    )
    plt.colorbar(label="Contact Frequency")
    plt.title(title)
    plt.xlabel("Genomic Position (bins)")
    plt.ylabel("Genomic Position (bins)")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_adaptation_loss(losses, save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Differentiable Parameter Tuning")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_parameter_history(history, save_path=None):
    spring = [x["spring"] for x in history]
    attraction = [x["attraction"] for x in history]
    noise = [x["noise"] for x in history]

    plt.figure(figsize=(9, 5))

    plt.plot(spring, label="Spring")
    plt.plot(attraction, label="Attraction")
    plt.plot(noise, label="Noise")

    plt.xlabel("Iteration")
    plt.ylabel("Parameter Value")
    plt.title("Physical Parameter Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_scatter_predictions(
    predictions,
    targets,
    parameter_names=None,
    save_path=None
):
    if parameter_names is None:
        parameter_names = ["Spring", "Attraction", "Noise"]

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, name in enumerate(parameter_names):
        ax = axes[i]
        ax.scatter(targets[:, i], predictions[:, i], alpha=0.5)
        ax.plot([targets[:, i].min(), targets[:, i].max()],
                [targets[:, i].min(), targets[:, i].max()],
                'r--', label='Perfect Prediction')
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(f"{name} Prediction")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_error_map(
    matrix1,
    matrix2,
    title="Error Map",
    max_bins=40,
    save_path=None
):
    matrix1 = np.asarray(matrix1)
    matrix2 = np.asarray(matrix2)

    error = matrix1 - matrix2

    plt.figure(figsize=(7, 6))
    plt.imshow(
        error[:max_bins, :max_bins],
        cmap="coolwarm",
        interpolation="nearest"
    )
    plt.colorbar(label="Difference")
    plt.title(title)
    plt.xlabel("Genomic Position (bins)")
    plt.ylabel("Genomic Position (bins)")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

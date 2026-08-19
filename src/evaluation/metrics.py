import numpy as np
import json
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_parameter_predictions(
    predictions,
    targets,
    parameter_names=None,
    save_path=None
):
    """
    Evaluate predicted physical parameters.
    """

    if parameter_names is None:
        parameter_names = ["Spring", "Attraction", "Noise"]

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    results = {}

    print("\n" + "=" * 60)
    print("Parameter Prediction Results")
    print("=" * 60)

    for i, name in enumerate(parameter_names):

        mae = mean_absolute_error(targets[:, i], predictions[:, i])
        mse = mean_squared_error(targets[:, i], predictions[:, i])
        r2 = r2_score(targets[:, i], predictions[:, i])

        results[name] = {"MAE": mae, "MSE": mse, "R2": r2}

        print(f"\n{name}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  R² : {r2:.4f}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return results


def compare_contact_matrices(target, simulated):
    """
    Compare target and simulated contact matrices.
    """

    target = np.asarray(target)
    simulated = np.asarray(simulated)

    if target.shape != simulated.shape:
        raise ValueError(f"Shape mismatch: {target.shape} vs {simulated.shape}")

    difference = simulated - target

    results = {
        "MSE": float(np.mean(difference ** 2)),
        "MAE": float(np.mean(np.abs(difference))),
        "max_error": float(np.max(np.abs(difference)))
    }

    print("\n" + "=" * 60)
    print("Contact Matrix Comparison")
    print("=" * 60)
    print(f"MSE: {results['MSE']:.4f}")
    print(f"MAE: {results['MAE']:.4f}")
    print(f"Max Error: {results['max_error']:.4f}")

    return results

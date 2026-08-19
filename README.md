# Differentiable Polymer–GNN for Chromatin Contact Modeling

A proof-of-concept framework combining a differentiable polymer simulator
with a Graph Neural Network (GNN) for modeling chromatin-like contact
patterns and inferring underlying biophysical parameters.

The project explores differentiable simulation, graph-based learning,
parameter inference, and simulation-to-target adaptation using synthetic
Hi-C-like contact matrices.

> **Status:** Research prototype / proof of concept.


## Research Overview

Chromatin organization can be viewed as a physical process in which
polymer interactions generate higher-order three-dimensional structure.

This project investigates whether a differentiable polymer model can be
coupled with graph-based machine learning to learn relationships between
chromatin-like contact patterns and underlying physical parameters.

The prototype focuses on three parameters:

- **Spring constant**
- **Attraction strength**
- **Noise level**

The workflow consists of:

1. Differentiable polymer simulation
2. Contact-map generation
3. Graph construction from contact matrices
4. Three-layer Graph Convolutional Network (GCN)
5. Biophysical parameter prediction
6. Gradient-based parameter tuning against a target contact matrix
7. Quantitative evaluation and visualization

## Key Components

| Component | Implementation |
|---|---|
| Differentiable simulation | PyTorch |
| Graph representation | PyTorch Geometric |
| GNN architecture | 3-layer GCN |
| Trainable parameters | 11,075 |
| Simulation data | Synthetic |
| Contact representation | Hi-C-like contact matrices |
| Parameter adaptation | Gradient-based optimization |
| Evaluation | MSE, MAE, R² |


## Methodology

The project consists of four main components:

1. **Differentiable Polymer Simulator**
   - Simulates a coarse-grained polymer chain in 3D.
   - Uses differentiable force calculations.
   - Physical parameters include:
     - Spring constant
     - Attraction strength
     - Noise amplitude

2. **Contact-Matrix Construction**
   - Pairwise particle distances are converted into soft contact probabilities using a sigmoid function.
   - This provides a differentiable representation of polymer contacts.

3. **Graph Neural Network**
   - Contact matrices are converted into graph representations.
   - A 3-layer GCN is used to infer the underlying physical parameters.
   - The model contains 11,075 trainable parameters.

4. **Simulation-to-Target Adaptation**
   - The simulator parameters are optimized against a target contact matrix.
   - Gradient-based optimization minimizes the MSE between simulated and target contact maps.

> **Important:** The current implementation is a proof-of-concept. The target contact matrix used in the demonstrated pipeline is synthetic/realistic Hi-C-like data rather than a validated experimental Hi-C dataset.

## Results

The current proof-of-concept pipeline produced the following results:

| Component | Result |
|---|---:|
| GNN architecture | 3-layer GCN |
| Trainable parameters | 11,075 |
| Synthetic simulation samples | 300 generated |
| Samples used in final evaluation | 200 |
| Contact-map adaptation MSE | 0.927297 |
| Adapted Spring constant | 1.1334 |
| Adapted Attraction strength | 0.3980 |
| Adapted Noise | 0.2079 |

### GNN Parameter Inference

The current GNN evaluation produced:

- Spring: R² = -0.002
- Attraction: R² = -6.255
- Noise: R² = -83.565

These results indicate that the current GNN does **not yet provide reliable parameter inference** and should be considered a baseline/prototype rather than a validated predictive model.

### Interpretation

The strongest demonstrated component of the project is the differentiable simulation and gradient-based parameter adaptation pipeline.

The GNN component currently serves primarily as an architectural proof-of-concept and requires further training and evaluation improvements before quantitative claims of accurate parameter inference can be made.

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

# Differentiable Polymer–GNN for Chromatin Contact Modeling


![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![PyG](https://img.shields.io/badge/PyTorch%20Geometric-GNN-orange)
![Scientific%20ML](https://img.shields.io/badge/Scientific-ML-purple)
![Differentiable Simulation](https://img.shields.io/badge/Differentiable-Simulation-green)
![Computational Genomics](https://img.shields.io/badge/Computational-Genomics-brightgreen)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)
![License](https://img.shields.io/badge/License-MIT-black)


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



## Limitations & Future Work

This implementation is intentionally a proof-of-concept and has several limitations.

### Current Limitations

- The demonstrated target contact map is **synthetic/realistic Hi-C-like data**, not experimentally validated Hi-C data.
- The polymer simulator is a simplified coarse-grained model and does not implement full molecular dynamics.
- Langevin dynamics is not currently implemented.
- The GNN currently shows poor quantitative parameter-inference performance.
- The domain-adaptation objective currently achieves an MSE of approximately **0.9273**, indicating substantial room for improvement.
- A complete end-to-end reproducibility pipeline has not yet been implemented.

### Future Work

Planned extensions include:

1. Integration of validated experimental Hi-C datasets.
2. Implementation of Langevin dynamics and more physically realistic polymer interactions.
3. Improved graph representations and edge features.
4. Better GNN training and validation strategies.
5. Simulation-to-real domain adaptation using experimental chromatin data.
6. Reproducible configuration, training, evaluation, and checkpointing pipelines.
7. Evaluation across multiple cell types and genomic regions.


## Reproducibility

The experiments were developed and tested in a Google Colab environment using Python and PyTorch.

### Main Dependencies

- Python
- PyTorch
- PyTorch Geometric
- NumPy
- SciPy
- scikit-learn
- pandas
- Matplotlib
- seaborn
- cooler
- bioframe

### Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install numpy matplotlib seaborn scikit-learn pandas
pip install bioframe cooler

Experimental Workflow
The main workflow is:
Generate or load a target Hi-C-like contact matrix.
Simulate a differentiable polymer system.
Convert contact maps into graph representations.
Train a GCN-based parameter-inference model on simulated data.
Optimize simulator parameters against the target contact matrix.
Compare simulated and target contact maps.
Report quantitative metrics and visualizations.
Reproducibility note: Exact numerical results may vary because the current implementation uses stochastic initialization and noise. Random seeds should be explicitly controlled in a future version.
## Project Structure

```text
differentiable-polymer-gnn/
│
├── README.md
│
├── notebooks/
│   └── differentiable_polymer_gnn.ipynb
│
├── src/
│   ├── polymer_simulator.py
│   ├── graph_utils.py
│   └── gnn_model.py
│
├── results/
│   ├── figures/
│   └── metrics/
│
├── requirements.txt
│
└── LICENSE

Main Components
polymer_simulator.py — differentiable coarse-grained polymer simulation.
graph_utils.py — conversion of contact matrices into graph representations.
gnn_model.py — GCN-based physical parameter inference model.
notebooks/ — experimental workflow and demonstrations.
results/ — generated figures and evaluation metrics.
The current repository may initially contain the implementation primarily as a notebook. The modular src/ structure represents the intended organization for further development

## Citation

If you use this project in research or experimentation, please cite the repository:

```bibtex
@software{Moafi_differentiable_polymer_gnn,
  author = {Sepideh, Moafi},
  title = {Differentiable Polymer Simulation with Graph Neural Networks},
  year = {2026},
  note = {Proof-of-concept research software},
}

Author
Sepideh Moafi 
AI/ML Researcher working at the intersection of:
Differentiable simulation
Scientific machine learning
Polymer and chromatin physics
Graph neural networks
Computational genomics
This project is part of an ongoing exploration of simulation-supervised learning and differentiable modeling for biological systems.


## License

This project is released under the MIT License.

See the `LICENSE` file for the full license text.

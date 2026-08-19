import numpy as np
import torch
from src.simulator.differentiable_polymer import DifferentiablePolymerSimulator
from src.graph.contact_to_graph import contact_matrix_to_graph

def generate_realistic_hic_data(n_bins=80, density=0.12):
    """
    Generate realistic Hi-C-like contact matrix with:
    - Power-law distance decay
    - Gaussian noise
    """
    np.random.seed(42)
    contact = np.zeros((n_bins, n_bins))
    
    for i in range(n_bins):
        for j in range(i+1, n_bins):
            prob = np.exp(-abs(i-j)/20) * density
            if np.random.random() < prob:
                score = np.random.exponential(10) + 1
                contact[i, j] = score
                contact[j, i] = score
    
    contact += np.random.randn(n_bins, n_bins) * 0.5
    contact = np.maximum(contact, 0)
    contact = contact / (contact.max() + 1e-8)
    
    return contact

def generate_dataset(n_samples=200, n_particles=40, n_steps=30):
    """Generate simulation dataset with diverse physical parameters"""
    X_list, y_list = [], []
    
    for i in range(n_samples):
        if i % 50 == 0:
            print(f"  Generating {i}/{n_samples}...")
        
        spring = np.random.uniform(0.3, 2.5)
        attraction = np.random.uniform(0.1, 1.2)
        noise = np.random.uniform(0.02, 0.2)
        
        sim = DifferentiablePolymerSimulator(n_particles, spring, attraction, noise)
        
        with torch.no_grad():
            positions = sim(n_steps=n_steps)
            contact = sim.compute_contact_matrix(positions)
        
        graph = contact_matrix_to_graph(contact.numpy())
        X_list.append(graph)
        y_list.append([spring, attraction, noise])
    
    return X_list, torch.tensor(y_list, dtype=torch.float32)

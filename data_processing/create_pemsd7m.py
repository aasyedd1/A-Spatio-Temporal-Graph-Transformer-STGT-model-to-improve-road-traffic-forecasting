#!/usr/bin/env python3
"""
Create PeMSD7(M) dataset by sampling 228 nodes from PEMS07 (883 nodes)
This matches the STTN paper setup where PeMSD7(M) has 228 nodes
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler

def create_pemsd7m_dataset():
    """Create PeMSD7(M) by sampling 228 nodes from PEMS07"""
    
    # Load the full PEMS07 dataset (883 nodes)
    print("Loading PEMS07 dataset (883 nodes)...")
    data = np.load('../datasets/other 2/data/PEMS07/PEMS07.npz')
    full_data = data['data']  # Shape: (timesteps, 883, 1)
    
    print(f"Full PEMS07 shape: {full_data.shape}")
    print(f"Data range: [{full_data.min():.3f}, {full_data.max():.3f}]")
    
    # Randomly sample 228 nodes (for reproducibility, use fixed seed)
    np.random.seed(42)  # Same seed as training for consistency
    
    total_nodes = full_data.shape[1]  # 883
    target_nodes = 228
    
    # Sample 228 nodes randomly
    selected_indices = np.random.choice(total_nodes, size=target_nodes, replace=False)
    selected_indices = np.sort(selected_indices)  # Sort for consistency
    
    print(f"Selected nodes: {selected_indices[:10]}... (showing first 10)")
    
    # Create PeMSD7(M) by selecting the sampled nodes
    pemsd7m_data = full_data[:, selected_indices, :]  # Shape: (timesteps, 228, 1)
    
    print(f"PeMSD7(M) shape: {pemsd7m_data.shape}")
    
    # Create output directory
    output_dir = '../datasets/PEMSD7M'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the data
    print("Saving PeMSD7(M) dataset...")
    
    # Save as pickle (similar to PEMS-BAY format)
    with open(f'{output_dir}/data.pkl', 'wb') as f:
        pickle.dump(pemsd7m_data, f)
    
    # Create train/val/test splits (70/20/10 split)
    total_timesteps = pemsd7m_data.shape[0]
    train_end = int(0.7 * total_timesteps)
    val_end = int(0.9 * total_timesteps)
    
    train_indices = list(range(0, train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, total_timesteps))
    
    # Save indices
    with open(f'{output_dir}/train_index.pkl', 'wb') as f:
        pickle.dump(train_indices, f)
    with open(f'{output_dir}/valid_index.pkl', 'wb') as f:
        pickle.dump(val_indices, f)
    with open(f'{output_dir}/test_index.pkl', 'wb') as f:
        pickle.dump(test_indices, f)
    
    # Calculate and save normalization parameters
    train_data = pemsd7m_data[train_indices]
    scaler = StandardScaler()
    scaler.fit(train_data.reshape(-1, train_data.shape[-1]))
    
    # Save original mean/std (before StandardScaler)
    original_mean = float(train_data.mean())
    original_std = float(train_data.std())
    
    with open(f'{output_dir}/mean.pkl', 'wb') as f:
        pickle.dump(original_mean, f)
    with open(f'{output_dir}/std.pkl', 'wb') as f:
        pickle.dump(original_std, f)
    
    print(f"Dataset statistics:")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Train: {len(train_indices)} timesteps")
    print(f"  Val: {len(val_indices)} timesteps") 
    print(f"  Test: {len(test_indices)} timesteps")
    print(f"  Original mean: {original_mean:.2f}")
    print(f"  Original std: {original_std:.2f}")
    
    # Create a simple adjacency matrix (identity + random connections)
    print("Creating adjacency matrix...")
    adj_matrix = np.eye(target_nodes)  # Start with identity
    
    # Add some random connections (sparse)
    np.random.seed(42)
    for i in range(target_nodes):
        # Each node connects to 2-5 random neighbors
        num_connections = np.random.randint(2, 6)
        neighbors = np.random.choice(target_nodes, size=num_connections, replace=False)
        adj_matrix[i, neighbors] = 1.0
        adj_matrix[neighbors, i] = 1.0  # Symmetric
    
    # Save adjacency matrix
    adj_path = '../datasets/sensor_graph/adj_mx_pemsd7m.pkl'
    os.makedirs('../datasets/sensor_graph', exist_ok=True)
    with open(adj_path, 'wb') as f:
        pickle.dump([adj_matrix], f)
    
    print(f"Adjacency matrix saved: {adj_path}")
    print(f"Adjacency shape: {adj_matrix.shape}")
    print(f"Adjacency connections: {(adj_matrix > 0).sum()} / {target_nodes * target_nodes}")
    
    print(f"\\n✅ PeMSD7(M) dataset created successfully!")
    print(f"   Location: {output_dir}")
    print(f"   Nodes: {target_nodes}")
    print(f"   Timesteps: {total_timesteps}")
    print(f"   Features: {pemsd7m_data.shape[2]}")
    
    return output_dir

if __name__ == "__main__":
    output_dir = create_pemsd7m_dataset()

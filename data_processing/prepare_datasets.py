#!/usr/bin/env python3

"""
PEMS-BAY dataset preparation script for USTGT
Only prepares PEMS-BAY dataset and splits.
"""

import os
import sys
import pickle
import numpy as np
import requests
from typing import Dict, Tuple

def create_dataset_structure():
    """Create the required dataset directory structure for PEMS-BAY only"""
    base_dir = "../datasets"
    dirs_to_create = [
        base_dir,
        f"{base_dir}/PEMS-BAY",
        f"{base_dir}/sensor_graph"
    ]
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def generate_metr_la_dummy_data():
    """Generate dummy METR-LA data for testing"""
    print("Generating dummy METR-LA dataset...")
    
    # METR-LA specifications
    num_nodes = 207
    num_timesteps = 34272  # From LSTTN README
    
    # Generate traffic data [timesteps, nodes, features]
    traffic_data = np.random.rand(num_timesteps, num_nodes, 1) * 80 + 10  # Speed 10-90 mph
    
    # Add some temporal patterns (morning/evening rush hours)
    for t in range(num_timesteps):
        hour = (t * 5 / 60) % 24  # 5-minute intervals
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            traffic_data[t] *= 0.6  # Slower traffic
        elif 22 <= hour or hour <= 6:  # Night hours
            traffic_data[t] *= 1.2  # Faster traffic
    
    # Save traffic data
    data_path = "../datasets/METR-LA/data.pkl"
    with open(data_path, 'wb') as f:
        pickle.dump(traffic_data, f)
    print(f"Saved METR-LA data: {traffic_data.shape} to {data_path}")
    
    # Generate adjacency matrix
    adj_matrix = np.random.rand(num_nodes, num_nodes)
    adj_matrix = (adj_matrix > 0.85).astype(float)  # Sparse 15% connectivity
    np.fill_diagonal(adj_matrix, 1.0)  # Self-loops
    
    # Make symmetric
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    adj_matrix = (adj_matrix > 0.5).astype(float)
    np.fill_diagonal(adj_matrix, 1.0)
    
    # Save adjacency matrix in LSTTN format (sensor_ids, sensor_id_to_ind, adj_mx)
    sensor_ids = [f"sensor_{i}" for i in range(num_nodes)]
    sensor_id_to_ind = {sensor_id: i for i, sensor_id in enumerate(sensor_ids)}
    adj_data = (sensor_ids, sensor_id_to_ind, adj_matrix)
    
    adj_path = "../datasets/sensor_graph/adj_mx_la.pkl"
    with open(adj_path, 'wb') as f:
        pickle.dump(adj_data, f)
    print(f"Saved METR-LA adjacency: {adj_matrix.shape} to {adj_path}")
    
    # Generate train/val/test indices
    generate_data_splits(num_timesteps, "../datasets/METR-LA")
    
    return traffic_data.shape, adj_matrix.shape

def generate_pems_bay_dummy_data():
    """Generate dummy PEMS-BAY data for testing"""
    print("Generating dummy PEMS-BAY dataset...")
    
    # PEMS-BAY specifications  
    num_nodes = 325
    num_timesteps = 52116  # From LSTTN README
    
    # Generate traffic data
    traffic_data = np.random.rand(num_timesteps, num_nodes, 1) * 70 + 15  # Speed 15-85 mph
    
    # Add temporal patterns
    for t in range(num_timesteps):
        hour = (t * 5 / 60) % 24
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            traffic_data[t] *= 0.7
        elif 22 <= hour or hour <= 6:  # Night hours
            traffic_data[t] *= 1.1
    
    # Save traffic data
    data_path = "../datasets/PEMS-BAY/data.pkl"
    with open(data_path, 'wb') as f:
        pickle.dump(traffic_data, f)
    print(f"Saved PEMS-BAY data: {traffic_data.shape} to {data_path}")
    
    # Generate adjacency matrix
    adj_matrix = np.random.rand(num_nodes, num_nodes)
    adj_matrix = (adj_matrix > 0.87).astype(float)  # Sparse 13% connectivity
    np.fill_diagonal(adj_matrix, 1.0)
    
    # Make symmetric
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    adj_matrix = (adj_matrix > 0.5).astype(float)
    np.fill_diagonal(adj_matrix, 1.0)
    
    # Save adjacency matrix
    sensor_ids = [f"sensor_{i}" for i in range(num_nodes)]
    sensor_id_to_ind = {sensor_id: i for i, sensor_id in enumerate(sensor_ids)}
    adj_data = (sensor_ids, sensor_id_to_ind, adj_matrix)
    
    adj_path = "../datasets/sensor_graph/adj_mx_bay.pkl"
    with open(adj_path, 'wb') as f:
        pickle.dump(adj_data, f)
    print(f"Saved PEMS-BAY adjacency: {adj_matrix.shape} to {adj_path}")
    
    # Generate train/val/test indices
    generate_data_splits(num_timesteps, "../datasets/PEMS-BAY")
    
    return traffic_data.shape, adj_matrix.shape

def generate_data_splits(num_timesteps: int, dataset_dir: str):
    """Generate train/validation/test splits"""
    seq_len = 12
    pred_len = 12
    
    # Create indices for valid sequences
    indices = []
    for i in range(seq_len, num_timesteps - pred_len):
        indices.append([i - seq_len, i, i + pred_len])
    
    indices = np.array(indices)
    num_samples = len(indices)
    
    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    
    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Save indices
    with open(f"{dataset_dir}/train_index.pkl", 'wb') as f:
        pickle.dump(train_indices, f)
    
    with open(f"{dataset_dir}/valid_index.pkl", 'wb') as f:
        pickle.dump(val_indices, f)
        
    with open(f"{dataset_dir}/test_index.pkl", 'wb') as f:
        pickle.dump(test_indices, f)
    
    print(f"Generated data splits - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

def generate_normalization_stats(dataset_name: str):
    """Generate normalization statistics for the dataset"""
    data_path = f"../datasets/{dataset_name}/data.pkl"
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Calculate mean and std for standard normalization
    mean = np.mean(data)
    std = np.std(data)
    
    # Calculate min and max for minmax normalization
    data_min = np.min(data)
    data_max = np.max(data)
    
    # Save normalization stats
    stats_dir = f"../datasets/{dataset_name}"
    
    with open(f"{stats_dir}/mean.pkl", 'wb') as f:
        pickle.dump(mean, f)
    
    with open(f"{stats_dir}/std.pkl", 'wb') as f:
        pickle.dump(std, f)
        
    with open(f"{stats_dir}/min.pkl", 'wb') as f:
        pickle.dump(data_min, f)
        
    with open(f"{stats_dir}/max.pkl", 'wb') as f:
        pickle.dump(data_max, f)
    
    print(f"Generated normalization stats for {dataset_name}:")
    print(f"  Mean: {mean:.4f}, Std: {std:.4f}")
    print(f"  Min: {data_min:.4f}, Max: {data_max:.4f}")

def download_real_datasets():
    """Download real datasets from Google Drive (if available)"""
    print("Real dataset download not implemented yet.")
    print("Please download datasets manually from:")
    print("https://drive.google.com/file/d/1GHQ071AICZW6rSsXBsjQGaUNPh047xqN/view?usp=drive_link")
    print("And extract them to ../datasets/ directory")


if __name__ == "__main__":
    print("=" * 60)
    print("USTGT PEMS-BAY Dataset Preparation")
    print("=" * 60)
    create_dataset_structure()
    pems_shape = generate_pems_bay_dummy_data()
    generate_normalization_stats("PEMS-BAY")
    print(f"\n✅ PEMS-BAY dataset preparation completed!")
    print(f"   PEMS-BAY: Traffic {pems_shape[0]}, Adjacency {pems_shape[1]}")

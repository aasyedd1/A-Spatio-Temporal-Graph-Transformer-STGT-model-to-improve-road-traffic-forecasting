"""
USTGT Training Script - RTX 4070 12GB Optimized
GPU-optimized implementation for traffic prediction
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp  # Mixed precision for RTX 4070
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import time
from typing import Dict, Tuple, Optional
import csv

# MAPE handling defaults. Can be overridden via config['metrics']:
#   mape_mode: 'masked'|'standard'|'symmetric'
#   mape_threshold: float (used for 'masked' mode)
MAPE_MODE = 'masked'
MAPE_THRESHOLD = 1.0


def write_epoch_log(csv_path: str, epoch: int, train_loss: float, val_loss: float,
                    val_metrics: Dict[str, float], lr: float, epoch_time: float):
    """Append a single epoch's metrics to a CSV file (creates header if missing)."""
    header = ['epoch', 'train_loss', 'val_loss', 'rmse', 'mae', 'mape', 'lr', 'epoch_time']
    row = [epoch, train_loss, val_loss, val_metrics.get('rmse', None),
           val_metrics.get('mae', None), val_metrics.get('mape', None), lr, epoch_time]

    write_header = not os.path.exists(csv_path)
    try:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)
    except Exception:
        # don't crash training for logging errors
        pass

# Try to import yaml, fallback to basic config if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: PyYAML not available. Using default configuration.")

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Import our unified model
from model import USTGT, create_ustgt_model

# GPU optimization settings
torch.backends.cudnn.benchmark = True  # Optimize for RTX 4070
torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed


class TrafficDataset(Dataset):
    """Dataset for traffic prediction"""
    
    def __init__(self, traffic_data, adj_matrix, seq_len=12, pred_len=3, 
                 external_features=None):
        """
        Args:
            traffic_data: [num_timesteps, num_nodes, input_dim]
            adj_matrix: [num_nodes, num_nodes]
            seq_len: Input sequence length
            pred_len: Prediction horizon
            external_features: Optional dict of external features
        """
        self.traffic_data = torch.FloatTensor(traffic_data)
        self.adj_matrix = torch.FloatTensor(adj_matrix)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.external_features = external_features
        
        self.num_samples = len(traffic_data) - seq_len - pred_len + 1
        
        # Ensure we have enough data
        if self.num_samples <= 0:
            raise ValueError(f"Not enough data: need at least {seq_len + pred_len} timesteps")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Get input sequence and target
        x = self.traffic_data[idx:idx+self.seq_len]  # [seq_len, num_nodes, input_dim]
        y = self.traffic_data[idx+self.seq_len:idx+self.seq_len+self.pred_len]  # [pred_len, num_nodes, input_dim]
        
        # Transpose to match model input format [num_nodes, seq_len, input_dim]
        x = x.permute(1, 0, 2)  # [num_nodes, seq_len, input_dim]
        y = y.permute(1, 0, 2)  # [num_nodes, pred_len, input_dim]
        
        sample = {
            'x': x,
            'y': y,
            'adj_matrix': self.adj_matrix
        }
        
        # Add external features if available
        if self.external_features is not None:
            sample['external_features'] = self.external_features
        
        return sample


def load_data(config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Load traffic data and adjacency matrix - compatible with LSTTN/STTN datasets"""
    data_dir = config['data']['data_dir']
    dataset_name = config['data']['dataset_name']
    
    # Map dataset names to match LSTTN/STTN conventions
    if dataset_name == 'METR-LA':
        data_path = os.path.join(data_dir, 'METR-LA', 'data.pkl')
        adj_path = os.path.join(data_dir, 'sensor_graph', 'adj_mx_la.pkl')
        num_nodes = 207
    elif dataset_name == 'PEMS-BAY':
        data_path = os.path.join(data_dir, 'PEMS-BAY', 'data.pkl')
        adj_path = os.path.join(data_dir, 'sensor_graph', 'adj_mx_bay.pkl')
        num_nodes = 325
    elif dataset_name == 'PEMS04':
        data_path = os.path.join(data_dir, 'PEMS04', 'data.pkl')
        adj_path = os.path.join(data_dir, 'sensor_graph', 'adj_mx_04.pkl')
        num_nodes = 307
    elif dataset_name == 'PEMS08':
        data_path = os.path.join(data_dir, 'PEMS08', 'data.pkl')
        adj_path = os.path.join(data_dir, 'sensor_graph', 'adj_mx_08.pkl')
        num_nodes = 170
    elif dataset_name == 'PEMSD7M':
        data_path = os.path.join(data_dir, 'PEMSD7M', 'data.pkl')
        adj_path = os.path.join(data_dir, 'sensor_graph', 'adj_mx_pemsd7m.pkl')
        num_nodes = 228
    else:
        # Fallback to old dataset names for backward compatibility
        if dataset_name == 'sz':
            speed_file = os.path.join(data_dir, 'sz_speed.csv')
            adj_file = os.path.join(data_dir, 'sz_adj.csv')
        elif dataset_name == 'los':
            speed_file = os.path.join(data_dir, 'los_speed.csv')
            adj_file = os.path.join(data_dir, 'Los_adj.pkl')
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Load legacy format data
        return load_legacy_data(speed_file, adj_file, config)
    
    # Load LSTTN/STTN format data
    try:
        # Load traffic data (LSTTN/STTN format)
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                traffic_data = pickle.load(f, encoding='latin-1')
            print(f"Loaded {dataset_name} traffic data: {traffic_data.shape}")
        else:
            print(f"Warning: {data_path} not found. Generating dummy data for {dataset_name}.")
            # Generate dummy data with correct dimensions
            num_timesteps = 20000 if dataset_name == 'METR-LA' else 30000
            traffic_data = np.random.rand(num_timesteps, num_nodes, 1) * 60 + 10
        
        # Load adjacency matrix
        if os.path.exists(adj_path):
            with open(adj_path, 'rb') as f:
                adj_data = pickle.load(f, encoding='latin-1')
                print(f"Loaded adjacency data type: {type(adj_data)}")
                
                # Handle different adjacency matrix formats
                if isinstance(adj_data, (tuple, list)) and len(adj_data) >= 3:
                    adj_matrix = adj_data[2]  # (sensor_ids, sensor_id_to_ind, adj_mx)
                    print(f"Extracted adjacency matrix from {type(adj_data).__name__}, type: {type(adj_matrix)}")
                elif isinstance(adj_data, (list, tuple)):
                    # Handle nested lists with different structures
                    try:
                        adj_matrix = np.array(adj_data, dtype=float)
                    except ValueError:
                        # If array conversion fails, try to handle the sparse format
                        print("Converting sparse adjacency format...")
                        adj_matrix = np.zeros((num_nodes, num_nodes))
                        for i, row in enumerate(adj_data):
                            if isinstance(row, (list, tuple)) and len(row) > 0:
                                for j, val in enumerate(row):
                                    if j < num_nodes and isinstance(val, (int, float)):
                                        adj_matrix[i, j] = float(val)
                else:
                    adj_matrix = adj_data
                    
                # Ensure adj_matrix is numpy array with correct shape
                if not isinstance(adj_matrix, np.ndarray):
                    adj_matrix = np.array(adj_matrix, dtype=float)
                    
                # Validate shape
                if adj_matrix.shape != (num_nodes, num_nodes):
                    print(f"Warning: Adjacency matrix shape {adj_matrix.shape} doesn't match expected ({num_nodes}, {num_nodes})")
                    # Create identity matrix as fallback
                    adj_matrix = np.eye(num_nodes)
                    
            print(f"Loaded {dataset_name} adjacency matrix: {adj_matrix.shape}")
        else:
            print(f"Warning: {adj_path} not found. Generating dummy adjacency for {dataset_name}.")
            # Generate dummy adjacency matrix
            adj_matrix = np.random.rand(num_nodes, num_nodes)
            adj_matrix = (adj_matrix > 0.8).astype(float)  # Sparse adjacency
            np.fill_diagonal(adj_matrix, 1.0)  # Self-loops
        
        # Ensure data is in correct format [timesteps, nodes, features]
        if len(traffic_data.shape) == 2:
            traffic_data = traffic_data[:, :, np.newaxis]
        
        return traffic_data, adj_matrix
        
    except Exception as e:
        print(f"Error loading {dataset_name} data: {e}")
        print("Falling back to dummy data generation...")
        return generate_dummy_data(config)


def load_legacy_data(speed_file: str, adj_file: str, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Load legacy format data (sz, los)"""
    # Load speed data
    if os.path.exists(speed_file):
        if speed_file.endswith('.csv'):
            speed_data = pd.read_csv(speed_file, header=None).values
        else:
            with open(speed_file, 'rb') as f:
                speed_data = pickle.load(f, encoding='latin-1')
    else:
        print(f"Warning: {speed_file} not found. Generating dummy data for testing.")
        num_timesteps = 1000
        num_nodes = config['model']['num_nodes']
        speed_data = np.random.rand(num_timesteps, num_nodes) * 60 + 10
    
    # Load adjacency matrix
    if os.path.exists(adj_file):
        if adj_file.endswith('.csv'):
            adj_matrix = pd.read_csv(adj_file, header=None).values
        else:
            with open(adj_file, 'rb') as f:
                adj_matrix = pickle.load(f, encoding='latin-1')
    else:
        print(f"Warning: {adj_file} not found. Generating dummy adjacency matrix.")
        num_nodes = config['model']['num_nodes']
        adj_matrix = np.random.rand(num_nodes, num_nodes)
        adj_matrix = (adj_matrix > 0.8).astype(float)
        np.fill_diagonal(adj_matrix, 1.0)
    
    # Ensure data is in correct format
    if len(speed_data.shape) == 2:
        speed_data = speed_data[:, :, np.newaxis]
    
    return speed_data, adj_matrix


def generate_dummy_data(config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Generate dummy data when real datasets are not available"""
    dataset_name = config['data']['dataset_name']
    
    # Use realistic dimensions based on dataset
    if dataset_name == 'METR-LA':
        num_nodes = 207
        num_timesteps = 20000
    elif dataset_name == 'PEMS-BAY':
        num_nodes = 325
        num_timesteps = 30000
    elif dataset_name == 'PEMS04':
        num_nodes = 307
        num_timesteps = 17000
    elif dataset_name == 'PEMS08':
        num_nodes = 170
        num_timesteps = 18000
    else:
        num_nodes = config['model']['num_nodes']
        num_timesteps = 1000
    
    print(f"Generating dummy {dataset_name} data: {num_timesteps} timesteps, {num_nodes} nodes")
    
    # Generate realistic traffic data
    traffic_data = np.random.rand(num_timesteps, num_nodes, 1) * 60 + 10
    
    # Generate sparse adjacency matrix
    adj_matrix = np.random.rand(num_nodes, num_nodes)
    adj_matrix = (adj_matrix > 0.8).astype(float)
    np.fill_diagonal(adj_matrix, 1.0)
    
    return traffic_data, adj_matrix


def create_data_splits(data: np.ndarray, config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/val/test sets"""
    # Use safe defaults if split keys are missing
    data_cfg = config.get('data', {})
    train_ratio = float(data_cfg.get('train_ratio', 0.7))
    val_ratio = float(data_cfg.get('val_ratio', 0.15))

    # Guard against invalid ratios
    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio >= 1.0:
        # Respect sensible defaults if config is malformed
        train_ratio = 0.7
        val_ratio = 0.15

    num_samples = len(data)
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data


def normalize_data(train_data: np.ndarray, val_data: np.ndarray, test_data: np.ndarray, 
                  config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Normalize the data"""
    if not config['data']['normalize']:
        return train_data, val_data, test_data, None
    
    scaler_type = config['data']['scaler_type']
    
    # Fit scaler on training data
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Reshape for fitting
    original_shape = train_data.shape
    train_data_flat = train_data.reshape(-1, original_shape[-1])
    scaler.fit(train_data_flat)
    
    # Transform all datasets
    train_data_norm = scaler.transform(train_data_flat).reshape(original_shape)
    
    val_data_flat = val_data.reshape(-1, val_data.shape[-1])
    val_data_norm = scaler.transform(val_data_flat).reshape(val_data.shape)
    
    test_data_flat = test_data.reshape(-1, test_data.shape[-1])
    test_data_norm = scaler.transform(test_data_flat).reshape(test_data.shape)
    
    return train_data_norm, val_data_norm, test_data_norm, scaler


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_flat = y_true_flat[mask]
    y_pred_flat = y_pred_flat[mask]
    
    if len(y_true_flat) == 0:
        return {'rmse': float('inf'), 'mae': float('inf'), 'mape': float('inf')}
    
    # Calculate metrics
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    
    # MAPE handling (configurable): 'standard', 'masked', 'symmetric'
    # Default behavior: masked mode (ignore y_true values with abs < threshold)
    try:
        # Check config if available in globals (train/test scripts set this)
        mape_mode = globals().get('METRICS_MAPE_MODE', MAPE_MODE)
        mape_threshold = float(globals().get('METRICS_MAPE_THRESHOLD', MAPE_THRESHOLD))
    except Exception:
        mape_mode = MAPE_MODE
        mape_threshold = MAPE_THRESHOLD

    mape = None
    masked_fraction = 0.0
    if mape_mode == 'symmetric' or mape_mode == 'smape':
        # sMAPE: symmetric mean absolute percentage error
        denom = (np.abs(y_true_flat) + np.abs(y_pred_flat))
        # avoid division by zero
        mask = denom > 1e-8
        if mask.sum() == 0:
            mape = float('inf')
        else:
            mape = np.mean(2.0 * np.abs(y_pred_flat[mask] - y_true_flat[mask]) / denom[mask]) * 100
            masked_fraction = 1.0 - (mask.sum() / len(denom))
    elif mape_mode == 'masked':
        # Mask out very small ground-truth values that inflate percent error
        mask = np.abs(y_true_flat) >= mape_threshold
        if mask.sum() == 0:
            mape = float('inf')
            masked_fraction = 1.0
        else:
            mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / np.abs(y_true_flat[mask]))) * 100
            masked_fraction = 1.0 - (mask.sum() / len(y_true_flat))
    else:
        # standard MAPE with a small floor to avoid division by zero
        denom = np.maximum(np.abs(y_true_flat), 1e-2)
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / denom)) * 100

    # STTN-style MAPE: mask exact zeros (v == 0), normalize mask, then compute mean(mask * |err| / v)
    # This matches the STTN implementation in STTN-main/utils/math_utils.py
    if mape_mode == 'sttn':
        # Use exact zeros mask
        mask = (y_true_flat != 0)
        if mask.sum() == 0:
            mape = float('inf')
            masked_fraction = 1.0
        else:
            mask = mask.astype('float32')
            # normalize mask so remaining entries preserve total weight
            mask = mask / np.mean(mask)
            # compute per-element percentage error using ground-truth denominator
            # avoid division by zero because zeros are masked
            mape_arr = np.abs(y_pred_flat - y_true_flat) / y_true_flat
            mape = np.mean(mask * mape_arr) * 100
            masked_fraction = 1.0 - (np.sum(y_true_flat != 0) / len(y_true_flat))
            masked_fraction = 1.0 - (np.count_nonzero(mask) / len(mask))

    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'mape_masked_fraction': masked_fraction
    }


def train_epoch_gpu_optimized(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, 
                             criterion: nn.Module, device: torch.device, scaler: amp.GradScaler,
                             gradient_clip: float = 3.0) -> float:
    """Train for one epoch with RTX 4070 GPU optimizations"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    data_iter = tqdm(dataloader, desc='Train (batch)', leave=False) if tqdm is not None else dataloader
    for batch in data_iter:
        # Move data to GPU with non_blocking for RTX 4070
        x = batch['x'].to(device, non_blocking=True)
        y = batch['y'].to(device, non_blocking=True)
        adj_matrix = batch['adj_matrix'][0].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass for RTX 4070 Tensor Cores
        with amp.autocast():
            outputs = model(x, adj_matrix)
            predictions = outputs['predictions']
            
            # Check for NaN/inf in predictions
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                print("NaN/inf detected in predictions, skipping batch")
                continue
            
            # Ensure target and prediction channel dims match for loss calculation
            if predictions.shape[-1] != y.shape[-1]:
                # Prefer trimming target channels to match predictions (primary channel)
                if y.shape[-1] > predictions.shape[-1]:
                    y = y[..., :predictions.shape[-1]]
                else:
                    # If predictions have more channels than target, replicate target
                    y = y.expand(*y.shape[:-1], predictions.shape[-1])
            loss = criterion(predictions, y)
            
            # Check for NaN/inf in loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("NaN/inf detected in loss, skipping batch")
                continue
            
            # Add uncertainty loss if available
            if 'uncertainty' in outputs and outputs['uncertainty'] is not None:
                uncertainty_loss = torch.mean(outputs['uncertainty'])
                if not (torch.isnan(uncertainty_loss).any() or torch.isinf(uncertainty_loss).any()):
                    loss = loss + 0.1 * uncertainty_loss
        
        # Mixed precision backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping with scaled gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        # Optimizer step with scaling
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate_epoch_gpu_optimized(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                                device: torch.device, scaler_data=None) -> Tuple[float, Dict[str, float]]:
    """Validate for one epoch with GPU optimizations"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        data_iter = tqdm(dataloader, desc='Val (batch)', leave=False) if tqdm is not None else dataloader
        for batch in data_iter:
            # Non-blocking GPU transfer for RTX 4070
            x = batch['x'].to(device, non_blocking=True)
            y = batch['y'].to(device, non_blocking=True)
            adj_matrix = batch['adj_matrix'][0].to(device, non_blocking=True)
            
            # Mixed precision inference for RTX 4070
            with amp.autocast():
                outputs = model(x, adj_matrix)
                predictions = outputs['predictions']
                loss = criterion(predictions, y)
            
            total_loss += loss.item()
            
            # Store for metric calculation (keep on GPU until needed)
            all_predictions.append(predictions)
            all_targets.append(y)
    
    avg_loss = total_loss / len(dataloader)
    
    # Concatenate on GPU then move to CPU for metrics
    predictions_concat = torch.cat(all_predictions, dim=0).cpu().numpy()
    targets_concat = torch.cat(all_targets, dim=0).cpu().numpy()
    
    # Denormalize if needed
    if scaler_data is not None:
        pred_flat = predictions_concat.reshape(-1, predictions_concat.shape[-1])
        target_flat = targets_concat.reshape(-1, targets_concat.shape[-1])

        # sklearn scaler expects the same number of features it was fitted on
        try:
            n_in = int(getattr(scaler_data, 'n_features_in_', pred_flat.shape[1]))
        except Exception:
            n_in = pred_flat.shape[1]

        if pred_flat.shape[1] != n_in:
            # Tile predictions across features so inverse_transform can run, then we'll
            # extract the primary channel afterwards
            if pred_flat.shape[1] == 1 and n_in > 1:
                pred_flat = np.tile(pred_flat, (1, n_in))
        if target_flat.shape[1] != n_in:
            if target_flat.shape[1] == 1 and n_in > 1:
                target_flat = np.tile(target_flat, (1, n_in))

        pred_denorm = scaler_data.inverse_transform(pred_flat).reshape(predictions_concat.shape[0],
                                                                      predictions_concat.shape[1],
                                                                      predictions_concat.shape[2], n_in)
        target_denorm = scaler_data.inverse_transform(target_flat).reshape(targets_concat.shape[0],
                                                                         targets_concat.shape[1],
                                                                         targets_concat.shape[2], n_in)
        
        # If multi-channel output, evaluate only primary channel (index 0)
        # Evaluate primary channel if multi-channel
        if pred_denorm.ndim == 4 and pred_denorm.shape[-1] > 1:
            metrics = calculate_metrics(target_denorm[..., 0], pred_denorm[..., 0])
        else:
            metrics = calculate_metrics(target_denorm, pred_denorm)
    else:
        if predictions_concat.ndim == 4 and predictions_concat.shape[-1] > 1:
            metrics = calculate_metrics(targets_concat[..., 0], predictions_concat[..., 0])
        else:
            metrics = calculate_metrics(targets_concat, predictions_concat)
    
    return avg_loss, metrics


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, device: torch.device) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    data_iter = tqdm(dataloader, desc='Train (batch)', leave=False) if tqdm is not None else dataloader
    for batch in data_iter:
        x = batch['x'].to(device)  # [batch_size, num_nodes, seq_len, input_dim]
        y = batch['y'].to(device)  # [batch_size, num_nodes, pred_len, input_dim]
        adj_matrix = batch['adj_matrix'][0].to(device)  # [num_nodes, num_nodes]
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x, adj_matrix)
        predictions = outputs['predictions']

        # Ensure prediction and target channel dims match for loss
        if predictions.shape[-1] != y.shape[-1]:
            if y.shape[-1] > predictions.shape[-1]:
                y = y[..., :predictions.shape[-1]]
            else:
                y = y.expand(*y.shape[:-1], predictions.shape[-1])

        # Calculate loss
        loss = criterion(predictions, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                  device: torch.device, scaler=None) -> Tuple[float, Dict[str, float]]:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        data_iter = tqdm(dataloader, desc='Val (batch)', leave=False) if tqdm is not None else dataloader
        for batch in data_iter:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            adj_matrix = batch['adj_matrix'][0].to(device)
            
            # Forward pass
            outputs = model(x, adj_matrix)
            predictions = outputs['predictions']
            
            # Ensure prediction and target channel dims match for loss
            if predictions.shape[-1] != y.shape[-1]:
                if y.shape[-1] > predictions.shape[-1]:
                    y = y[..., :predictions.shape[-1]]
                else:
                    y = y.expand(*y.shape[:-1], predictions.shape[-1])

            # Calculate loss
            loss = criterion(predictions, y)
            total_loss += loss.item()
            
            # Store for metric calculation
            if scaler is not None:
                # Denormalize for metric calculation
                pred_np = predictions.cpu().numpy()
                target_np = y.cpu().numpy()
                
                # Reshape and denormalize
                pred_flat = pred_np.reshape(-1, pred_np.shape[-1])
                target_flat = target_np.reshape(-1, target_np.shape[-1])

                # Ensure scaler inverse_transform can accept the number of features
                try:
                    n_in = int(getattr(scaler, 'n_features_in_', pred_flat.shape[1]))
                except Exception:
                    n_in = pred_flat.shape[1]

                if pred_flat.shape[1] != n_in:
                    if pred_flat.shape[1] == 1 and n_in > 1:
                        pred_flat = np.tile(pred_flat, (1, n_in))
                if target_flat.shape[1] != n_in:
                    if target_flat.shape[1] == 1 and n_in > 1:
                        target_flat = np.tile(target_flat, (1, n_in))

                pred_denorm = scaler.inverse_transform(pred_flat).reshape(pred_np.shape[0], pred_np.shape[1], pred_np.shape[2], n_in)
                target_denorm = scaler.inverse_transform(target_flat).reshape(target_np.shape[0], target_np.shape[1], target_np.shape[2], n_in)

                all_predictions.append(pred_denorm)
                all_targets.append(target_denorm)
            else:
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate metrics (primary channel only if multi-channel)
    predictions_concat = np.concatenate(all_predictions, axis=0)
    targets_concat = np.concatenate(all_targets, axis=0)
    if predictions_concat.ndim == 4 and predictions_concat.shape[-1] > 1:
        metrics = calculate_metrics(targets_concat[..., 0], predictions_concat[..., 0])
    else:
        metrics = calculate_metrics(targets_concat, predictions_concat)
    
    return avg_loss, metrics


def train_model_gpu_optimized(config: Dict):
    """Main training function optimized for RTX 4070 12GB GPU"""
    return _train_model_common(config, use_gpu_optimized=True)


def _train_model_common(config: Dict, resume: Optional[str] = None, use_gpu_optimized: bool = False):
    """Common train entrypoint used by GPU-optimized and CPU training paths.
    If resume is provided and points to a checkpoint, training resumes from that epoch.
    """
    # Determine whether to use GPU path
    if use_gpu_optimized:
        # delegate to GPU-optimized specific logic in this same function (we'll follow same flow)
        pass
    
    # GPU optimization settings for RTX 4070
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Enable optimizations for RTX 4070
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()  # Clear cache
        
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    print(f"Using device: {device}")

    # Wire metrics config (MAPE mode/threshold) into module globals so calculate_metrics uses them
    try:
        metrics_cfg = config.get('metrics', {}) if isinstance(config, dict) else {}
        globals()['METRICS_MAPE_MODE'] = metrics_cfg.get('mape_mode', MAPE_MODE)
        globals()['METRICS_MAPE_THRESHOLD'] = float(metrics_cfg.get('mape_threshold', MAPE_THRESHOLD))
    except Exception:
        globals()['METRICS_MAPE_MODE'] = MAPE_MODE
        globals()['METRICS_MAPE_THRESHOLD'] = float(MAPE_THRESHOLD)
    
    # Load data
    print("Loading data...")
    speed_data, adj_matrix = load_data(config)
    # Default to primary channel unless user requests all channels
    data_cfg = config.get('data', {})
    use_all_channels = bool(data_cfg.get('use_all_channels', False))
    if speed_data.ndim == 3 and speed_data.shape[-1] > 1 and not use_all_channels:
        print(f"Note: dataset has {speed_data.shape[-1]} features per node; selecting primary channel 0 for training (set data.use_all_channels=True to keep all).")
        speed_data = speed_data[..., 0:1]
    # Default to primary channel unless user requests all channels
    data_cfg = config.get('data', {})
    use_all_channels = bool(data_cfg.get('use_all_channels', False))
    if speed_data.ndim == 3 and speed_data.shape[-1] > 1 and not use_all_channels:
        print(f"Note: dataset has {speed_data.shape[-1]} features per node; selecting primary channel 0 for training (set data.use_all_channels=True to keep all).")
        speed_data = speed_data[..., 0:1]
    # If data has multiple feature channels per node (e.g., speed/flow/occupancy),
    # default to using only the primary channel (index 0) unless explicitly requested.
    data_cfg = config.get('data', {})
    use_all_channels = bool(data_cfg.get('use_all_channels', False))
    if speed_data.ndim == 3 and speed_data.shape[-1] > 1 and not use_all_channels:
        print(f"Note: dataset has {speed_data.shape[-1]} features per node; selecting primary channel 0 for training (set data.use_all_channels=True to keep all).")
        speed_data = speed_data[..., 0:1]
    
    # Create data splits
    train_data, val_data, test_data = create_data_splits(speed_data, config)
    print(f"Data splits - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
    
    # Normalize data
    train_data_norm, val_data_norm, test_data_norm, scaler = normalize_data(
        train_data, val_data, test_data, config
    )
    
    # Create datasets
    seq_len = config['data']['seq_len']
    pred_len = config['data']['pred_len']
    
    train_dataset = TrafficDataset(train_data_norm, adj_matrix, seq_len, pred_len)
    val_dataset = TrafficDataset(val_data_norm, adj_matrix, seq_len, pred_len)
    
    # GPU-optimized data loaders for RTX 4070
    batch_size = config['training']['batch_size']
    num_workers = config['hardware']['num_workers']
    pin_memory = config['hardware']['pin_memory']
    persistent_workers = config['hardware'].get('persistent_workers', True)
    prefetch_factor = config['hardware'].get('prefetch_factor', 4)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True  # For consistent batch sizes
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    
    # Create model
    print("Creating model...")
    # Ensure model input/output dims match the data. If the config requests using all channels
    # (data.use_all_channels=True) then override model.input_dim to match; otherwise we kept
    # the data to a single channel above and will respect config's input_dim.
    try:
        data_input_dim = train_data_norm.shape[-1]
        model_cfg = config.get('model', {})
        if bool(data_cfg.get('use_all_channels', False)):
            if model_cfg.get('input_dim', None) != data_input_dim:
                print(f"Note: overriding model.input_dim from {model_cfg.get('input_dim', None)} to {data_input_dim} to match data (use_all_channels=True)")
                model_cfg['input_dim'] = int(data_input_dim)
        # Also ensure num_nodes matches adjacency if available
        if 'num_nodes' in model_cfg and adj_matrix is not None:
            if model_cfg.get('num_nodes') != adj_matrix.shape[0]:
                print(f"Note: overriding model.num_nodes from {model_cfg.get('num_nodes')} to {adj_matrix.shape[0]} to match adjacency")
                model_cfg['num_nodes'] = int(adj_matrix.shape[0])
        config['model'] = model_cfg
    except Exception:
        pass

    model = create_ustgt_model(config)
    model = model.to(device)
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # GPU memory estimation
    if device.type == 'cuda':
        model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
        print(f"Estimated model memory: {model_memory:.2f} GB")
    
    # Compile model for faster training (PyTorch 2.0+)
    try:
        # Allow disabling compilation via environment var to avoid long JIT pauses during debugging
        disable_compile = os.environ.get('USTGT_DISABLE_COMPILE', '0') == '1'
        if disable_compile:
            print("Model compilation disabled via USTGT_DISABLE_COMPILE=1")
        else:
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled for faster training!")
    except Exception as e:
        print(f"Model compilation not available: {e}")
    
    # Create optimizer according to config
    opt_name = config['training'].get('optimizer', 'Adam')
    lr = float(config['training'].get('learning_rate', 1e-3))
    weight_decay = float(config['training'].get('weight_decay', 0.0))
    if opt_name.lower() in ('adam', 'adamw'):
        optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif opt_name.lower() in ('rmsprop', 'rms'):
        optimizer = optim.RMSprop(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.0, eps=1e-8
        )
    else:
        # fallback to Adam
        optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    # Mixed precision scaler for RTX 4070 Tensor Cores
    use_amp = config['hardware'].get('mixed_precision', True)
    scaler = amp.GradScaler() if use_amp and device.type == 'cuda' else None

    # Create loss criterion according to config
    loss_name = config['training'].get('loss', 'mse').lower()
    if loss_name == 'mae' or loss_name == 'l1':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = None
    scheduler_type = config['training'].get('scheduler', None)
    if scheduler_type == 'multistep':
        params = config['training'].get('scheduler_params', {})
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=params.get('milestones', []),
            gamma=params.get('gamma', 0.1)
        )
    elif scheduler_type == 'cosine':
        params = config['training'].get('scheduler_params', {})
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config['training'].get('max_epochs', 50)),
            eta_min=params.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'step':
        params = config['training'].get('scheduler_params', {})
        step_size = int(params.get('step_size', 5))
        gamma = float(params.get('gamma', 0.7))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        scheduler = None
    
    # Training loop
    print("Starting GPU-optimized training...")
    best_val_loss = float('inf')
    patience_counter = 0
    max_epochs = config['training']['max_epochs']
    patience = config['training']['patience']
    gradient_clip = config['training']['gradient_clip']
    
    # Create checkpoint directory (include dataset and model in subdir)
    dataset_name = config['data'].get('dataset_name', 'dataset')
    model_name = config.get('model', {}).get('name', 'USTGT') if isinstance(config.get('model', {}), dict) else 'USTGT'
    checkpoint_dir_base = config['logging'].get('checkpoint_dir', 'checkpoints')
    checkpoint_dir = os.path.join(checkpoint_dir_base, f"{model_name}_{dataset_name}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training metrics tracking
    train_losses = []
    val_losses = []
    val_rmses = []
    
    for epoch in range(max_epochs):
        start_time = time.time()
        
        # Train with GPU optimizations
        if scaler is not None:
            train_loss = train_epoch_gpu_optimized(
                model, train_loader, optimizer, criterion, device, scaler, gradient_clip
            )
        else:
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate with GPU optimizations
        val_loss, val_metrics = validate_epoch_gpu_optimized(
            model, val_loader, criterion, device, scaler
        )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        epoch_time = time.time() - start_time

        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_rmses.append(val_metrics['rmse'])

        # Show progress every epoch with progress bar
        current_lr = optimizer.param_groups[0]['lr']
        progress = (epoch + 1) / max_epochs * 100

        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * (epoch + 1) // max_epochs)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        print(f"Epoch [{epoch+1:6d}/{max_epochs}] |{bar}| {progress:5.1f}% | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"RMSE: {val_metrics['rmse']:.4f} | MAE: {val_metrics['mae']:.4f} | "
              f"Time: {epoch_time:.2f}s | LR: {current_lr:.2e}", flush=True)

        # Show GPU memory usage every 100 epochs
        if device.type == 'cuda' and epoch % 100 == 0:
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            print(f"    GPU Memory - Used: {memory_used:.2f}GB, Cached: {memory_cached:.2f}GB")

        # Save checkpoint every log_every_n_steps epochs
        if epoch % config['logging']['log_every_n_steps'] == 0 and epoch > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'{dataset_name}_{model_name}_checkpoint_epoch_{epoch}.pth')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"    Checkpoint saved: {checkpoint_path}", flush=True)

        # Log detailed progress every log_every_n_steps
        if epoch % config['logging']['log_every_n_steps'] == 0:
            print(f"    ===== DETAILED LOG (Epoch {epoch}) =====", flush=True)
            print(f"    Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            print(f"    RMSE: {val_metrics['rmse']:.6f} | MAE: {val_metrics['mae']:.6f} | MAPE: {val_metrics['mape']:.3f}%")
            print(f"    Learning Rate: {current_lr:.8f}")
            print(f"    Best Val Metric: {best_val_loss:.6f}")
            print(f"    Time per Epoch: {epoch_time:.3f}s")
            if device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_cached = torch.cuda.memory_reserved() / 1024**3
                print(f"    GPU Memory - Used: {memory_used:.3f}GB, Cached: {memory_cached:.3f}GB")
            print(f"    =========================================", flush=True)
        
        # Save best model
        monitor_metric = config['logging']['monitor_metric']
        current_metric = val_loss if monitor_metric == 'val_loss' else val_metrics.get('rmse', val_loss)
        
        if current_metric < best_val_loss:
            best_val_loss = current_metric
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config
            }
            
            checkpoint_path = os.path.join(checkpoint_dir, f'{dataset_name}_{model_name}_best_epoch_{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Best model saved: {monitor_metric}={current_metric:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    print("Training completed!")
    print(f"Best validation {monitor_metric}: {best_val_loss:.4f}")
    
    # Final GPU memory cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_rmses': val_rmses,
        'best_val_loss': best_val_loss
    }


def train_model(config: Dict, resume: Optional[str] = None):
    """Main training function"""

    # Set device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    speed_data, adj_matrix = load_data(config)
    
    # Create data splits
    train_data, val_data, test_data = create_data_splits(speed_data, config)
    print(f"Data splits - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
    
    # Normalize data
    train_data_norm, val_data_norm, test_data_norm, scaler = normalize_data(
        train_data, val_data, test_data, config
    )
    
    # Create datasets
    seq_len = config['data']['seq_len']
    pred_len = config['data']['pred_len']
    
    train_dataset = TrafficDataset(train_data_norm, adj_matrix, seq_len, pred_len)
    val_dataset = TrafficDataset(val_data_norm, adj_matrix, seq_len, pred_len)
    test_dataset = TrafficDataset(test_data_norm, adj_matrix, seq_len, pred_len)
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    num_workers = config['hardware']['num_workers']
    pin_memory = config['hardware']['pin_memory']
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    # Create model
    print("Creating model...")
    # Ensure model input/output dims match the data
    try:
        data_input_dim = train_data.shape[-1]
        model_cfg = config.get('model', {})
        if model_cfg.get('input_dim', None) != data_input_dim:
            print(f"Note: overriding model.input_dim from {model_cfg.get('input_dim', None)} to {data_input_dim} to match data")
            model_cfg['input_dim'] = int(data_input_dim)
        # Also ensure num_nodes matches adjacency if available
        if 'num_nodes' in model_cfg and adj_matrix is not None:
            if model_cfg.get('num_nodes') != adj_matrix.shape[0]:
                print(f"Note: overriding model.num_nodes from {model_cfg.get('num_nodes')} to {adj_matrix.shape[0]} to match adjacency")
                model_cfg['num_nodes'] = int(adj_matrix.shape[0])
        config['model'] = model_cfg
    except Exception:
        pass

    model = create_ustgt_model(config)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and criterion
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    if config['training']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['max_epochs'],
            eta_min=config['training']['scheduler_params']['eta_min']
        )
    else:
        scheduler = None
    
    # Training loop
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Model: USTGT with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Dataset: {config['data']['dataset_name']}")
    print(f"Max Epochs: {config['training']['max_epochs']:,}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print(f"Device: {device}")
    print(f"Checkpoints will be saved every {config['logging']['log_every_n_steps']} epochs")
    print("=" * 80)
    best_val_loss = float('inf')
    patience_counter = 0
    max_epochs = config['training']['max_epochs']
    patience = config['training']['patience']
    
    # Create checkpoint directory
    checkpoint_dir = config['logging']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    start_epoch = 0
    # If resume checkpoint specified, load it
    if resume is not None and os.path.exists(resume):
        try:
            ckpt = torch.load(resume)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt and 'optimizer' in locals():
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception:
                    pass
            if 'scheduler_state_dict' in ckpt and 'scheduler' in locals() and scheduler is not None:
                try:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                except Exception:
                    pass
            start_epoch = int(ckpt.get('epoch', 0)) + 1
            best_val_loss = float(ckpt.get('val_loss', best_val_loss))
            print(f"Resuming from checkpoint {resume}, starting at epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load resume checkpoint {resume}: {e}")

    for epoch in range(start_epoch, max_epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, scaler)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        epoch_time = time.time() - start_time

        # Always print a concise epoch summary so the user sees progress each epoch
        current_lr = optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0 else 0.0
        print(f"Epoch {epoch+1:3d}/{max_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val RMSE: {val_metrics['rmse']:.4f} | "
              f"Val MAE: {val_metrics['mae']:.4f} | "
              f"Time: {epoch_time:.2f}s | LR: {current_lr:.2e}", flush=True)

        # Check for improvement
        monitor_metric = config['logging']['monitor_metric']
        if monitor_metric == 'val_loss':
            current_metric = val_loss
        else:
            current_metric = val_metrics.get(monitor_metric.replace('val_', ''), val_loss)

        if current_metric < best_val_loss:
            best_val_loss = current_metric
            patience_counter = 0

            # Save best model if configured to do so (default: True)
            if config.get('logging', {}).get('save_best_model', True):
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_metrics': val_metrics,
                        'config': config,
                        'scaler': scaler
                    }, os.path.join(checkpoint_dir, 'best_model.pth'))
                    print(f"✓ Best model saved (epoch {epoch})", flush=True)
                except Exception as e:
                    print(f"Failed to save best model: {e}")
        else:
            patience_counter += 1

        # Periodic checkpointing according to log_every_n_steps (if enabled)
        save_every = config.get('logging', {}).get('log_every_n_steps', None)
        if save_every is not None and save_every > 0 and epoch % save_every == 0 and epoch > 0:
            try:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'config': config,
                    'scaler': scaler
                }, checkpoint_path)
                print(f"    Checkpoint saved: {checkpoint_path}", flush=True)
            except Exception as e:
                print(f"Failed to save periodic checkpoint: {e}")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # Append epoch metrics to CSV log
        try:
            csv_path = os.path.join(checkpoint_dir, 'training_log.csv')
            write_epoch_log(csv_path, epoch, train_loss, val_loss, val_metrics, current_lr, epoch_time)
        except Exception as e:
            print(f"Failed to write epoch CSV log: {e}")
    
    # Save final checkpoint (last state)
    final_checkpoint_path = os.path.join(checkpoint_dir, 'last_model.pth')
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if 'optimizer' in locals() else None,
            'val_loss': val_loss if 'val_loss' in locals() else None,
            'val_metrics': val_metrics if 'val_metrics' in locals() else None,
            'config': config,
            'scaler': scaler if 'scaler' in locals() else None
        }, final_checkpoint_path)
        print(f"Final checkpoint saved: {final_checkpoint_path}", flush=True)
    except Exception as e:
        print(f"Failed to save final checkpoint: {e}")

    # Load best model and evaluate on test set
    print("Evaluating on test set...")
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        # Try weights_only for newer PyTorch; if that fails (e.g., WeightsUnpicklingError),
        # fall back to a full torch.load (only do this for trusted local checkpoints).
        checkpoint = None
        try:
            checkpoint = torch.load(best_model_path, weights_only=True)
        except Exception as e:
            print(f"Warning: weights_only load failed: {e}; falling back to full torch.load (trusted local file only).")
            try:
                checkpoint = torch.load(best_model_path)
            except Exception as e2:
                print(f"Failed to load checkpoint with full torch.load: {e2}")
                checkpoint = None

        # Load model weights only if present
        if checkpoint is not None and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            print("Best model file did not contain a model_state_dict or could not be loaded; using current model")
    else:
        print("No best model found, using current model for evaluation")
    
    test_loss, test_metrics = validate_epoch(model, test_loader, criterion, device, scaler)
    
    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test MAE: {test_metrics['mae']:.4f}")
    print(f"Test MAPE: {test_metrics['mape']:.2f}%")
    
    return model, test_metrics


def get_default_config():
    """Return default configuration when YAML is not available"""
    return {
        'data': {
            'data_dir': '../datasets',
            'dataset_name': 'METR-LA',  # Use METR-LA for comparison with baselines
            'seq_len': 12,
            'pred_len': 12,  # 12 steps = 1 hour prediction
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'normalize': True,
            'scaler_type': 'standard'
        },
        'model': {
            'num_nodes': 207,  # METR-LA has 207 nodes
            'input_dim': 1,
            'hidden_dim': 64,
            'output_dim': 1,
            'seq_len': 12,
            'pred_len': 12,
            'num_heads': 8,
            'num_layers': 3,
            'dropout': 0.1,
            'use_uncertainty': True,
            'use_external_features': False  # Disable for fair comparison
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'max_epochs': 50,
            'patience': 15,
            'gradient_clip': 1.0,
            'scheduler': 'cosine',
            'scheduler_params': {
                'eta_min': 1e-5
            }
        },
        'hardware': {
            'device': 'cuda',
            'num_workers': 4,
            'pin_memory': True
        },
        'logging': {
            'checkpoint_dir': 'checkpoints',
            'save_top_k': 3,
            'monitor_metric': 'val_rmse',
            'log_every_n_steps': 10
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Train USTGT model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--test-run', action='store_true',
                       help='Run a quick test with dummy data')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file to resume training from')

    args = parser.parse_args()
    
    # Load configuration or run presets automatically
    config = None
    if args.config == 'config.yaml':
        # If preset config files exist, run them sequentially so user only needs to run train.py
        preset_paths = [os.path.join('configs', 'lsttn_preset.yaml'), os.path.join('configs', 'sttn_preset.yaml')]
        found = [p for p in preset_paths if os.path.exists(p)]
        if found:
            print(f"Found presets: {found}. Running each preset sequentially.")
            for p in found:
                try:
                    if YAML_AVAILABLE:
                        with open(p, 'r') as f:
                            cfg = yaml.safe_load(f)
                    else:
                        cfg = get_default_config()
                except Exception as e:
                    print(f"Failed to load preset {p}: {e}; skipping")
                    continue

                # Ensure data split keys exist
                data = cfg.get('data', {})
                data.setdefault('train_ratio', 0.7)
                data.setdefault('val_ratio', 0.15)
                data.setdefault('test_ratio', 0.15)
                cfg['data'] = data

                print(f"Running preset config: {p}")
                train_model(cfg, resume=args.resume)

            print("All presets finished.")
            return
        # no presets found, fall back to loading provided config file or defaults
    if YAML_AVAILABLE and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {args.config}")
    else:
        config = get_default_config()
        print("Using default configuration (YAML not available or config file not found)")

    # Ensure data split defaults exist for any loaded config
    data_cfg = config.get('data', {})
    data_cfg.setdefault('train_ratio', 0.7)
    data_cfg.setdefault('val_ratio', 0.15)
    data_cfg.setdefault('test_ratio', max(0.0, 1.0 - data_cfg['train_ratio'] - data_cfg['val_ratio']))
    config['data'] = data_cfg
    
    # Quick test mode
    if args.test_run:
        print("Running quick test with reduced parameters...")
        config['training']['max_epochs'] = 2
        config['training']['batch_size'] = 8
        config['model']['num_nodes'] = 10
        config['model']['hidden_dim'] = 32
    
    # Train model (optionally resume)
    model, metrics = train_model(config, resume=args.resume)
    
    print(f"Training completed successfully!")


if __name__ == '__main__':
    main()

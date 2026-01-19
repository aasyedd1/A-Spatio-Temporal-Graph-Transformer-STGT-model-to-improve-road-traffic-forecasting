"""
GPU-Optimized metrics for USTGT model - RTX 4070 12GB optimized
Includes all metrics used in baseline models for fair comparison
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked Mean Absolute Error (GPU optimized, matching LSTTN implementation)"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked Mean Squared Error (GPU optimized, matching LSTTN implementation)"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked Root Mean Squared Error (GPU optimized, matching LSTTN implementation)"""
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mape(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan, 
                min_val: float = 1.0) -> torch.Tensor:
    """Masked Mean Absolute Percentage Error (GPU optimized, with proper masking)
    
    Args:
        preds: Predictions tensor
        labels: Ground truth labels tensor  
        null_val: Null values to ignore
        min_val: Minimum absolute value threshold to avoid division by very small numbers
    """
    # Mask null values
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    
    # Additional mask for values too close to zero (causes huge MAPE)
    magnitude_mask = torch.abs(labels) >= min_val
    
    # Combine masks
    combined_mask = mask & magnitude_mask
    
    if combined_mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device)
    
    # Calculate MAPE only for valid values
    masked_labels = labels[combined_mask]
    masked_preds = preds[combined_mask]
    
    mape = torch.abs(masked_preds - masked_labels) / torch.abs(masked_labels)
    
    # Return percentage
    return torch.mean(mape) * 100.0


def all_metrics_with_denorm_mape(pred: torch.Tensor, real: torch.Tensor, 
                                 pred_denorm: torch.Tensor = None, real_denorm: torch.Tensor = None,
                                 mape_min_val: float = 2.0) -> Tuple[float, float, float]:
    """Calculate metrics with MAPE on denormalized scale, MAE/RMSE on normalized scale
    
    Args:
        pred: Predictions tensor (normalized scale for MAE/RMSE)
        real: Ground truth tensor (normalized scale for MAE/RMSE)
        pred_denorm: Predictions tensor (denormalized scale for MAPE)
        real_denorm: Ground truth tensor (denormalized scale for MAPE)  
        mape_min_val: Minimum absolute value for MAPE calculation
    """
    # Calculate MAE and RMSE on normalized scale (for comparison with baselines)
    mae = masked_mae(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    
    # Calculate MAPE on denormalized scale (for meaningful percentage)
    if pred_denorm is not None and real_denorm is not None:
        mape = masked_mape(pred_denorm, real_denorm, null_val=0.0, min_val=mape_min_val).item()
    else:
        # Fallback to normalized scale if denormalized not available
        mape = masked_mape(pred, real, null_val=0.0, min_val=mape_min_val).item()
    
    return mae, mape, rmse


def all_metrics(pred: torch.Tensor, real: torch.Tensor, mape_min_val: float = 1.0) -> Tuple[float, float, float]:
    """Calculate all standard metrics (GPU optimized, with proper MAPE masking)
    
    Args:
        pred: Predictions tensor
        real: Ground truth tensor
        mape_min_val: Minimum absolute value for MAPE calculation (avoids division by very small numbers)
    """
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, null_val=0.0, min_val=mape_min_val).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


def uncertainty_metrics(predictions: torch.Tensor, uncertainties: torch.Tensor, 
                       targets: torch.Tensor, confidence_level: float = 0.95) -> Dict[str, float]:
    """Calculate uncertainty-specific metrics for USTGT (GPU optimized)"""
    z_score = torch.tensor(1.96 if confidence_level == 0.95 else 2.58, device=predictions.device)
    upper_bound = predictions + z_score * torch.sqrt(uncertainties)
    lower_bound = predictions - z_score * torch.sqrt(uncertainties)
    
    within_interval = (targets >= lower_bound) & (targets <= upper_bound)
    coverage_ratio = torch.mean(within_interval.float())
    interval_width = torch.mean(upper_bound - lower_bound)
    
    errors = torch.abs(predictions - targets)
    try:
        uncertainty_corr = torch.corrcoef(torch.stack([
            errors.flatten(), 
            torch.sqrt(uncertainties).flatten()
        ]))[0, 1]
    except:
        uncertainty_corr = torch.tensor(0.0, device=predictions.device)
    
    return {
        'coverage_ratio': coverage_ratio.item(),
        'interval_width': interval_width.item(),
        'uncertainty_correlation': uncertainty_corr.item() if not torch.isnan(uncertainty_corr) else 0.0
    }


def comprehensive_evaluation(predictions: torch.Tensor, targets: torch.Tensor, 
                           uncertainties: Optional[torch.Tensor] = None,
                           mape_min_val: float = 1.0) -> Dict[str, float]:
    """Comprehensive evaluation combining standard and uncertainty metrics (GPU optimized)
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        uncertainties: Uncertainty estimates (optional)
        mape_min_val: Minimum absolute value for MAPE calculation
    """
    mae, mape, rmse = all_metrics(predictions, targets, mape_min_val=mape_min_val)
    
    results = {
        'mae': mae,
        'mape': mape,
        'rmse': rmse
    }
    
    # Add masked fraction info for MAPE
    if targets.numel() > 0:
        total_values = targets.numel()
        valid_for_mape = (torch.abs(targets) >= mape_min_val).sum().item()
        results['mape_masked_fraction'] = 1.0 - (valid_for_mape / total_values)
    
    if uncertainties is not None:
        uncertainty_results = uncertainty_metrics(predictions, uncertainties, targets)
        results.update(uncertainty_results)
    
    return results


def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """Print metrics in a formatted way for comparison"""
    print(f"\n{model_name} Performance Metrics:")
    print("-" * 40)
    
    if 'mae' in metrics:
        print(f"MAE:  {metrics['mae']:.4f}")
    if 'rmse' in metrics:
        print(f"RMSE: {metrics['rmse']:.4f}")
    if 'mape' in metrics:
        print(f"MAPE: {metrics['mape']:.4f}%")
    
    if 'coverage_ratio' in metrics:
        print(f"\nUncertainty Metrics:")
        print(f"Coverage Ratio: {metrics['coverage_ratio']:.4f}")
        print(f"Interval Width: {metrics['interval_width']:.4f}")
        print(f"Uncertainty Correlation: {metrics['uncertainty_correlation']:.4f}")

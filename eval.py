"""
USTGT Evaluation Script
Comprehensive evaluation including uncertainty analysis and visualization
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import pickle
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List

# Import dependencies
from train import TrafficDataset, load_data, create_data_splits, normalize_data, calculate_metrics
from test import load_trained_model, evaluate_model, count_parameters
from model import create_ustgt_model

# Default MAPE config fallbacks
try:
    MAPE_MODE
except NameError:
    MAPE_MODE = 'masked'

try:
    MAPE_THRESHOLD
except NameError:
    MAPE_THRESHOLD = 1.0


def evaluate_uncertainty_quality(predictions: np.ndarray, targets: np.ndarray, 
                                uncertainties: np.ndarray, confidence_levels: List[float] = [0.9, 0.95, 0.99]) -> Dict[str, float]:
    """Evaluate uncertainty calibration quality"""
    if uncertainties is None:
        return {}
    
    results = {}
    
    for confidence in confidence_levels:
        # Calculate prediction intervals
        z_score = {0.9: 1.645, 0.95: 1.96, 0.99: 2.576}[confidence]
        std_dev = np.sqrt(uncertainties)
        
        lower_bound = predictions - z_score * std_dev
        upper_bound = predictions + z_score * std_dev
        
        # Calculate coverage (fraction of targets within intervals)
        within_interval = (targets >= lower_bound) & (targets <= upper_bound)
        coverage = np.mean(within_interval)
        
        # Calculate average interval width
        interval_width = np.mean(upper_bound - lower_bound)
        
        results[f'coverage_{int(confidence*100)}'] = coverage
        results[f'interval_width_{int(confidence*100)}'] = interval_width
    
    return results


def analyze_prediction_errors(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, np.ndarray]:
    """Analyze prediction error patterns"""
    errors = predictions - targets
    abs_errors = np.abs(errors)
    
    analysis = {
        'mean_error': np.mean(errors),
        'mean_abs_error': np.mean(abs_errors),
        'error_std': np.std(errors),
        'error_skewness': scipy.stats.skew(errors.flatten()) if 'scipy' in globals() else 0,
        'error_kurtosis': scipy.stats.kurtosis(errors.flatten()) if 'scipy' in globals() else 0,
        'max_error': np.max(abs_errors),
        'min_error': np.min(abs_errors),
        'error_percentiles': np.percentile(abs_errors, [25, 50, 75, 90, 95, 99])
    }
    
    return analysis


def evaluate_temporal_patterns(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Evaluate how well the model captures temporal patterns"""
    batch_size, num_nodes, pred_len, _ = predictions.shape
    
    # Calculate temporal correlation for each prediction horizon
    temporal_correlations = []
    
    for h in range(pred_len):
        pred_h = predictions[:, :, h, 0].flatten()
        target_h = targets[:, :, h, 0].flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(pred_h) | np.isnan(target_h))
        if np.sum(mask) > 10:  # Need enough valid points
            corr = np.corrcoef(pred_h[mask], target_h[mask])[0, 1]
            temporal_correlations.append(corr)
        else:
            temporal_correlations.append(0.0)
    
    return {
        'temporal_correlations': temporal_correlations,
        'avg_temporal_correlation': np.mean(temporal_correlations),
        'temporal_correlation_decay': temporal_correlations[0] - temporal_correlations[-1] if len(temporal_correlations) > 1 else 0
    }


def evaluate_spatial_patterns(predictions: np.ndarray, targets: np.ndarray, adj_matrix: np.ndarray) -> Dict[str, float]:
    """Evaluate how well the model captures spatial patterns"""
    batch_size, num_nodes, pred_len, _ = predictions.shape
    
    # Calculate spatial correlation between neighboring nodes
    spatial_correlations = []
    
    # Find connected node pairs from adjacency matrix
    connected_pairs = np.where(adj_matrix > 0)
    
    for i, j in zip(connected_pairs[0], connected_pairs[1]):
        if i != j:  # Skip self-connections
            # Get predictions and targets for both nodes
            pred_i = predictions[:, i, :, 0].flatten()
            pred_j = predictions[:, j, :, 0].flatten()
            target_i = targets[:, i, :, 0].flatten()
            target_j = targets[:, j, :, 0].flatten()
            
            # Calculate correlation between predicted and actual spatial relationships
            if len(pred_i) > 10:
                pred_corr = np.corrcoef(pred_i, pred_j)[0, 1] if np.std(pred_i) > 0 and np.std(pred_j) > 0 else 0
                target_corr = np.corrcoef(target_i, target_j)[0, 1] if np.std(target_i) > 0 and np.std(target_j) > 0 else 0
                
                if not (np.isnan(pred_corr) or np.isnan(target_corr)):
                    spatial_correlations.append(abs(pred_corr - target_corr))
    
    return {
        'avg_spatial_correlation_error': np.mean(spatial_correlations) if spatial_correlations else 0,
        'spatial_correlation_std': np.std(spatial_correlations) if spatial_correlations else 0,
        'num_spatial_pairs': len(spatial_correlations)
    }


def plot_error_distribution(predictions: np.ndarray, targets: np.ndarray, save_path: str = None):
    """Plot distribution of prediction errors"""
    errors = (predictions - targets).flatten()
    errors = errors[~np.isnan(errors)]  # Remove NaN values
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram of errors
    axes[0].hist(errors, bins=50, alpha=0.7, density=True, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].set_xlabel('Prediction Error')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot for normality check
    try:
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normality Check)')
        axes[1].grid(True, alpha=0.3)
    except ImportError:
        axes[1].text(0.5, 0.5, 'scipy not available\nfor Q-Q plot', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Q-Q Plot (scipy required)')
    
    # Box plot
    axes[2].boxplot(errors, vert=True)
    axes[2].set_ylabel('Prediction Error')
    axes[2].set_title('Error Box Plot')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution plot saved to {save_path}")
    
    plt.show()


def plot_uncertainty_calibration(predictions: np.ndarray, targets: np.ndarray, 
                                uncertainties: np.ndarray, save_path: str = None):
    """Plot uncertainty calibration"""
    if uncertainties is None:
        print("No uncertainty data available for calibration plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Flatten arrays
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    unc_flat = uncertainties.flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(pred_flat) | np.isnan(target_flat) | np.isnan(unc_flat))
    pred_flat = pred_flat[mask]
    target_flat = target_flat[mask]
    unc_flat = unc_flat[mask]
    
    # Uncertainty vs Error scatter plot
    errors = np.abs(pred_flat - target_flat)
    std_dev = np.sqrt(unc_flat)
    
    axes[0].scatter(std_dev, errors, alpha=0.5, s=1)
    axes[0].plot([0, std_dev.max()], [0, std_dev.max()], 'r--', linewidth=2, label='Perfect Calibration')
    axes[0].set_xlabel('Predicted Uncertainty (Std Dev)')
    axes[0].set_ylabel('Absolute Error')
    axes[0].set_title('Uncertainty vs Error')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reliability plot
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    observed_frequencies = []
    expected_frequencies = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find data points in this confidence bin
        confidence = (bin_lower + bin_upper) / 2
        z_score = {0.5: 0.674, 0.6: 0.842, 0.7: 1.036, 0.8: 1.282, 0.9: 1.645}.get(confidence, 1.96)
        
        if confidence < 0.5:
            z_score = 0.5 * confidence / 0.5
        elif confidence > 0.9:
            z_score = 1.645 + (confidence - 0.9) * (2.576 - 1.645) / 0.1
        
        # Calculate prediction intervals
        lower_bound = pred_flat - z_score * std_dev
        upper_bound = pred_flat + z_score * std_dev
        
        # Check coverage
        within_interval = (target_flat >= lower_bound) & (target_flat <= upper_bound)
        observed_freq = np.mean(within_interval)
        
        observed_frequencies.append(observed_freq)
        expected_frequencies.append(confidence)
    
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
    axes[1].plot(expected_frequencies, observed_frequencies, 'bo-', linewidth=2, label='Model Calibration')
    axes[1].set_xlabel('Expected Frequency')
    axes[1].set_ylabel('Observed Frequency')
    axes[1].set_title('Reliability Plot')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Uncertainty calibration plot saved to {save_path}")
    
    plt.show()


def comprehensive_evaluation(checkpoint_path: str, config_path: str = None, 
                           output_dir: str = 'evaluation_results'):
    """Run comprehensive evaluation of the model"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading trained model...")
    model, config, scaler = load_trained_model(checkpoint_path, device)
    # Count trainable parameters and include in results
    try:
        num_parameters = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    except Exception:
        num_parameters = None
    
    # Load data
    print("Loading data...")
    speed_data, adj_matrix = load_data(config)
    
    # Create data splits
    train_data, val_data, test_data = create_data_splits(speed_data, config)
    
    # Normalize data
    if scaler is not None:
        test_data_flat = test_data.reshape(-1, test_data.shape[-1])
        test_data_norm = scaler.transform(test_data_flat).reshape(test_data.shape)
    else:
        test_data_norm = test_data
    
    # Create test dataset and dataloader
    seq_len = config['data']['seq_len']
    pred_len = config['data']['pred_len']
    test_dataset = TrafficDataset(test_data_norm, adj_matrix, seq_len, pred_len)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device, scaler, return_predictions=True)
    
    predictions = results['predictions']
    targets = results['targets']
    uncertainties = results['uncertainties']
    
    print(f"Evaluation completed. Predictions shape: {predictions.shape}")
    
    # Basic metrics
    basic_metrics = results['metrics']
    print(f"Basic Metrics - RMSE: {basic_metrics['rmse']:.4f}, MAE: {basic_metrics['mae']:.4f}, MAPE: {basic_metrics['mape']:.2f}%")
    
    # Error analysis
    print("Analyzing prediction errors...")
    error_analysis = analyze_prediction_errors(predictions, targets)
    
    # Temporal pattern analysis
    print("Analyzing temporal patterns...")
    temporal_analysis = evaluate_temporal_patterns(predictions, targets)
    
    # Spatial pattern analysis
    print("Analyzing spatial patterns...")
    spatial_analysis = evaluate_spatial_patterns(predictions, targets, adj_matrix)
    
    # Uncertainty analysis
    uncertainty_analysis = {}
    if uncertainties is not None:
        print("Analyzing uncertainty quality...")
        uncertainty_analysis = evaluate_uncertainty_quality(predictions, targets, uncertainties)
    
    # Generate plots
    print("Generating evaluation plots...")
    
    # Error distribution
    plot_error_distribution(
        predictions, targets,
        save_path=os.path.join(output_dir, 'error_distribution.png')
    )
    
    # Uncertainty calibration
    if uncertainties is not None:
        plot_uncertainty_calibration(
            predictions, targets, uncertainties,
            save_path=os.path.join(output_dir, 'uncertainty_calibration.png')
        )
    
    # Compile comprehensive results
    comprehensive_results = {
        'basic_metrics': basic_metrics,
        'error_analysis': error_analysis,
        'temporal_analysis': temporal_analysis,
        'spatial_analysis': spatial_analysis,
        'uncertainty_analysis': uncertainty_analysis,
    'model_config': config,
    'num_parameters': num_parameters
    }
    
    # Save results
    with open(os.path.join(output_dir, 'comprehensive_evaluation.pkl'), 'wb') as f:
        pickle.dump(comprehensive_results, f)
    
    # Generate detailed report
    generate_comprehensive_report(comprehensive_results, os.path.join(output_dir, 'evaluation_report.txt'))
    
    print(f"Comprehensive evaluation completed! Results saved to {output_dir}")
    
    return comprehensive_results


def generate_comprehensive_report(results: Dict, save_path: str = None):
    """Generate detailed evaluation report"""
    report = []
    report.append("="*80)
    report.append("USTGT COMPREHENSIVE EVALUATION REPORT")
    report.append("="*80)
    report.append("")
    
    # Basic metrics
    basic = results['basic_metrics']
    report.append("BASIC PERFORMANCE METRICS:")
    report.append(f"  RMSE: {basic['rmse']:.4f}")
    report.append(f"  MAE:  {basic['mae']:.4f}")
    report.append(f"  MAPE: {basic['mape']:.2f}%")
    report.append("")
    
    # Error analysis
    error = results['error_analysis']
    report.append("ERROR ANALYSIS:")
    report.append(f"  Mean Error: {error['mean_error']:.4f}")
    report.append(f"  Error Std Dev: {error['error_std']:.4f}")
    report.append(f"  Max Absolute Error: {error['max_error']:.4f}")
    report.append(f"  Error Percentiles (25%, 50%, 75%, 90%, 95%, 99%):")
    percentiles = error['error_percentiles']
    report.append(f"    {percentiles[0]:.3f}, {percentiles[1]:.3f}, {percentiles[2]:.3f}, {percentiles[3]:.3f}, {percentiles[4]:.3f}, {percentiles[5]:.3f}")
    report.append("")
    
    # Temporal analysis
    temporal = results['temporal_analysis']
    report.append("TEMPORAL PATTERN ANALYSIS:")
    report.append(f"  Average Temporal Correlation: {temporal['avg_temporal_correlation']:.4f}")
    report.append(f"  Correlation Decay: {temporal['temporal_correlation_decay']:.4f}")
    report.append("")
    
    # Spatial analysis
    spatial = results['spatial_analysis']
    report.append("SPATIAL PATTERN ANALYSIS:")
    report.append(f"  Average Spatial Correlation Error: {spatial['avg_spatial_correlation_error']:.4f}")
    report.append(f"  Number of Spatial Pairs Analyzed: {spatial['num_spatial_pairs']}")
    report.append("")
    
    # Uncertainty analysis
    uncertainty = results['uncertainty_analysis']
    if uncertainty:
        report.append("UNCERTAINTY ANALYSIS:")
        for key, value in uncertainty.items():
            if 'coverage' in key:
                confidence = key.split('_')[1]
                report.append(f"  {confidence}% Confidence Coverage: {value:.3f}")
            elif 'interval_width' in key:
                confidence = key.split('_')[2]
                report.append(f"  {confidence}% Confidence Interval Width: {value:.4f}")
        report.append("")
    
    # Model configuration summary
    config = results['model_config']
    report.append("MODEL CONFIGURATION:")
    model_params = config['model']
    report.append(f"  Nodes: {model_params.get('num_nodes', 'N/A')}")
    report.append(f"  Hidden Dim: {model_params.get('hidden_dim', 'N/A')}")
    report.append(f"  Layers: {model_params.get('num_layers', 'N/A')}")
    report.append(f"  Sequence Length: {model_params.get('seq_len', 'N/A')}")
    report.append(f"  Prediction Length: {model_params.get('pred_len', 'N/A')}")
    report.append("")
    
    report_text = "\n".join(report)
    print(report_text)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"Comprehensive report saved to {save_path}")
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description='Comprehensive USTGT model evaluation')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Run comprehensive evaluation
    results = comprehensive_evaluation(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir
    )
    
    print("Evaluation completed successfully!")


if __name__ == '__main__':
    main()

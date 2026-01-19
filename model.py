"""
USTGT: Uncertainty-Aware Spatio-Temporal Graph Transformer
Complete unified model - everything in one clean file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Tuple, Optional, List


class USTGT(nn.Module):
    """
    Uncertainty-Aware Spatio-Temporal Graph Transformer
    
    A unified model that combines:
    - Graph attention for spatial modeling
    - Temporal attention for time series modeling
    - Bayesian uncertainty quantification
    - Multi-modal feature fusion
    - Multi-horizon prediction
    """
    
    def __init__(self, 
                 num_nodes: int = 207,           # Matching LSTTN/STTN (METR-LA)
                 input_dim: int = 1,             # Traffic flow dimension
                 hidden_dim: int = 64,           # Matching LSTTN hidden_dim
                 output_dim: int = 1,            # Single output prediction
                 seq_len: int = 12,              # Matching LSTTN/STTN short_seq_len
                 pred_len: int = 12,             # Matching LSTTN/STTN prediction horizon
                 num_heads: int = 8,             # Attention heads
                 num_layers: int = 4,            # Matching LSTTN num_transformer_encoder_layers=4
                 dropout: float = 0.1,           # Matching LSTTN transformer dropout
                 mlp_hidden_dim: int = 128,      # Matching LSTTN mlp_hidden_dim
                 use_uncertainty: bool = True,
                 use_external_features: bool = False):  # Disabled for fair baseline comparison
        
        super(USTGT, self).__init__()
        
        # Model configuration
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.mlp_hidden_dim = mlp_hidden_dim  # Added to match LSTTN
        self.use_uncertainty = use_uncertainty
        self.use_external_features = use_external_features
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 1. SPATIAL ATTENTION (Graph Transformer)
        self.spatial_attention = nn.ModuleList([
            SpatialAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 2. TEMPORAL ATTENTION (Temporal Transformer)
        self.temporal_attention = nn.ModuleList([
            TemporalAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 3. EXTERNAL FEATURE FUSION (if enabled)
        if use_external_features:
            self.external_fusion = ExternalFeatureFusion(hidden_dim, num_heads, dropout)
        
        # 4. UNCERTAINTY QUANTIFICATION (if enabled)
        if use_uncertainty:
            self.uncertainty_head = UncertaintyHead(hidden_dim, output_dim, dropout)
        
        # 5. PREDICTION HEAD
        self.prediction_head = PredictionHead(hidden_dim, output_dim, pred_len)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights properly to prevent NaN
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights properly to prevent numerical instability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Small gain for stability
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Parameter):
                if module.dim() > 1:
                    nn.init.xavier_uniform_(module, gain=0.1)
                else:
                    nn.init.zeros_(module)
        
    def forward(self, 
                traffic_data: torch.Tensor,
                adj_matrix: torch.Tensor,
                external_features: Optional[Dict[str, torch.Tensor]] = None,
                return_uncertainty: bool = False):
        """
        Forward pass
        
        Args:
            traffic_data: [batch_size, num_nodes, seq_len, input_dim]
            adj_matrix: [num_nodes, num_nodes]
            external_features: Optional dictionary of external features
            return_uncertainty: Whether to return uncertainty estimates
        
        Returns:
            Dictionary with predictions and optional uncertainty
        """
        batch_size, num_nodes, seq_len, _ = traffic_data.shape
        
        # Debug: Check adjacency matrix
        if torch.all(adj_matrix == 0):
            print("WARNING: Adjacency matrix is all zeros! Creating identity matrix.")
            adj_matrix = torch.eye(num_nodes, device=adj_matrix.device)
        elif torch.isnan(adj_matrix).any() or torch.isinf(adj_matrix).any():
            print("WARNING: NaN/inf in adjacency matrix! Using identity matrix.")
            adj_matrix = torch.eye(num_nodes, device=adj_matrix.device)
        
        # Input projection
        x = self.input_projection(traffic_data)  # [B, N, T, H]
        
        # Debug: Check for NaN/inf after input projection
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN/inf detected after input projection!")
            print(f"Input data range: [{traffic_data.min():.6f}, {traffic_data.max():.6f}]")
            print(f"Projected data range: [{x.min():.6f}, {x.max():.6f}]")
            # Replace NaN/inf with small random values
            x = torch.where(torch.isnan(x) | torch.isinf(x), 
                           torch.randn_like(x) * 0.01, x)
        
        # 1. Spatial modeling with graph attention
        for spatial_layer in self.spatial_attention:
            x = spatial_layer(x, adj_matrix)
        
        # 2. Temporal modeling with causal attention
        for temporal_layer in self.temporal_attention:
            x = temporal_layer(x)
        
        # 3. External feature fusion (if available)
        if self.use_external_features and external_features is not None:
            x = self.external_fusion(x, external_features)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Use last timestep for prediction
        x_pred = x[:, :, -1, :]  # [B, N, H]
        
        # 4. Generate predictions
        predictions = self.prediction_head(x_pred)  # [B, N, pred_len, output_dim]
        
        results = {'predictions': predictions}
        
        # 5. Uncertainty estimation (if enabled)
        if self.use_uncertainty and return_uncertainty:
            uncertainty_mean, uncertainty_var = self.uncertainty_head(x_pred)
            results['uncertainty_mean'] = uncertainty_mean
            results['uncertainty_variance'] = uncertainty_var
        
        return results


class SpatialAttentionBlock(nn.Module):
    """Spatial attention block using graph attention mechanism"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor):
        """
        Args:
            x: [batch_size, num_nodes, seq_len, hidden_dim]
            adj_matrix: [num_nodes, num_nodes]
        """
        batch_size, num_nodes, seq_len, hidden_dim = x.shape
        
        # Process each timestep
        outputs = []
        for t in range(seq_len):
            x_t = x[:, :, t, :]  # [B, N, H]
            
            # Multi-head attention with adjacency constraint
            residual = x_t
            x_t = self.norm1(x_t)
            
            # Compute Q, K, V and reshape for multi-head attention
            q = self.q_proj(x_t).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x_t).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x_t).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
            # q, k, v shape: [B, H, N, D] where H is num_heads, D is head_dim
            
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # scores shape: [B, H, N, N]
            
            # Apply adjacency mask - ensure proper broadcasting
            # adj_matrix: [N, N] -> [1, 1, N, N] -> [B, H, N, N]
            adj_mask = adj_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
            adj_mask = adj_mask.expand(batch_size, self.num_heads, num_nodes, num_nodes)
            # Use large negative value instead of -inf to prevent NaN in softmax
            scores = scores.masked_fill(adj_mask == 0, -1e9)
            
            # Apply attention
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            attn_output = torch.matmul(attn_weights, v)  # [B, H, N, D]
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_nodes, hidden_dim)
            
            # Residual connection
            x_t = residual + self.dropout(self.out_proj(attn_output))
            
            # Feed-forward network
            residual = x_t
            x_t = self.norm2(x_t)
            x_t = residual + self.dropout(self.ffn(x_t))
            
            outputs.append(x_t)
        
        # Stack outputs back to [B, N, T, H]
        output = torch.stack(outputs, dim=2)
        return output


class TemporalAttentionBlock(nn.Module):
    """Temporal attention block using causal attention"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [batch_size, num_nodes, seq_len, hidden_dim]
        """
        batch_size, num_nodes, seq_len, hidden_dim = x.shape
        
        # Reshape for temporal attention: [B*N, T, H]
        x_reshaped = x.view(batch_size * num_nodes, seq_len, hidden_dim)
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(x.device)
        
        # Self-attention with causal mask
        residual = x_reshaped
        x_reshaped = self.norm1(x_reshaped)
        
        attn_output, _ = self.multihead_attn(
            x_reshaped, x_reshaped, x_reshaped,
            attn_mask=causal_mask,
            need_weights=False
        )
        
        x_reshaped = residual + self.dropout(attn_output)
        
        # Feed-forward network
        residual = x_reshaped
        x_reshaped = self.norm2(x_reshaped)
        x_reshaped = residual + self.dropout(self.ffn(x_reshaped))
        
        # Reshape back to [B, N, T, H]
        output = x_reshaped.view(batch_size, num_nodes, seq_len, hidden_dim)
        return output


class ExternalFeatureFusion(nn.Module):
    """Fusion module for external features (weather, incidents, POI)"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Feature processors
        self.weather_proj = nn.Linear(4, hidden_dim)  # weather features
        self.incident_proj = nn.Linear(3, hidden_dim)  # incident features
        self.poi_proj = nn.Linear(4, hidden_dim)  # POI features
        
        # Cross-attention for fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, 
                traffic_features: torch.Tensor, 
                external_features: Dict[str, torch.Tensor]):
        """
        Args:
            traffic_features: [batch_size, num_nodes, seq_len, hidden_dim]
            external_features: Dictionary with 'weather', 'incident', 'poi' features
        """
        batch_size, num_nodes, seq_len, hidden_dim = traffic_features.shape
        
        # Process external features
        fused_external = torch.zeros_like(traffic_features)
        
        if 'weather' in external_features:
            weather = self.weather_proj(external_features['weather'])
            fused_external = fused_external + weather
        
        if 'incident' in external_features:
            incident = self.incident_proj(external_features['incident'])
            fused_external = fused_external + incident
        
        if 'poi' in external_features:
            poi = external_features['poi']
            # POI is static, expand to sequence length
            if poi.dim() == 3:  # [B, N, poi_dim]
                poi = poi.unsqueeze(2).expand(-1, -1, seq_len, -1)
            poi_proj = self.poi_proj(poi)
            fused_external = fused_external + poi_proj
        
        # Cross-attention fusion
        traffic_flat = traffic_features.view(batch_size * num_nodes, seq_len, hidden_dim)
        external_flat = fused_external.view(batch_size * num_nodes, seq_len, hidden_dim)
        
        attended_external, _ = self.cross_attention(
            traffic_flat, external_flat, external_flat
        )
        attended_external = attended_external.view(batch_size, num_nodes, seq_len, hidden_dim)
        
        # Gating mechanism
        combined = torch.cat([traffic_features, attended_external], dim=-1)
        gate_weights = self.gate(combined)
        
        # Apply gating
        output = gate_weights * attended_external + (1 - gate_weights) * traffic_features
        output = self.layer_norm(output)
        
        return output


class UncertaintyHead(nn.Module):
    """Bayesian uncertainty quantification head"""
    
    def __init__(self, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        
        # Mean prediction head
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Variance prediction head
        self.var_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus()  # Ensure positive variance
        )
        
        # Monte Carlo Dropout for uncertainty estimation
        self.mc_dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, num_samples: int = 50):
        """
        Args:
            x: [batch_size, num_nodes, hidden_dim]
            num_samples: Number of MC samples for uncertainty estimation
        """
        if self.training:
            # During training, single forward pass
            x = self.mc_dropout(x)
            mean = self.mean_head(x)
            var = self.var_head(x)
            return mean, var
        else:
            # During inference, Monte Carlo sampling
            means = []
            for _ in range(num_samples):
                x_sample = self.mc_dropout(x)
                mean_sample = self.mean_head(x_sample)
                means.append(mean_sample)
            
            # Compute epistemic uncertainty from samples
            means = torch.stack(means, dim=0)  # [num_samples, B, N, output_dim]
            epistemic_mean = torch.mean(means, dim=0)
            epistemic_var = torch.var(means, dim=0)
            
            # Compute aleatoric uncertainty
            aleatoric_var = self.var_head(x)
            
            # Total uncertainty
            total_var = epistemic_var + aleatoric_var
            
            return epistemic_mean, total_var


class PredictionHead(nn.Module):
    """Multi-horizon prediction head"""
    
    def __init__(self, hidden_dim: int, output_dim: int, pred_len: int):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        
        # Prediction network
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_len * output_dim)
        )
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [batch_size, num_nodes, hidden_dim]
        
        Returns:
            predictions: [batch_size, num_nodes, pred_len, output_dim]
        """
        batch_size, num_nodes, hidden_dim = x.shape
        
        # Generate predictions
        pred_flat = self.predictor(x)  # [B, N, pred_len * output_dim]
        
        # Reshape to multi-step predictions
        predictions = pred_flat.view(batch_size, num_nodes, self.pred_len, self.output_dim)
        
        return predictions


# Utility function to create the model
def create_ustgt_model(config: Dict) -> USTGT:
    """
    Create USTGT model from configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        USTGT model instance
    """
    model_config = config.get('model', {})
    
    model = USTGT(
        num_nodes=model_config.get('num_nodes', 207),
        input_dim=model_config.get('input_dim', 1),
        hidden_dim=model_config.get('hidden_dim', 64),
        output_dim=model_config.get('output_dim', 1),
        seq_len=model_config.get('seq_len', 12),
        pred_len=model_config.get('pred_len', 12),
        num_heads=model_config.get('num_heads', 8),
        num_layers=model_config.get('num_layers', 3),
        dropout=model_config.get('dropout', 0.1),
        use_uncertainty=model_config.get('use_uncertainty', True),
        use_external_features=model_config.get('use_external_features', True)
    )
    
    return model

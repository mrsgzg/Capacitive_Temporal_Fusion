# -*- coding: utf-8 -*-
"""
Multi-task CNN + Transformer model for liquid + bottle classification
Shares the CNN+Transformer backbone and has two independent classification heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    
    def __init__(self, d_model: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, :, :], persistent=False)
    
    def _rotate_half(self, x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, x, seq_len: int = None):
        if seq_len is None:
            seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        cos = self.cos_cached[:, :seq_len, :]
        sin = self.sin_cached[:, :seq_len, :]
        return (x * cos) + (self._rotate_half(x) * sin)


class PositionalEncodingLearned(nn.Module):
    """Learnable positional encoding"""
    
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, seq_len, d_model) * (1.0 / math.sqrt(d_model)))
    
    def forward(self, x):
        return x + self.pe


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm, GELU, and Dropout"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x


class ResidualConvBlock(nn.Module):
    """Residual convolutional block"""
    
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = self.gelu(x)
        return x


class AttentionPooling(nn.Module):
    """Attention-based pooling"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
    
    def forward(self, x):
        weights = self.query(x)
        weights = F.softmax(weights, dim=1)
        output = (x * weights).sum(dim=1)
        return output


class StageWisePooling(nn.Module):
    """Extract features from precontact, closing, and hold stages"""
    
    def __init__(self, d_model: int, seq_len: int = 384):
        super().__init__()
        self.seq_len = seq_len
        self.precontact_end = int(seq_len * 0.3)
        self.closing_end = int(seq_len * 0.6)
        
        self.stage_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
        )
    
    def forward(self, x):
        precontact = x[:, :self.precontact_end, :].mean(dim=1)
        closing = x[:, self.precontact_end:self.closing_end, :].mean(dim=1)
        hold = x[:, self.closing_end:, :].mean(dim=1)
        combined = torch.cat([precontact, closing, hold], dim=1)
        output = self.stage_proj(combined)
        return output


class MultiHeadPooling(nn.Module):
    """Multiple pooling strategies combined"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.attention_pool = AttentionPooling(d_model)
    
    def forward(self, x):
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1)[0]
        attn_pool = self.attention_pool(x)
        return torch.cat([mean_pool, max_pool, attn_pool], dim=1)


class CNNTransformerMTL(nn.Module):
    """
    Multi-task CNN+Transformer model for liquid and bottle classification
    
    Returns two predictions in forward():
    - liquid_logits: (B, num_liquid_classes)
    - bottle_logits: (B, num_bottle_classes)
    
    Args:
        seq_len: Sequence length (default: 384)
        in_ch: Input channels (default: 4)
        d_model: Transformer hidden dimension (default: 512)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 8)
        dim_ff: Feed-forward dimension (default: 1024)
        dropout: Dropout rate (default: 0.12)
        num_liquid_classes: Number of liquid classes (default: 19)
        num_bottle_classes: Number of bottle types (default: 4)
        pooling_strategy: 'mean', 'attention', 'stage_wise', or 'multi'
        use_residual: Whether to use residual connections (default: True)
        use_rope: Whether to use RoPE (default: False)
    """
    
    def __init__(
        self,
        seq_len: int = 384,
        in_ch: int = 4,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dim_ff: int = 1024,
        dropout: float = 0.12,
        num_liquid_classes: int = 19,
        num_bottle_classes: int = 4,
        pooling_strategy: str = "stage_wise",
        use_residual: bool = True,
        use_rope: bool = False,
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.in_ch = in_ch
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_liquid_classes = num_liquid_classes
        self.num_bottle_classes = num_bottle_classes
        self.pooling_strategy = pooling_strategy
        self.use_residual = use_residual
        self.use_rope = use_rope
        
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        
        # ===== CNN Embedding Layer =====
        self.initial_conv = ConvBlock(in_ch, 64, kernel_size=7, dropout=dropout)
        self.conv_scale1 = ConvBlock(64, 128, kernel_size=5, dropout=dropout)
        
        if use_residual:
            self.residual_block = ResidualConvBlock(128, kernel_size=3, dropout=dropout)
        
        self.conv_scale2 = ConvBlock(128, d_model, kernel_size=3, dropout=dropout)
        
        # ===== Positional Encoding =====
        if use_rope:
            self.positional_encoding = RotaryPositionalEmbedding(d_model, max_seq_len=seq_len)
        else:
            self.positional_encoding = PositionalEncodingLearned(seq_len, d_model)
        
        # ===== Transformer Encoder =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        
        # ===== Pooling Layers =====
        if pooling_strategy == "mean":
            self.pooling = nn.AdaptiveAvgPool1d(1)
            pool_out_dim = d_model
        elif pooling_strategy == "attention":
            self.pooling = AttentionPooling(d_model)
            pool_out_dim = d_model
        elif pooling_strategy == "stage_wise":
            self.pooling = StageWisePooling(d_model, seq_len)
            pool_out_dim = d_model
        elif pooling_strategy == "multi":
            self.pooling = MultiHeadPooling(d_model)
            pool_out_dim = d_model * 3
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        # ===== Task 1: Liquid Classification Head =====
        self.ln_liquid = nn.LayerNorm(pool_out_dim)
        self.dropout_liquid = nn.Dropout(dropout)
        self.classifier_liquid = nn.Linear(pool_out_dim, num_liquid_classes)
        
        # ===== Task 2: Bottle Classification Head =====
        self.ln_bottle = nn.LayerNorm(pool_out_dim)
        self.dropout_bottle = nn.Dropout(dropout)
        self.classifier_bottle = nn.Linear(pool_out_dim, num_bottle_classes)
    
    def forward(self, x):
        """
        x: (B, C, L) - (batch_size, 4, 384)
        Returns: (liquid_logits, bottle_logits)
                 - liquid_logits: (B, num_liquid_classes)
                 - bottle_logits: (B, num_bottle_classes)
        """
        # CNN embedding: (B, in_ch, L) -> (B, d_model, L)
        x = self.initial_conv(x)
        x = self.conv_scale1(x)
        
        if self.use_residual:
            x = self.residual_block(x)
        
        x = self.conv_scale2(x)
        
        # Transpose for transformer: (B, d_model, L) -> (B, L, d_model)
        x = x.transpose(1, 2)
        
        # Positional encoding: (B, L, d_model)
        x = self.positional_encoding(x)
        
        # Transformer: (B, L, d_model) -> (B, L, d_model)
        x = self.transformer_encoder(x)
        
        # Pooling: (B, L, d_model) -> (B, pool_dim)
        if self.pooling_strategy == "mean":
            x = x.transpose(1, 2)
            x = self.pooling(x).squeeze(-1)
        else:
            x = self.pooling(x)
        
        # Shared representation: (B, pool_dim)
        
        # ===== Liquid Classification Head =====
        x_liquid = self.ln_liquid(x)
        x_liquid = self.dropout_liquid(x_liquid)
        liquid_logits = self.classifier_liquid(x_liquid)
        
        # ===== Bottle Classification Head =====
        x_bottle = self.ln_bottle(x)
        x_bottle = self.dropout_bottle(x_bottle)
        bottle_logits = self.classifier_bottle(x_bottle)
        
        return liquid_logits, bottle_logits

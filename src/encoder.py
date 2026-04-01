"""
Transformer Encoder
===================
基于 "Attention Is All You Need" (Vaswani et al., 2017)

Encoder 结构:
Input -> [Embedding + Positional Encoding] -> 
    [Multi-Head Attention -> Add&Norm -> FFN -> Add&Norm] × N -> Output
"""

import torch
import torch.nn as nn
from typing import Optional

from attention import MultiHeadAttention
from feedforward import PositionwiseFeedForward, SublayerConnection, LayerNorm


class EncoderLayer(nn.Module):
    """
    单层编码器
    
    结构:
    1. Multi-Head Self-Attention
    2. Add & Norm (残差连接 + 层归一化)
    3. Position-wise Feed-Forward
    4. Add & Norm
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.sublayer_connections = nn.ModuleList([
            SublayerConnection(d_model, dropout) for _ in range(2)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, src_len, d_model]
            src_mask: [batch_size, 1, src_len, src_len] (padding mask)
            
        Returns:
            output: [batch_size, src_len, d_model]
        """
        # 1. Self-Attention + Add&Norm
        x = self.sublayer_connections[0](
            x, lambda x: self.self_attn(x, x, x, src_mask)[0]
        )
        
        # 2. Feed-Forward + Add&Norm
        x = self.sublayer_connections[1](x, self.feed_forward)
        
        return x


class TransformerEncoder(nn.Module):
    """
    完整编码器 = N × EncoderLayer
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        n_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = LayerNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, src_len, d_model] (已包含位置编码)
            src_mask: [batch_size, 1, src_len, src_len]
            
        Returns:
            output: [batch_size, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


if __name__ == "__main__":
    # 测试代码
    batch_size, src_len, d_model = 2, 20, 512
    n_heads, n_layers = 8, 6
    
    # 模拟输入（已经过 embedding + positional encoding）
    x = torch.randn(batch_size, src_len, d_model)
    
    # 测试单层
    layer = EncoderLayer(d_model, n_heads, d_ff=2048)
    output = layer(x)
    print(f"Single layer output shape: {output.shape}")
    
    # 测试完整编码器
    encoder = TransformerEncoder(d_model, n_heads, d_ff=2048, n_layers=n_layers)
    output = encoder(x)
    print(f"Full encoder output shape: {output.shape}")
    
    # 测试带 mask
    src_mask = torch.ones(batch_size, 1, src_len, src_len)
    src_mask[:, :, :, 15:] = 0  # 模拟 padding
    output = encoder(x, src_mask)
    print(f"With mask output shape: {output.shape}")
    
    print("✓ Encoder tests passed!")

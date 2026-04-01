"""
Transformer Decoder
===================
基于 "Attention Is All You Need" (Vaswani et al., 2017)

Decoder 结构:
Input -> [Embedding + Positional Encoding] ->
    [Masked Self-Attn -> Add&Norm -> 
     Cross Attn -> Add&Norm -> 
     FFN -> Add&Norm] × N -> Output
"""

import torch
import torch.nn as nn
from typing import Optional

from attention import MultiHeadAttention
from feedforward import PositionwiseFeedForward, SublayerConnection, LayerNorm


class DecoderLayer(nn.Module):
    """
    单层解码器
    
    结构:
    1. Masked Multi-Head Self-Attention (防止看到未来信息)
    2. Add & Norm
    3. Multi-Head Cross-Attention (关注编码器输出)
    4. Add & Norm
    5. Position-wise Feed-Forward
    6. Add & Norm
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        # 三个子层
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.sublayer_connections = nn.ModuleList([
            SublayerConnection(d_model, dropout) for _ in range(3)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, tgt_len, d_model] (解码器输入)
            encoder_output: [batch_size, src_len, d_model] (编码器输出)
            tgt_mask: [batch_size, 1, tgt_len, tgt_len] (look-ahead + padding mask)
            src_mask: [batch_size, 1, src_len, src_len] (编码器 padding mask)
            
        Returns:
            output: [batch_size, tgt_len, d_model]
        """
        # 1. Masked Self-Attention
        x = self.sublayer_connections[0](
            x, lambda x: self.self_attn(x, x, x, tgt_mask)[0]
        )
        
        # 2. Cross-Attention (Q from decoder, K,V from encoder)
        x = self.sublayer_connections[1](
            x, lambda x: self.cross_attn(x, encoder_output, encoder_output, src_mask)[0]
        )
        
        # 3. Feed-Forward
        x = self.sublayer_connections[2](x, self.feed_forward)
        
        return x


class TransformerDecoder(nn.Module):
    """
    完整解码器 = N × DecoderLayer
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
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = LayerNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, tgt_len, d_model]
            encoder_output: [batch_size, src_len, d_model]
            tgt_mask: [batch_size, 1, tgt_len, tgt_len]
            src_mask: [batch_size, 1, src_len, src_len]
            
        Returns:
            output: [batch_size, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        return self.norm(x)


def create_look_ahead_mask(size: int) -> torch.Tensor:
    """
    创建 Look-ahead mask (防止解码器看到未来位置)
    
    返回一个下三角矩阵，例如 size=5:
    [[1, 0, 0, 0, 0],
     [1, 1, 0, 0, 0],
     [1, 1, 1, 0, 0],
     [1, 1, 1, 1, 0],
     [1, 1, 1, 1, 1]]
    """
    mask = torch.tril(torch.ones(size, size))
    return mask  # [size, size]


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    创建 Padding mask
    
    Args:
        seq: [batch_size, seq_len]
        pad_idx: padding token 的索引
        
    Returns:
        mask: [batch_size, 1, 1, seq_len]
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask  # [batch_size, 1, 1, seq_len]


def create_masks(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 0):
    """
    创建所有需要的 mask
    
    Returns:
        src_mask: [batch_size, 1, src_len, src_len]
        tgt_mask: [batch_size, 1, tgt_len, tgt_len]
    """
    # Source mask (padding mask)
    src_mask = create_padding_mask(src, pad_idx)  # [batch, 1, 1, src_len]
    src_mask = src_mask.expand(-1, -1, src.size(1), -1)  # [batch, 1, src_len, src_len]
    
    # Target mask (look-ahead + padding)
    tgt_pad_mask = create_padding_mask(tgt, pad_idx)  # [batch, 1, 1, tgt_len]
    tgt_look_ahead = create_look_ahead_mask(tgt.size(1)).to(tgt.device)  # [tgt_len, tgt_len]
    
    # 组合: padding mask & look-ahead mask
    tgt_mask = tgt_pad_mask & tgt_look_ahead.unsqueeze(0).unsqueeze(0).bool()
    # [batch, 1, tgt_len, tgt_len]
    
    return src_mask, tgt_mask


if __name__ == "__main__":
    # 测试代码
    batch_size, src_len, tgt_len, d_model = 2, 20, 15, 512
    n_heads, n_layers = 8, 6
    
    # 模拟输入
    x = torch.randn(batch_size, tgt_len, d_model)
    encoder_output = torch.randn(batch_size, src_len, d_model)
    
    # 测试单层
    layer = DecoderLayer(d_model, n_heads, d_ff=2048)
    output = layer(x, encoder_output)
    print(f"Single layer output shape: {output.shape}")
    
    # 测试完整解码器
    decoder = TransformerDecoder(d_model, n_heads, d_ff=2048, n_layers=n_layers)
    output = decoder(x, encoder_output)
    print(f"Full decoder output shape: {output.shape}")
    
    # 测试 mask 创建
    src = torch.randint(0, 100, (batch_size, src_len))
    tgt = torch.randint(0, 100, (batch_size, tgt_len))
    tgt[:, 10:] = 0  # 模拟 padding
    
    src_mask, tgt_mask = create_masks(src, tgt, pad_idx=0)
    print(f"Source mask shape: {src_mask.shape}")
    print(f"Target mask shape: {tgt_mask.shape}")
    
    # 使用 mask 测试
    output = decoder(x, encoder_output, tgt_mask, src_mask)
    print(f"With masks output shape: {output.shape}")
    
    print("✓ Decoder tests passed!")

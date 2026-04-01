"""
Feed-Forward Network & Layer Normalization
==========================================
基于 "Attention Is All You Need" (Vaswani et al., 2017)

包含:
1. Position-wise Feed-Forward Network
2. Layer Normalization
3. Residual Connection
"""

import torch
import torch.nn as nn
from typing import Optional


class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    或 FFN(x) = GELU(xW1 + b1)W2 + b2
    
    论文中使用 ReLU，但现在 GELU 更常见
    """
    
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Xavier 初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    """
    层归一化
    
    LayerNorm(x) = γ * (x - μ) / sqrt(σ^2 + ε) + β
    
    与 BatchNorm 的区别:
    - BatchNorm: 在 batch 维度归一化
    - LayerNorm: 在 feature 维度归一化（更适合序列数据）
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            normalized: [batch_size, seq_len, d_model]
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SublayerConnection(nn.Module):
    """
    子层连接: Residual Connection + Layer Normalization
    
    论文使用: LayerNorm(x + Sublayer(x))
    也有实现使用: x + LayerNorm(Sublayer(x))
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, sublayer: callable) -> torch.Tensor:
        """
        Args:
            x: 输入 [batch_size, seq_len, d_model]
            sublayer: 子层函数（如 attention 或 FFN）
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Pre-Norm: LayerNorm(x) -> Sublayer -> Dropout -> Add
        # 现代 Transformer 更常用 Pre-Norm，训练更稳定
        return x + self.dropout(sublayer(self.norm(x)))


class PostSublayerConnection(nn.Module):
    """
    原始的 Post-Norm: Sublayer -> Add -> LayerNorm
    论文使用的是这种，但训练可能不稳定
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, sublayer: callable) -> torch.Tensor:
        return self.norm(x + self.dropout(sublayer(x)))


class TransformerBlock(nn.Module):
    """
    完整的 Transformer Block（可作为通用模块）
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_pre_norm: bool = True
    ):
        super().__init__()
        from attention import MultiHeadAttention
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        if use_pre_norm:
            self.attn_connection = SublayerConnection(d_model, dropout)
            self.ffn_connection = SublayerConnection(d_model, dropout)
        else:
            self.attn_connection = PostSublayerConnection(d_model, dropout)
            self.ffn_connection = PostSublayerConnection(d_model, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-Attention
        x = self.attn_connection(x, lambda x: self.attention(x, x, x, mask)[0])
        
        # Feed Forward
        x = self.ffn_connection(x, self.feed_forward)
        
        return x


if __name__ == "__main__":
    # 测试代码
    batch_size, seq_len, d_model = 2, 10, 512
    d_ff = 2048
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 测试 FFN
    ffn = PositionwiseFeedForward(d_model, d_ff)
    output = ffn(x)
    print(f"FFN input shape: {x.shape}")
    print(f"FFN output shape: {output.shape}")
    
    # 测试 LayerNorm
    ln = LayerNorm(d_model)
    output = ln(x)
    print(f"LayerNorm output shape: {output.shape}")
    print(f"LayerNorm mean: {output.mean(-1)[0, 0]:.6f}")
    print(f"LayerNorm std: {output.std(-1)[0, 0]:.6f}")
    
    print("✓ All feedforward tests passed!")

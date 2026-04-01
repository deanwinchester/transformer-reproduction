"""
Multi-Head Attention Implementation
====================================
Based on "Attention Is All You Need" (Vaswani et al., 2017)

Key features:
- Scaled Dot-Product Attention
- Multi-head parallel attention
- Support for padding masks and look-ahead masks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力机制
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, n_heads, seq_len, d_k]
            key: [batch_size, n_heads, seq_len, d_k]
            value: [batch_size, n_heads, seq_len, d_v]
            mask: [batch_size, 1, seq_len, seq_len] or broadcastable
            
        Returns:
            output: [batch_size, n_heads, seq_len, d_v]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        d_k = query.size(-1)
        
        # 1. 计算 QK^T / sqrt(d_k)
        # [batch, n_heads, seq_len, d_k] @ [batch, n_heads, d_k, seq_len]
        # = [batch, n_heads, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 2. 应用 mask（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 3. Softmax 归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 4. 与 V 相乘
        # [batch, n_heads, seq_len, seq_len] @ [batch, n_heads, seq_len, d_v]
        # = [batch, n_heads, seq_len, d_v]
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    允许模型在不同位置共同关注来自不同表示子空间的信息。
    """
    
    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # 线性投影层: W_Q, W_K, W_V, W_O
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Xavier 初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len, seq_len]
            
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        
        # 1. 线性投影并分头
        # [batch, seq_len, d_model] -> [batch, seq_len, n_heads, d_k] -> [batch, n_heads, seq_len, d_k]
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        
        # 3. 拼接多头输出
        # [batch, n_heads, seq_len, d_k] -> [batch, seq_len, n_heads, d_k] -> [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 4. 最终线性投影
        output = self.w_o(attn_output)
        output = self.dropout(output)
        
        return output, attention_weights


class SelfAttention(MultiHeadAttention):
    """自注意力（Query = Key = Value）"""
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return super().forward(x, x, x, mask)


if __name__ == "__main__":
    # 测试代码
    batch_size, seq_len, d_model = 2, 10, 512
    n_heads = 8
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 测试多头注意力
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    output, attn_weights = mha(x, x, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"✓ Multi-Head Attention test passed!")

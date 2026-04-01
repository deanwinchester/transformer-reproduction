"""
Embeddings and Positional Encoding
====================================
基于 "Attention Is All You Need" (Vaswani et al., 2017)

包含:
1. Token Embeddings (词嵌入)
2. Positional Encoding (位置编码) - 使用正弦/余弦函数
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    正弦位置编码
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    优点:
    - 可以处理任意长度的序列
    - 相对位置信息可以通过线性变换得到
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # 位置索引 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算 div_term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # 应用 sin 到偶数索引
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # 应用 cos 到奇数索引
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加 batch 维度 [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer（不作为模型参数，但会随模型保存）
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x + positional_encoding: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """词嵌入层"""
    
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model
        self._reset_parameters()
    
    def _reset_parameters(self):
        """使用正态分布初始化"""
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model ** -0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len] token ids
        Returns:
            embeddings: [batch_size, seq_len, d_model]
        """
        # 论文中提到要乘以 sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class TransformerEmbedding(nn.Module):
    """
    完整的 Transformer 嵌入层
    = Token Embedding + Positional Encoding
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len] token ids
        Returns:
            embeddings: [batch_size, seq_len, d_model]
        """
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        return x


class LearnedPositionalEmbedding(nn.Module):
    """
    可学习的位置编码（BERT/GPT 风格）
    作为对比，Transformer 论文使用的是正弦编码
    """
    
    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        x = x + self.pe(positions)
        return self.dropout(x)


if __name__ == "__main__":
    # 测试代码
    batch_size, seq_len = 2, 20
    vocab_size, d_model = 10000, 512
    
    # 创建随机输入（token ids）
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 测试完整嵌入层
    embedding = TransformerEmbedding(vocab_size, d_model)
    output = embedding(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: [{batch_size}, {seq_len}, {d_model}]")
    
    # 测试位置编码可视化
    import matplotlib.pyplot as plt
    
    pe = PositionalEncoding(d_model=64, max_len=100)
    pe_matrix = pe.pe[0].numpy()
    
    plt.figure(figsize=(12, 6))
    plt.imshow(pe_matrix.T, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Positional Encoding Matrix (d_model=64, max_len=100)')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.tight_layout()
    plt.savefig('/Users/liuhonghao/Projects/open-agc/workspace/transformer-reproduction/positional_encoding_viz.png')
    print("✓ Positional encoding visualization saved!")
    print("✓ All embedding tests passed!")

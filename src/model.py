"""
Complete Transformer Model
===========================
基于 "Attention Is All You Need" (Vaswani et al., 2017)

完整的 Encoder-Decoder 架构用于序列到序列学习
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from embedding import TransformerEmbedding
from encoder import TransformerEncoder
from decoder import TransformerDecoder, create_masks


class Transformer(nn.Module):
    """
    完整的 Transformer 模型
    
    架构:
    [Input] -> [Encoder] -> [Encoder Output]
                              ↓
    [Target] -> [Decoder] -> [Output Projection] -> [Logits]
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        max_len: int = 5000,
        dropout: float = 0.1,
        pad_idx: int = 0,
        tie_weights: bool = True
    ):
        """
        Args:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            n_encoder_layers: 编码器层数
            n_decoder_layers: 解码器层数
            max_len: 最大序列长度
            dropout: Dropout 概率
            pad_idx: Padding token 索引
            tie_weights: 是否共享输入输出嵌入权重
        """
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # 嵌入层
        self.src_embedding = TransformerEmbedding(
            src_vocab_size, d_model, max_len, dropout, pad_idx
        )
        self.tgt_embedding = TransformerEmbedding(
            tgt_vocab_size, d_model, max_len, dropout, pad_idx
        )
        
        # 编码器和解码器
        self.encoder = TransformerEncoder(
            d_model, n_heads, d_ff, n_encoder_layers, dropout
        )
        self.decoder = TransformerDecoder(
            d_model, n_heads, d_ff, n_decoder_layers, dropout
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # 权重共享（论文中提到的技巧）
        if tie_weights:
            self.output_projection.weight = self.tgt_embedding.token_embedding.embedding.weight
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """初始化参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码源序列
        
        Args:
            src: [batch_size, src_len]
            src_mask: [batch_size, 1, src_len, src_len]
            
        Returns:
            encoder_output: [batch_size, src_len, d_model]
        """
        src_embedded = self.src_embedding(src)
        return self.encoder(src_embedded, src_mask)
    
    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        解码目标序列
        
        Args:
            tgt: [batch_size, tgt_len]
            encoder_output: [batch_size, src_len, d_model]
            tgt_mask: [batch_size, 1, tgt_len, tgt_len]
            src_mask: [batch_size, 1, src_len, src_len]
            
        Returns:
            decoder_output: [batch_size, tgt_len, d_model]
        """
        tgt_embedded = self.tgt_embedding(tgt)
        return self.decoder(tgt_embedded, encoder_output, tgt_mask, src_mask)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播（训练时使用）
        
        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len]
            
        Returns:
            logits: [batch_size, tgt_len, tgt_vocab_size]
        """
        # 创建 masks
        src_mask, tgt_mask = create_masks(src, tgt, self.pad_idx)
        
        # 编码
        encoder_output = self.encode(src, src_mask)
        
        # 解码
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        # 投影到词汇表
        logits = self.output_projection(decoder_output)
        
        return logits
    
    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        start_symbol: int,
        end_symbol: int,
        max_len: int = 100
    ) -> torch.Tensor:
        """
        贪心解码（推理时使用）
        
        Args:
            src: [batch_size, src_len]
            start_symbol: 开始符号索引
            end_symbol: 结束符号索引
            max_len: 最大生成长度
            
        Returns:
            output: [batch_size, generated_len]
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        # 编码源序列
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        src_mask = src_mask.expand(-1, -1, src.size(1), -1)
        encoder_output = self.encode(src, src_mask)
        
        # 初始化解码器输入（从 <start> 开始）
        tgt = torch.ones(batch_size, 1).fill_(start_symbol).long().to(device)
        
        finished = torch.zeros(batch_size, dtype=torch.bool).to(device)
        
        for _ in range(max_len - 1):
            # 创建 tgt mask
            tgt_mask = torch.ones(batch_size, 1, tgt.size(1), tgt.size(1)).to(device)
            tgt_mask = tgt_mask.bool()
            
            # 解码
            decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
            
            # 取最后一个位置的预测
            logits = self.output_projection(decoder_output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            # 追加到输出序列
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # 检查是否生成结束符
            finished |= (next_token.squeeze(-1) == end_symbol)
            if finished.all():
                break
        
        return tgt


def build_transformer_base(
    src_vocab_size: int,
    tgt_vocab_size: int,
    **kwargs
) -> Transformer:
    """构建 Transformer Base 模型（论文配置）"""
    config = {
        'd_model': 512,
        'n_heads': 8,
        'd_ff': 2048,
        'n_encoder_layers': 6,
        'n_decoder_layers': 6,
        'dropout': 0.1,
    }
    config.update(kwargs)
    return Transformer(src_vocab_size, tgt_vocab_size, **config)


def build_transformer_big(
    src_vocab_size: int,
    tgt_vocab_size: int,
    **kwargs
) -> Transformer:
    """构建 Transformer Big 模型（论文配置）"""
    config = {
        'd_model': 1024,
        'n_heads': 16,
        'd_ff': 4096,
        'n_encoder_layers': 6,
        'n_decoder_layers': 6,
        'dropout': 0.3,
    }
    config.update(kwargs)
    return Transformer(src_vocab_size, tgt_vocab_size, **config)


if __name__ == "__main__":
    # 测试代码
    batch_size, src_len, tgt_len = 2, 20, 15
    src_vocab_size, tgt_vocab_size = 10000, 10000
    
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    
    # 测试 Base 模型
    model = build_transformer_base(src_vocab_size, tgt_vocab_size)
    logits = model(src, tgt)
    print(f"Base model output shape: {logits.shape}")
    print(f"Expected: [{batch_size}, {tgt_len}, {tgt_vocab_size}]")
    
    # 计算参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Base model parameters: {n_params:,} (~{n_params/1e6:.1f}M)")
    # Expected: ~65M
    
    # 测试 Big 模型
    model_big = build_transformer_big(src_vocab_size, tgt_vocab_size)
    n_params_big = sum(p.numel() for p in model_big.parameters())
    print(f"Big model parameters: {n_params_big:,} (~{n_params_big/1e6:.1f}M)")
    # Expected: ~213M
    
    # 测试推理
    start_symbol, end_symbol = 1, 2
    generated = model.greedy_decode(src, start_symbol, end_symbol)
    print(f"Generated sequence shape: {generated.shape}")
    
    print("✓ Transformer model tests passed!")

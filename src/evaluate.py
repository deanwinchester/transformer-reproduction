"""
Transformer Evaluation Script
=============================
评估指标:
- BLEU (1-4)
- Perplexity
- Translation quality
"""

import os
import yaml
import torch
import argparse
from typing import List, Dict
from tqdm import tqdm
import math

from model import build_transformer_base, build_transformer_big
from utils import load_checkpoint


def calculate_bleu(references: List[List[str]], hypotheses: List[str]) -> Dict[str, float]:
    """
    计算 BLEU 分数
    
    使用 NLTK 或 sacrebleu
    """
    try:
        from sacrebleu import corpus_bleu
        bleu = corpus_bleu(hypotheses, [references])
        return {
            'bleu': bleu.score,
            'bleu_1': bleu.precisions[0],
            'bleu_2': bleu.precisions[1],
            'bleu_3': bleu.precisions[2],
            'bleu_4': bleu.precisions[3],
        }
    except ImportError:
        print("Warning: sacrebleu not installed. Install with: pip install sacrebleu")
        return {}


def calculate_perplexity(loss: float) -> float:
    """计算困惑度"""
    return math.exp(loss)


@torch.no_grad()
def greedy_decode_batch(
    model,
    src: torch.Tensor,
    start_symbol: int,
    end_symbol: int,
    max_len: int = 100,
    pad_idx: int = 0
) -> torch.Tensor:
    """批量贪心解码"""
    model.eval()
    batch_size = src.size(0)
    device = src.device
    
    # 编码
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    src_mask = src_mask.expand(-1, -1, src.size(1), -1)
    encoder_output = model.encode(src, src_mask)
    
    # 初始化
    tgt = torch.ones(batch_size, 1).fill_(start_symbol).long().to(device)
    finished = torch.zeros(batch_size, dtype=torch.bool).to(device)
    
    for _ in range(max_len - 1):
        tgt_mask = torch.ones(batch_size, 1, tgt.size(1), tgt.size(1)).to(device).bool()
        
        decoder_output = model.decode(tgt, encoder_output, tgt_mask, src_mask)
        logits = model.output_projection(decoder_output[:, -1, :])
        next_token = logits.argmax(dim=-1, keepdim=True)
        
        tgt = torch.cat([tgt, next_token], dim=1)
        
        finished |= (next_token.squeeze(-1) == end_symbol)
        if finished.all():
            break
    
    return tgt


def evaluate_translation(
    model,
    dataloader,
    src_tokenizer,
    tgt_tokenizer,
    device,
    start_symbol: int = 1,
    end_symbol: int = 2
):
    """评估翻译质量"""
    model.eval()
    
    hypotheses = []
    references = []
    total_loss = 0.0
    total_tokens = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        
        # 生成翻译
        generated = greedy_decode_batch(model, src, start_symbol, end_symbol)
        
        # 解码为文本
        for i in range(src.size(0)):
            hyp = tgt_tokenizer.decode(generated[i].cpu().tolist(), skip_special_tokens=True)
            ref = tgt_tokenizer.decode(tgt[i].cpu().tolist(), skip_special_tokens=True)
            
            hypotheses.append(hyp)
            references.append(ref)
    
    # 计算 BLEU
    bleu_scores = calculate_bleu(references, hypotheses)
    
    # 保存翻译结果
    return {
        'bleu_scores': bleu_scores,
        'hypotheses': hypotheses,
        'references': references
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Transformer')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='data/wmt14')
    parser.add_argument('--output', type=str, default='translations.txt')
    parser.add_argument('--beam-size', type=int, default=1, help='Beam search size (1=greedy)')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    if config['model_size'] == 'base':
        model = build_transformer_base(
            src_vocab_size=config['src_vocab_size'],
            tgt_vocab_size=config['tgt_vocab_size']
        )
    else:
        model = build_transformer_big(
            src_vocab_size=config['src_vocab_size'],
            tgt_vocab_size=config['tgt_vocab_size']
        )
    
    # 加载检查点
    load_checkpoint(args.checkpoint, model)
    model = model.to(device)
    
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # 评估（需要实现数据加载）
    # results = evaluate_translation(model, test_loader, ...)
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()

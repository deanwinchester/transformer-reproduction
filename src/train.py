"""
Transformer Training Script
===========================
复现 "Attention Is All You Need" 训练流程

特性:
- Label Smoothing
- Learning Rate Scheduling (warmup + inverse sqrt)
- Multi-GPU distributed training
- Gradient clipping
- Mixed precision training (optional)
"""

import os
import yaml
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional
import argparse
from tqdm import tqdm

from model import build_transformer_base, build_transformer_big
from utils import setup_logger, save_checkpoint, load_checkpoint


class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失
    
    论文中使用 ε_ls = 0.1
    """
    
    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch_size * seq_len, vocab_size]
            target: [batch_size * seq_len]
        """
        # 忽略 padding
        non_pad_mask = (target != self.padding_idx)
        n_tokens = non_pad_mask.sum()
        
        if n_tokens == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # 应用 log softmax
        log_probs = torch.log_softmax(pred, dim=-1)
        
        # 计算损失
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        
        smooth_loss = -log_probs.sum(dim=-1)
        
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss / self.vocab_size
        loss = loss.masked_select(non_pad_mask).sum() / n_tokens
        
        return loss


class NoamLRScheduler:
    """
    Noam 学习率调度器
    
    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    
    特点:
    - 线性 warmup
    - 然后按 step^(-0.5) 衰减
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0
        
    def step(self):
        """更新学习率"""
        self.step_num += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def _compute_lr(self) -> float:
        step = max(1, self.step_num)
        lr = self.factor * (
            self.d_model ** (-0.5) *
            min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )
        return lr
    
    def state_dict(self):
        return {
            'step_num': self.step_num,
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
            'factor': self.factor
        }
    
    def load_state_dict(self, state_dict):
        self.step_num = state_dict['step_num']
        self.d_model = state_dict['d_model']
        self.warmup_steps = state_dict['warmup_steps']
        self.factor = state_dict['factor']


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Adam,
    scheduler: NoamLRScheduler,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    max_grad_norm: float = 1.0,
    logger=None
) -> Dict[str, float]:
    """训练一个 epoch"""
    
    model.train()
    total_loss = 0.0
    total_tokens = 0
    
    pbar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(pbar):
        src = batch['src'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        
        # 前向传播（混合精度）
        if scaler is not None:
            with autocast():
                logits = model(src, tgt_input)
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    tgt_output.view(-1)
                )
        else:
            logits = model(src, tgt_input)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                tgt_output.view(-1)
            )
        
        # 反向传播
        optimizer.zero_grad()
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        scheduler.step()
        
        # 统计
        n_tokens = (tgt_output != 0).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler._compute_lr():.2e}'
        })
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    
    return {'loss': avg_loss, 'lr': scheduler._compute_lr()}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """验证"""
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for batch in tqdm(dataloader, desc="Validating"):
        src = batch['src'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        
        logits = model(src, tgt_input)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            tgt_output.view(-1)
        )
        
        n_tokens = (tgt_output != 0).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    
    return {'loss': avg_loss}


def main():
    parser = argparse.ArgumentParser(description='Train Transformer')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--data-dir', type=str, default='data/wmt14', help='Data directory')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Checkpoint save directory')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger('transformer_train', os.path.join(args.log_dir, 'train.log'))
    logger.info(f"Config: {config}")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 创建模型
    if config['model_size'] == 'base':
        model = build_transformer_base(
            src_vocab_size=config['src_vocab_size'],
            tgt_vocab_size=config['tgt_vocab_size'],
            max_len=config.get('max_len', 5000),
            dropout=config.get('dropout', 0.1)
        )
    else:
        model = build_transformer_big(
            src_vocab_size=config['src_vocab_size'],
            tgt_vocab_size=config['tgt_vocab_size'],
            max_len=config.get('max_len', 5000),
            dropout=config.get('dropout', 0.3)
        )
    
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    
    # 损失函数
    criterion = LabelSmoothingCrossEntropy(
        vocab_size=config['tgt_vocab_size'],
        padding_idx=0,
        smoothing=config.get('label_smoothing', 0.1)
    )
    
    # 优化器
    optimizer = Adam(
        model.parameters(),
        lr=0,  # 将由 scheduler 控制
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # 学习率调度器
    scheduler = NoamLRScheduler(
        optimizer,
        d_model=config['d_model'],
        warmup_steps=config.get('warmup_steps', 4000)
    )
    
    # 混合精度
    scaler = GradScaler() if args.fp16 and torch.cuda.is_available() else None
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # 数据加载（这里需要根据实际情况实现）
    # from data.dataset import WMT14Dataset, collate_fn
    # train_dataset = WMT14Dataset(args.data_dir, split='train')
    # val_dataset = WMT14Dataset(args.data_dir, split='valid')
    # train_loader = DataLoader(...)
    # val_loader = DataLoader(...)
    
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, config['epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, scaler, config.get('max_grad_norm', 1.0), logger
        )
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, LR: {train_metrics['lr']:.2e}")
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device)
        logger.info(f"Valid - Loss: {val_metrics['loss']:.4f}")
        
        # 保存检查点
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
        
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'config': config
        }, is_best, args.save_dir)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

"""
Dataset Classes for WMT14
=========================
PyTorch Dataset and DataLoader for machine translation
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import random


class TranslationDataset(Dataset):
    """
    机器翻译数据集
    
    格式:
    - 源语言文件: train.en
    - 目标语言文件: train.de
    """
    
    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        src_tokenizer,
        tgt_tokenizer,
        max_len: int = 100,
        sort_by_length: bool = False
    ):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        
        # 读取数据
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_sentences = [line.strip() for line in f]
        
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_sentences = [line.strip() for line in f]
        
        assert len(self.src_sentences) == len(self.tgt_sentences), \
            "Source and target files must have same number of lines"
        
        # 过滤超长句子
        self.data = []
        for src, tgt in zip(self.src_sentences, self.tgt_sentences):
            src_len = len(src_tokenizer.encode(src))
            tgt_len = len(tgt_tokenizer.encode(tgt))
            if src_len <= max_len and tgt_len <= max_len:
                self.data.append((src, tgt))
        
        print(f"Loaded {len(self.data)} sentence pairs (filtered from {len(self.src_sentences)})")
        
        # 按长度排序（用于更高效的批处理）
        if sort_by_length:
            self.data.sort(key=lambda x: len(src_tokenizer.encode(x[0])))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        
        # 编码
        src_ids = self.src_tokenizer.encode(src_text)
        tgt_ids = self.tgt_tokenizer.encode(tgt_text)
        
        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt': torch.tensor(tgt_ids, dtype=torch.long),
            'src_text': src_text,
            'tgt_text': tgt_text
        }


def collate_fn(batch: List[Dict], pad_idx: int = 0) -> Dict[str, torch.Tensor]:
    """
    批次处理函数
    
    将不同长度的序列填充到相同长度
    """
    # 分离源和目标
    src_sequences = [item['src'] for item in batch]
    tgt_sequences = [item['tgt'] for item in batch]
    
    # 计算最大长度
    src_max_len = max(len(s) for s in src_sequences)
    tgt_max_len = max(len(t) for t in tgt_sequences)
    
    # 填充
    src_padded = torch.full((len(batch), src_max_len), pad_idx, dtype=torch.long)
    tgt_padded = torch.full((len(batch), tgt_max_len), pad_idx, dtype=torch.long)
    
    for i, (src, tgt) in enumerate(zip(src_sequences, tgt_sequences)):
        src_padded[i, :len(src)] = src
        tgt_padded[i, :len(tgt)] = tgt
    
    # 目标序列：输入是 [start, ...]，输出是 [..., end]
    # 通常 start_symbol = 1, end_symbol = 2
    tgt_input = tgt_padded[:, :-1]  # 去掉最后一个
    tgt_output = tgt_padded[:, 1:]  # 去掉第一个（start symbol）
    
    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'tgt_input': tgt_input,
        'tgt_output': tgt_output
    }


class BucketingTranslationDataset(Dataset):
    """
    使用桶（bucketing）的翻译数据集
    
    将相似长度的句子分到同一个桶，减少填充
    """
    
    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        src_tokenizer,
        tgt_tokenizer,
        max_len: int = 100,
        bucket_boundaries: List[int] = None
    ):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        
        # 默认桶边界
        if bucket_boundaries is None:
            bucket_boundaries = [20, 40, 60, 80, 100]
        self.bucket_boundaries = bucket_boundaries
        
        # 读取数据
        with open(src_file, 'r', encoding='utf-8') as f:
            src_sentences = [line.strip() for line in f]
        
        with open(tgt_file, 'r', encoding='utf-8') as f:
            tgt_sentences = [line.strip() for line in f]
        
        # 分配到桶
        self.buckets = {bound: [] for bound in bucket_boundaries + [max_len + 1]}
        
        for src, tgt in zip(src_sentences, tgt_sentences):
            src_len = len(src_tokenizer.encode(src))
            tgt_len = len(tgt_tokenizer.encode(tgt))
            
            if src_len > max_len or tgt_len > max_len:
                continue
            
            # 找到合适的桶
            bucket_idx = next(i for i, bound in enumerate(bucket_boundaries) 
                            if src_len <= bound)
            bucket_key = bucket_boundaries[bucket_idx]
            
            self.buckets[bucket_key].append((src, tgt))
        
        # 展平
        self.data = []
        for bound in bucket_boundaries:
            self.data.extend(self.buckets[bound])
        
        print(f"Loaded {len(self.data)} sentence pairs into {len(bucket_boundaries)} buckets")
        for bound in bucket_boundaries:
            print(f"  Bucket <= {bound}: {len(self.buckets[bound])} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        
        src_ids = self.src_tokenizer.encode(src_text)
        tgt_ids = self.tgt_tokenizer.encode(tgt_text)
        
        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt': torch.tensor(tgt_ids, dtype=torch.long),
            'src_text': src_text,
            'tgt_text': tgt_text
        }


def create_dataloaders(
    data_dir: str,
    src_tokenizer,
    tgt_tokenizer,
    batch_size: int = 64,
    max_len: int = 100,
    num_workers: int = 4,
    use_bucket: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建 DataLoader
    
    Returns:
        train_loader, val_loader, test_loader
    """
    splits = ['train', 'valid', 'test']
    dataloaders = []
    
    for split in splits:
        src_file = os.path.join(data_dir, split, f'{split}.en')
        tgt_file = os.path.join(data_dir, split, f'{split}.de')
        
        if not os.path.exists(src_file):
            print(f"Warning: {src_file} not found, skipping {split}")
            dataloaders.append(None)
            continue
        
        if use_bucket and split == 'train':
            dataset = BucketingTranslationDataset(
                src_file, tgt_file, src_tokenizer, tgt_tokenizer, max_len
            )
        else:
            dataset = TranslationDataset(
                src_file, tgt_file, src_tokenizer, tgt_tokenizer, max_len
            )
        
        shuffle = (split == 'train')
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        dataloaders.append(loader)
    
    return tuple(dataloaders)


if __name__ == "__main__":
    # 测试代码
    class DummyTokenizer:
        def encode(self, text):
            return [ord(c) % 100 for c in text.split()]
    
    # 创建临时测试文件
    os.makedirs('test_data', exist_ok=True)
    with open('test_data/test.en', 'w') as f:
        for i in range(100):
            f.write(f"This is sentence number {i} .\n")
    with open('test_data/test.de', 'w') as f:
        for i in range(100):
            f.write(f"Das ist Satz Nummer {i} .\n")
    
    # 测试数据集
    tokenizer = DummyTokenizer()
    dataset = TranslationDataset(
        'test_data/test.en',
        'test_data/test.de',
        tokenizer,
        tokenizer,
        max_len=50
    )
    
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample src: {sample['src']}")
    print(f"Sample tgt: {sample['tgt']}")
    
    # 测试 DataLoader
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"Batch src shape: {batch['src'].shape}")
    print(f"Batch tgt shape: {batch['tgt'].shape}")
    
    print("✓ Dataset tests passed!")

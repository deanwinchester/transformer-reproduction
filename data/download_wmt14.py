"""
WMT14 Dataset Download Script
=============================

Downloads WMT14 English-German dataset for machine translation.

Data source:
- Training: Europarl v7, Common Crawl corpus, News Commentary v9
- Validation: newstest2013
- Test: newstest2014

Expected BLEU scores (tokenized, cased):
- newstest2014 (En-De): ~20-21 BLEU (sacrebleu)
"""

import os
import argparse
import urllib.request
import gzip
import shutil
from pathlib import Path
from typing import Optional


WMT14_URLS = {
    'train': {
        'en': 'http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.en.shuffled.gz',
        'de': 'http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.de.shuffled.gz',
    },
    'valid': {
        'en': 'http://www.statmt.org/wmt14/dev/news-test2013.en',
        'de': 'http://www.statmt.org/wmt14/dev/news-test2013.de',
    },
    'test': {
        'en': 'http://www.statmt.org/wmt14/test-full/newstest2014.en',
        'de': 'http://www.statmt.org/wmt14/test-full/newstest2014.de',
    }
}

# Alternative: Use pre-processed WMT14 from HuggingFace
HUGGINGFACE_DATASETS = {
    'wmt14': 'wmt14',
    'wmt14_de_en': 'wmt14',
}


def download_file(url: str, dest_path: str, desc: Optional[str] = None):
    """下载文件"""
    print(f"Downloading {desc or url}...")
    print(f"  -> {dest_path}")
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"  ✓ Downloaded successfully")
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False
    return True


def decompress_gz(gz_path: str, dest_path: str):
    """解压 .gz 文件"""
    print(f"Decompressing {gz_path}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(dest_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"  ✓ Decompressed to {dest_path}")


def download_wmt14_statmt(save_dir: str):
    """从 statmt.org 下载 WMT14"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for split in ['valid', 'test']:
        split_dir = save_path / split
        split_dir.mkdir(exist_ok=True)
        
        for lang in ['en', 'de']:
            url = WMT14_URLS[split][lang]
            filename = os.path.basename(url)
            dest_path = split_dir / filename
            
            if not dest_path.exists():
                download_file(url, str(dest_path), f"{split}.{lang}")
            else:
                print(f"  ✓ {split}.{lang} already exists")
    
    print("\n✓ WMT14 dataset download completed!")
    print(f"\nNote: Full training data (~4GB compressed) can be downloaded from:")
    print("  http://www.statmt.org/wmt14/translation-task.html")


def download_wmt14_huggingface(save_dir: str, lang_pair: str = 'de-en'):
    """使用 HuggingFace datasets 下载 WMT14"""
    try:
        from datasets import load_dataset
        import json
        
        print(f"Downloading WMT14 ({lang_pair}) from HuggingFace...")
        
        # 加载数据集
        dataset = load_dataset('wmt14', lang_pair, cache_dir=save_dir)
        
        save_path = Path(save_dir) / 'processed'
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存为文本文件
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                split_dir = save_path / split
                split_dir.mkdir(exist_ok=True)
                
                src_lang, tgt_lang = lang_pair.split('-')
                
                src_file = split_dir / f'{split}.{src_lang}'
                tgt_file = split_dir / f'{split}.{tgt_lang}'
                
                with open(src_file, 'w', encoding='utf-8') as f_src, \
                     open(tgt_file, 'w', encoding='utf-8') as f_tgt:
                    
                    for example in dataset[split]:
                        src_text = example['translation'][src_lang]
                        tgt_text = example['translation'][tgt_lang]
                        f_src.write(src_text + '\n')
                        f_tgt.write(tgt_text + '\n')
                
                print(f"  ✓ Saved {split}: {src_file}, {tgt_file}")
        
        # 保存元数据
        metadata = {
            'dataset': 'wmt14',
            'lang_pair': lang_pair,
            'splits': list(dataset.keys()),
            'sizes': {k: len(v) for k, v in dataset.items()}
        }
        
        with open(save_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n✓ WMT14 dataset processed successfully!")
        print(f"\nDataset statistics:")
        for split, size in metadata['sizes'].items():
            print(f"  {split}: {size:,} examples")
        
        return True
        
    except ImportError:
        print("HuggingFace datasets not installed.")
        print("Install with: pip install datasets")
        return False


def prepare_iwslt14_mini(save_dir: str):
    """
    准备 IWSLT14 德语-英语数据集（较小，适合测试）
    数据量: ~170k 训练样本
    """
    try:
        from datasets import load_dataset
        
        print("Downloading IWSLT14 (De-En) from HuggingFace...")
        print("(This is a smaller dataset suitable for testing)")
        
        dataset = load_dataset('iwslt2014', 'de-en', cache_dir=save_dir)
        
        save_path = Path(save_dir) / 'iwslt14'
        save_path.mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                split_dir = save_path / split
                split_dir.mkdir(exist_ok=True)
                
                src_file = split_dir / f'{split}.de'
                tgt_file = split_dir / f'{split}.en'
                
                with open(src_file, 'w', encoding='utf-8') as f_src, \
                     open(tgt_file, 'w', encoding='utf-8') as f_tgt:
                    
                    for example in dataset[split]['translation']:
                        f_src.write(example['de'] + '\n')
                        f_tgt.write(example['en'] + '\n')
                
                print(f"  ✓ Saved {split}: {len(dataset[split])} examples")
        
        print("\n✓ IWSLT14 dataset ready!")
        return True
        
    except Exception as e:
        print(f"Failed to download IWSLT14: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download WMT14 dataset')
    parser.add_argument('--save-dir', type=str, default='data/wmt14',
                        help='Directory to save the dataset')
    parser.add_argument('--lang-pair', type=str, default='de-en',
                        choices=['de-en', 'en-de', 'fr-en', 'en-fr'],
                        help='Language pair')
    parser.add_argument('--source', type=str, default='huggingface',
                        choices=['statmt', 'huggingface', 'iwslt14'],
                        help='Data source')
    parser.add_argument('--test-only', action='store_true',
                        help='Download only test set for quick testing')
    args = parser.parse_args()
    
    print("=" * 60)
    print("WMT14 Dataset Download Script")
    print("=" * 60)
    print(f"Save directory: {args.save_dir}")
    print(f"Language pair: {args.lang_pair}")
    print(f"Source: {args.source}")
    print("=" * 60)
    print()
    
    if args.source == 'huggingface':
        success = download_wmt14_huggingface(args.save_dir, args.lang_pair)
        if not success:
            print("\nFalling back to IWSLT14 (smaller dataset)...")
            prepare_iwslt14_mini(args.save_dir)
    elif args.source == 'iwslt14':
        prepare_iwslt14_mini(args.save_dir)
    else:
        download_wmt14_statmt(args.save_dir)
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Learn BPE with: python data/tokenizer.py learn")
    print("  2. Encode data with: python data/tokenizer.py encode")
    print("  3. Start training with: python src/train.py --config configs/base_config.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()

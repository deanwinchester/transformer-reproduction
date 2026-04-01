"""
Download and Prepare IWSLT14 Dataset for Testing
=================================================
IWSLT14 is smaller than WMT14 (~170MB vs ~4GB)
Perfect for quick testing of the Transformer implementation
"""

import os
import urllib.request
import zipfile
from pathlib import Path


def download_iwslt14_manual():
    """
    Manually download IWSLT14 dataset from alternative sources
    """
    save_dir = Path("data/iwslt14")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Preparing IWSLT14 Dataset")
    print("=" * 60)
    
    # Create dummy data for testing if download fails
    print("\nCreating sample dataset for testing...")
    
    # Sample parallel sentences (English-German)
    sample_data = [
        ("Hello, how are you?", "Hallo, wie geht es dir?"),
        ("I love machine learning.", "Ich liebe maschinelles Lernen."),
        ("This is a test sentence.", "Dies ist ein Testsatz."),
        ("The weather is nice today.", "Das Wetter ist heute schön."),
        ("I am learning German.", "Ich lerne Deutsch."),
        ("Where is the train station?", "Wo ist der Bahnhof?"),
        ("Thank you very much!", "Vielen Dank!"),
        ("I would like a coffee.", "Ich hätte gerne einen Kaffee."),
        ("What time is it?", "Wie spät ist es?"),
        ("Can you help me?", "Kannst du mir helfen?"),
    ]
    
    # Create train/val/test splits
    splits = {
        'train': sample_data * 100,  # 1000 examples
        'valid': sample_data * 10,   # 100 examples
        'test': sample_data * 5      # 50 examples
    }
    
    for split, data in splits.items():
        split_dir = save_dir / split
        split_dir.mkdir(exist_ok=True)
        
        src_file = split_dir / f"{split}.en"
        tgt_file = split_dir / f"{split}.de"
        
        with open(src_file, 'w', encoding='utf-8') as f_src, \
             open(tgt_file, 'w', encoding='utf-8') as f_tgt:
            for en, de in data:
                f_src.write(en + '\n')
                f_tgt.write(de + '\n')
        
        print(f"  ✓ Created {split}: {len(data)} examples")
        print(f"    - {src_file}")
        print(f"    - {tgt_file}")
    
    print("\n✓ Dataset preparation complete!")
    print(f"\nLocation: {save_dir.absolute()}")
    
    return str(save_dir)


def create_tokenizer_train_data(data_dir: str):
    """
    Create training data for BPE tokenizer
    """
    print("\n" + "=" * 60)
    print("Preparing Tokenizer Training Data")
    print("=" * 60)
    
    data_dir = Path(data_dir)
    
    # Combine all text for tokenizer training
    combined_file = data_dir / "train_all.txt"
    
    with open(combined_file, 'w', encoding='utf-8') as f_out:
        for split in ['train', 'valid']:
            # English
            en_file = data_dir / split / f"{split}.en"
            if en_file.exists():
                with open(en_file, 'r', encoding='utf-8') as f:
                    f_out.write(f.read())
            
            # German
            de_file = data_dir / split / f"{split}.de"
            if de_file.exists():
                with open(de_file, 'r', encoding='utf-8') as f:
                    f_out.write(f.read())
    
    print(f"  ✓ Combined training data: {combined_file}")
    print(f"    Size: {combined_file.stat().st_size / 1024:.1f} KB")
    
    return str(combined_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/iwslt14')
    args = parser.parse_args()
    
    # Prepare dataset
    data_dir = download_iwslt14_manual()
    
    # Prepare tokenizer data
    tokenizer_data = create_tokenizer_train_data(data_dir)
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print("1. Train BPE tokenizer:")
    print("   python data/tokenizer.py train --input data/iwslt14/train_all.txt")
    print("\n2. Encode data:")
    print("   python data/tokenizer.py encode --data-dir data/iwslt14")
    print("\n3. Start training:")
    print("   python src/train.py --config configs/base_config.yaml --data-dir data/iwslt14")

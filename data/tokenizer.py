"""
Simple BPE Tokenizer
====================
A simplified BPE tokenizer for testing without requiring sentencepiece
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


class SimpleBPETokenizer:
    """
    简化的 BPE Tokenizer
    
    用于快速测试，不包含完整的 BPE 训练算法
    使用简单的字符级 + 常见子词分割
    """
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}
        
        # Special tokens
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
    def train(self, texts: List[str]):
        """
        训练 tokenizer（简化版）
        
        实际 BPE 算法会更复杂，这里使用简单的词频统计
        """
        print(f"Training tokenizer with vocab_size={self.vocab_size}")
        
        # Initialize vocab with special tokens
        self.vocab = {token: i for i, token in enumerate(self.special_tokens)}
        
        # Tokenize by whitespace and punctuation
        word_freqs = {}
        for text in texts:
            # Simple tokenization
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            for word in words:
                word_freqs[word] = word_freqs.get(word, 0) + 1
        
        # Add most frequent words to vocab
        sorted_words = sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)
        
        for word, freq in sorted_words:
            if len(self.vocab) >= self.vocab_size:
                break
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        # Build reverse vocab
        self.reverse_vocab = {i: token for token, i in self.vocab.items()}
        
        print(f"  Vocab size: {len(self.vocab)}")
        print(f"  Trained on {len(texts)} texts")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """编码文本"""
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.vocab[self.bos_token])
        
        # Simple tokenization
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Character-level fallback
                for char in word:
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    else:
                        tokens.append(self.vocab[self.unk_token])
        
        if add_special_tokens:
            tokens.append(self.vocab[self.eos_token])
        
        return tokens
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """解码 IDs"""
        tokens = []
        for idx in ids:
            token = self.reverse_vocab.get(idx, self.unk_token)
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        
        # Simple detokenization
        text = ' '.join(tokens)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def save(self, path: str):
        """保存 tokenizer"""
        data = {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {path}")
    
    def load(self, path: str):
        """加载 tokenizer"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.vocab_size = data['vocab_size']
        self.special_tokens = data['special_tokens']
        self.reverse_vocab = {i: token for token, i in self.vocab.items()}
        print(f"Tokenizer loaded from {path}")
    
    def __len__(self):
        return len(self.vocab)


def train_tokenizer(input_file: str, output_dir: str, vocab_size: int = 10000):
    """训练 tokenizer"""
    print(f"\nReading training data from {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(texts)} lines")
    
    # Train
    tokenizer = SimpleBPETokenizer(vocab_size=vocab_size)
    tokenizer.train(texts)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'tokenizer.json')
    tokenizer.save(save_path)
    
    # Test
    test_text = "Hello world!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\nTest:")
    print(f"  Input:  '{test_text}'")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: '{decoded}'")
    
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description='Tokenizer utilities')
    parser.add_argument('command', choices=['train', 'test'])
    parser.add_argument('--input', type=str, help='Input text file')
    parser.add_argument('--output-dir', type=str, default='data/iwslt14')
    parser.add_argument('--vocab-size', type=int, default=8000)
    args = parser.parse_args()
    
    if args.command == 'train':
        if not args.input:
            print("Error: --input required for training")
            return
        train_tokenizer(args.input, args.output_dir, args.vocab_size)
    
    elif args.command == 'test':
        tokenizer_path = os.path.join(args.output_dir, 'tokenizer.json')
        if not os.path.exists(tokenizer_path):
            print(f"Error: Tokenizer not found at {tokenizer_path}")
            return
        
        tokenizer = SimpleBPETokenizer()
        tokenizer.load(tokenizer_path)
        
        test_sentences = [
            "Hello, how are you?",
            "I love machine learning.",
            "This is a test."
        ]
        
        print("\nTokenizer Test:")
        for sent in test_sentences:
            encoded = tokenizer.encode(sent)
            decoded = tokenizer.decode(encoded)
            print(f"\n  Input:    '{sent}'")
            print(f"  Tokens:   {len(encoded)}")
            print(f"  IDs:      {encoded}")
            print(f"  Decoded:  '{decoded}'")


if __name__ == "__main__":
    main()

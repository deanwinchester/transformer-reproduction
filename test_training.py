"""
Quick Test Script for Transformer Training
============================================
This script tests the entire training pipeline with a small model
"""

import sys
sys.path.insert(0, '/Users/liuhonghao/Projects/open-agc/workspace/transformer-reproduction')

import torch
import torch.nn as nn
from pathlib import Path

print("=" * 60)
print("Transformer Training Test")
print("=" * 60)

# 1. Test imports
print("\n1. Testing imports...")
try:
    from src.model import build_transformer_base
    from src.embedding import TransformerEmbedding
    from src.attention import MultiHeadAttention
    from src.encoder import TransformerEncoder
    from src.decoder import TransformerDecoder
    from data.tokenizer import SimpleBPETokenizer
    from data.dataset import TranslationDataset, collate_fn
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# 2. Test tokenizer
print("\n2. Testing tokenizer...")
try:
    tokenizer = SimpleBPETokenizer()
    tokenizer.load('data/iwslt14/tokenizer.json')
    
    test_text = "Hello world!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"   Input:  '{test_text}'")
    print(f"   Encoded: {encoded}")
    print(f"   Decoded: '{decoded}'")
    print(f"   ✓ Tokenizer working (vocab size: {len(tokenizer)})")
except Exception as e:
    print(f"   ✗ Tokenizer failed: {e}")
    sys.exit(1)

# 3. Test dataset
print("\n3. Testing dataset...")
try:
    dataset = TranslationDataset(
        src_file='data/iwslt14/train/train.en',
        tgt_file='data/iwslt14/train/train.de',
        src_tokenizer=tokenizer,
        tgt_tokenizer=tokenizer,
        max_len=50
    )
    
    print(f"   Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"   Sample src: {sample['src']}")
    print(f"   Sample tgt: {sample['tgt']}")
    print(f"   ✓ Dataset working")
except Exception as e:
    print(f"   ✗ Dataset failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Test DataLoader
print("\n4. Testing DataLoader...")
try:
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    batch = next(iter(loader))
    print(f"   Batch src shape: {batch['src'].shape}")
    print(f"   Batch tgt shape: {batch['tgt'].shape}")
    print(f"   ✓ DataLoader working")
except Exception as e:
    print(f"   ✗ DataLoader failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Test model creation
print("\n5. Testing model creation...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    model = build_transformer_base(
        src_vocab_size=len(tokenizer),
        tgt_vocab_size=len(tokenizer),
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_encoder_layers=2,
        n_decoder_layers=2,
        dropout=0.1
    )
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {n_params:,}")
    print(f"   ✓ Model created")
except Exception as e:
    print(f"   ✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. Test forward pass
print("\n6. Testing forward pass...")
try:
    src = batch['src'].to(device)
    tgt_input = batch['tgt_input'].to(device)
    tgt_output = batch['tgt_output'].to(device)
    
    logits = model(src, tgt_input)
    
    print(f"   Input shape:  {src.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Expected:     [batch, tgt_len, vocab_size]")
    print(f"   ✓ Forward pass working")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. Test loss computation
print("\n7. Testing loss computation...")
try:
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    loss = criterion(
        logits.view(-1, logits.size(-1)),
        tgt_output.view(-1)
    )
    
    print(f"   Loss: {loss.item():.4f}")
    print(f"   ✓ Loss computation working")
except Exception as e:
    print(f"   ✗ Loss computation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 8. Test backward pass
print("\n8. Testing backward pass...")
try:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"   ✓ Backward pass working")
except Exception as e:
    print(f"   ✗ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 9. Test generation (greedy decode)
print("\n9. Testing generation...")
try:
    model.eval()
    with torch.no_grad():
        generated = model.greedy_decode(
            src[:1],  # Just first sample
            start_symbol=2,  # <s>
            end_symbol=3,    # </s>
            max_len=20
        )
    
    print(f"   Generated IDs: {generated[0].tolist()}")
    decoded = tokenizer.decode(generated[0].tolist())
    print(f"   Decoded: '{decoded}'")
    print(f"   ✓ Generation working")
except Exception as e:
    print(f"   ✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 10. Mini training loop
print("\n10. Testing mini training loop (3 steps)...")
try:
    model.train()
    losses = []
    
    for i, batch in enumerate(loader):
        if i >= 3:
            break
        
        src = batch['src'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        
        optimizer.zero_grad()
        logits = model(src, tgt_input)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.view(-1))
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"   Step {i+1}: Loss = {loss.item():.4f}")
    
    avg_loss = sum(losses) / len(losses)
    print(f"   Average loss: {avg_loss:.4f}")
    print(f"   ✓ Training loop working")
except Exception as e:
    print(f"   ✗ Training loop failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nThe Transformer implementation is working correctly!")
print("\nNext steps:")
print("  1. Train on full data: python src/train.py --config configs/test_config.yaml")
print("  2. Or run with more epochs for better results")
print("=" * 60)

#!/usr/bin/env python3
"""
Script to download and export SmolLM2-360M-Instruct weights to a binary format.
The binary format is compatible with the C++ inference engine.

Usage:
    python scripts/export_smollm2_weights.py [--output-dir weights/smollm2]
"""

import os
import sys
import struct
import argparse
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def download_model(model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"):
    """Download the SmolLM2 model from HuggingFace."""
    print(f"Downloading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for compatibility
        trust_remote_code=True
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer


def export_weights_binary(model, output_path: str):
    """
    Export model weights to binary format.
    
    Binary format:
        For each tensor:
        - name_len (int32): length of tensor name
        - name (bytes): tensor name
        - shape_len (int32): number of dimensions
        - shape (int32 * shape_len): dimension sizes
        - data (float32 * total_size): tensor data in row-major order
    """
    state_dict = model.state_dict()
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, "wb") as f:
        for name, param in state_dict.items():
            print(f"  Exporting: {name} {list(param.shape)}")
            
            # Write name
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("i", len(name_bytes)))
            f.write(name_bytes)
            
            # Write shape
            shape = list(param.shape)
            f.write(struct.pack("i", len(shape)))
            for dim in shape:
                f.write(struct.pack("i", dim))
            
            # Write data as float32
            data = param.detach().cpu().float().numpy()
            f.write(data.tobytes())
    
    print(f"\nExported {len(state_dict)} tensors to {output_path}")


def export_tokenizer(tokenizer, output_dir: str):
    """Export tokenizer files to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the tokenizer files
    tokenizer.save_pretrained(output_dir)
    
    # The save_pretrained will create vocab.json and merges.txt
    print(f"Exported tokenizer to {output_dir}")
    
    # List saved files
    for file in os.listdir(output_dir):
        filepath = os.path.join(output_dir, file)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            print(f"  {file}: {size:,} bytes")


def print_model_info(model):
    """Print model architecture information."""
    config = model.config
    
    print("\n" + "=" * 60)
    print("SmolLM2 Model Configuration")
    print("=" * 60)
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print(f"  head_dim: {config.hidden_size // config.num_attention_heads}")
    print(f"  max_position_embeddings: {config.max_position_embeddings}")
    print(f"  rms_norm_eps: {config.rms_norm_eps}")
    print(f"  rope_theta: {config.rope_theta}")
    print(f"  tie_word_embeddings: {config.tie_word_embeddings}")
    print("=" * 60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    
    # Print layer weight shapes
    print("\nWeight tensor shapes:")
    for name, param in model.named_parameters():
        print(f"  {name}: {list(param.shape)}")


def verify_weights(model, weights_path: str):
    """Verify that exported weights can be loaded correctly."""
    print("\nVerifying exported weights...")
    
    state_dict = model.state_dict()
    
    with open(weights_path, "rb") as f:
        tensor_count = 0
        while True:
            # Read name length
            name_len_bytes = f.read(4)
            if not name_len_bytes:
                break
            
            name_len = struct.unpack("i", name_len_bytes)[0]
            name = f.read(name_len).decode("utf-8")
            
            # Read shape
            shape_len = struct.unpack("i", f.read(4))[0]
            shape = [struct.unpack("i", f.read(4))[0] for _ in range(shape_len)]
            
            # Read data
            total_size = 1
            for dim in shape:
                total_size *= dim
            data = f.read(total_size * 4)  # 4 bytes per float32
            
            # Verify against original
            if name in state_dict:
                original = state_dict[name]
                if list(original.shape) == shape:
                    # Quick check: compare first and last few values
                    loaded = torch.frombuffer(bytearray(data), dtype=torch.float32)
                    original_flat = original.detach().cpu().float().flatten()
                    
                    if torch.allclose(loaded[:10], original_flat[:10], atol=1e-6) and \
                       torch.allclose(loaded[-10:], original_flat[-10:], atol=1e-6):
                        tensor_count += 1
                    else:
                        print(f"  WARNING: Data mismatch for {name}")
                else:
                    print(f"  WARNING: Shape mismatch for {name}: {shape} vs {list(original.shape)}")
            else:
                print(f"  WARNING: Tensor {name} not found in original model")
    
    print(f"Verified {tensor_count} tensors successfully")
    return tensor_count == len(state_dict)


def main():
    parser = argparse.ArgumentParser(description="Export SmolLM2 weights to binary format")
    parser.add_argument(
        "--model-name",
        type=str,
        default="HuggingFaceTB/SmolLM2-360M-Instruct",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="weights/smollm2",
        help="Output directory for weights and tokenizer"
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Only export weights, skip tokenizer"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify exported weights"
    )
    parser.add_argument(
        "--print-info",
        action="store_true",
        default=True,
        help="Print model architecture information"
    )
    
    args = parser.parse_args()
    
    # Download model
    print("=" * 60)
    print("SmolLM2 Weight Exporter")
    print("=" * 60)
    
    model, tokenizer = download_model(args.model_name)
    
    if args.print_info:
        print_model_info(model)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Export weights
    weights_path = os.path.join(args.output_dir, "smollm2_weights.bin")
    print(f"\nExporting weights to {weights_path}...")
    export_weights_binary(model, weights_path)
    
    # Export tokenizer
    if not args.weights_only:
        print(f"\nExporting tokenizer to {args.output_dir}...")
        export_tokenizer(tokenizer, args.output_dir)
    
    # Verify
    if args.verify:
        success = verify_weights(model, weights_path)
        if success:
            print("\n✓ All weights exported and verified successfully!")
        else:
            print("\n✗ Weight verification failed!")
            sys.exit(1)
    
    # Print final summary
    weights_size = os.path.getsize(weights_path)
    print(f"\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print(f"  Weights file: {weights_path}")
    print(f"  Weights size: {weights_size:,} bytes ({weights_size / 1e9:.2f} GB)")
    print(f"  Tokenizer dir: {args.output_dir}")
    print("\nTo use in C++:")
    print(f"  ModelLoader loader(\"{weights_path}\");")
    print(f"  SmolLM2Tokenizer tokenizer;")
    print(f"  tokenizer.load(\"{args.output_dir}\");")
    print("=" * 60)


if __name__ == "__main__":
    main()

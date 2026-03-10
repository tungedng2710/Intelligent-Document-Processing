#!/usr/bin/env python3
"""
Merge Qwen3-VL base model with LoRA adapter checkpoint for vLLM deployment.

This script:
1. Loads the base Qwen3-VL model
2. Loads the LoRA adapter weights from checkpoint
3. Merges them into a single model
4. Saves the merged model in vLLM-compatible format
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from unsloth import FastVisionModel


def merge_and_save_model(
    base_model_name: str,
    lora_checkpoint_path: str,
    output_dir: str,
    max_seq_length: int = 2048,
    dtype: torch.dtype = None,
    load_in_4bit: bool = False,
):
    """
    Merge base model with LoRA adapter and save for vLLM.
    
    Args:
        base_model_name: HuggingFace model name or path (e.g., "Qwen/Qwen3-VL-8B-Instruct")
        lora_checkpoint_path: Path to LoRA checkpoint directory
        output_dir: Directory to save merged model
        max_seq_length: Maximum sequence length
        dtype: Data type for model weights (None for auto)
        load_in_4bit: Whether to load in 4-bit (should be False for merging)
    """
    
    print(f"=" * 80)
    print(f"Starting model merge process")
    print(f"=" * 80)
    print(f"Base model: {base_model_name}")
    print(f"LoRA checkpoint: {lora_checkpoint_path}")
    print(f"Output directory: {output_dir}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"Data type: {dtype}")
    print(f"=" * 80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load the base model with LoRA adapter
    print("\n[1/3] Loading base model with LoRA adapter...")
    try:
        model, tokenizer = FastVisionModel.from_pretrained(
            base_model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        print("✓ Base model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load base model: {e}")
        raise
    
    # Step 2: Load LoRA weights
    print("\n[2/3] Loading LoRA adapter weights...")
    try:
        # Load PEFT model with adapter
        from peft import PeftModel
        
        model = PeftModel.from_pretrained(
            model,
            lora_checkpoint_path,
            is_trainable=False,
        )
        print("✓ LoRA adapter loaded successfully")
        
        # Print adapter info
        print(f"\nAdapter configuration:")
        print(f"  - LoRA rank (r): {model.peft_config['default'].r}")
        print(f"  - LoRA alpha: {model.peft_config['default'].lora_alpha}")
        print(f"  - LoRA dropout: {model.peft_config['default'].lora_dropout}")
        print(f"  - Target modules: {model.peft_config['default'].target_modules}")
        
    except Exception as e:
        print(f"✗ Failed to load LoRA adapter: {e}")
        raise
    
    # Step 3: Merge LoRA weights into base model
    print("\n[3/3] Merging LoRA weights into base model...")
    try:
        # Merge and unload - this combines LoRA weights with base weights
        model = model.merge_and_unload()
        print("✓ LoRA weights merged successfully")
        
        # Save the merged model
        print(f"\nSaving merged model to: {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print("✓ Merged model saved successfully")
        
    except Exception as e:
        print(f"✗ Failed to merge and save model: {e}")
        raise
    
    print(f"\n" + "=" * 80)
    print(f"SUCCESS! Merged model saved to: {output_dir}")
    print(f"=" * 80)
    print(f"\nYou can now use this model with vLLM:")
    print(f"  vllm serve {output_dir} --trust-remote-code")
    print(f"\nOr in Python:")
    print(f"  from vllm import LLM")
    print(f"  llm = LLM(model='{output_dir}', trust_remote_code=True)")
    print(f"=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Merge Qwen3-VL base model with LoRA checkpoint for vLLM deployment"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Base model name or path (default: Qwen/Qwen3-VL-8B-Instruct)"
    )
    
    parser.add_argument(
        "--lora-checkpoint",
        type=str,
        default="outputs/checkpoint-2000",
        help="Path to LoRA checkpoint directory (default: outputs/checkpoint-2000)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="merged_model",
        help="Output directory for merged model (default: merged_model)"
    )
    
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    # Validate paths
    lora_checkpoint = Path(args.lora_checkpoint)
    if not lora_checkpoint.exists():
        print(f"Error: LoRA checkpoint not found at: {args.lora_checkpoint}")
        print(f"Please check the path and try again.")
        sys.exit(1)
    
    # Check for required files in checkpoint
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing_files = [f for f in required_files if not (lora_checkpoint / f).exists()]
    if missing_files:
        print(f"Error: Checkpoint directory is missing required files: {missing_files}")
        print(f"This doesn't appear to be a valid LoRA checkpoint directory.")
        sys.exit(1)
    
    # Run the merge
    try:
        merge_and_save_model(
            base_model_name=args.base_model,
            lora_checkpoint_path=args.lora_checkpoint,
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            dtype=dtype,
            load_in_4bit=False,  # Must be False for merging
        )
    except Exception as e:
        print(f"\n✗ Merge failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

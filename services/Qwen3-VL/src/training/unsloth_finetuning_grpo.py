"""
GRPO-based Fine-tuning for Qwen3-VL on Bill of Lading Document Extraction
==========================================================================
"""

from unsloth import FastVisionModel
from trl import GRPOConfig, GRPOTrainer
import torch
import re
import math
import random
import json
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
from datetime import datetime

# Logging and visualization
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('grpo_training.log')
    ]
)
logger = logging.getLogger(__name__)

# Import sophisticated reward functions
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.reward_functions import (
    block_classification_reward_v2,
    table_accuracy_reward,
    content_similarity_reward,
    format_compliance_reward_v2,
    table_first_matching_reward,
    table_structure_reward,  # TEDS-based table structure reward
    parse_blocks,
    get_blocks_by_type,
    hungarian_match,
    combined_similarity,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Model settings - Use SFT checkpoint as base for Stage 2
    "model_name": "outputs/checkpoint-300",  # Load from SFT Stage 1 checkpoint
    "base_model_name": "Qwen/Qwen3-VL-8B-Instruct",  # Original base model (for reference)
    "load_in_4bit": False,  # Match SFT settings
    "max_seq_length": 8192,  # Match SFT max_length
    
    # LoRA settings - Match SFT
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    
    # Dataset paths - Same as SFT
    "synthetic_image_root": "/home/app/idp/data/synthetic-bol/images",
    "synthetic_text_root": "/home/app/idp/data/synthetic-bol/markdown",
    "real_image_root": "/home/app/idp/data/real-bol/bol_sribd_images",
    "real_text_root": "/home/app/idp/data/real-bol/bol_markdown",
    
    # GRPO Training settings
    "batch_size": 6,  # GRPO typically uses small batch sizes
    "gradient_accumulation_steps": 4,
    "num_generations": 5,  # Number of samples per prompt for GRPO
    "learning_rate": 5e-6,  # Lower LR for GRPO refinement
    "num_train_epochs": 1,
    "max_prompt_length": 2048,
    "max_completion_length": 6144,
    "output_dir": "outputs_grpo",
    
    # Logging and checkpointing
    "save_steps": 10,  # Save checkpoint every 10 steps
    "save_total_limit": 10,  # Keep last 10 checkpoints
    "logging_steps": 1,  # Log every step
    "logging_dir": "outputs_grpo/logs",  # TensorBoard logs directory
    
    # Dataset settings - Match SFT
    "anchor_ratio": 0.7,  # Ratio of real samples
    
    # Seed
    "seed": 3407,
}

# =============================================================================
# CUSTOM CALLBACK FOR RL METRICS VISUALIZATION
# =============================================================================

class GRPOMetricsCallback(TrainerCallback):
    """
    Custom callback to track and visualize detailed RL training metrics.
    Logs to TensorBoard and saves metrics to JSON for later analysis.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics_history = []
        self.reward_history = {
            'block_classification': [],
            'table_accuracy': [],
            'content_similarity': [],
            'format_compliance': [],
            'total_reward': [],
        }
        self.step_times = []
        self.start_time = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)
        
        # Initialize custom TensorBoard writer for detailed metrics
        self.tb_writer = SummaryWriter(log_dir=f"{output_dir}/logs/grpo_detailed")
        logger.info(f"GRPO Metrics Callback initialized. Logs at: {output_dir}/logs")
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        logger.info(f"Training started at {self.start_time}")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        step = state.global_step
        
        # Extract and log RL-specific metrics
        metrics_to_track = {
            # Reward metrics
            'reward': logs.get('reward', None),
            'reward_std': logs.get('reward_std', None),
            'rewards/block_classification_reward_v2': logs.get('rewards/block_classification_reward_v2', None),
            'rewards/table_accuracy_reward': logs.get('rewards/table_accuracy_reward', None),
            'rewards/content_similarity_reward': logs.get('rewards/content_similarity_reward', None),
            'rewards/format_compliance_reward_v2': logs.get('rewards/format_compliance_reward_v2', None),
            
            # Policy metrics
            'loss': logs.get('loss', None),
            'policy_loss': logs.get('policy_loss', None),
            'kl': logs.get('kl', None),
            'kl_coef': logs.get('kl_coef', None),
            
            # Generation metrics
            'completion_length': logs.get('completion_length', None),
            'prompt_length': logs.get('prompt_length', None),
            
            # Learning metrics
            'learning_rate': logs.get('learning_rate', None),
            'grad_norm': logs.get('grad_norm', None),
        }
        
        # Log to TensorBoard with proper grouping
        for key, value in metrics_to_track.items():
            if value is not None:
                # Create proper grouping for TensorBoard
                if 'reward' in key.lower():
                    self.tb_writer.add_scalar(f"Rewards/{key}", value, step)
                elif 'loss' in key.lower():
                    self.tb_writer.add_scalar(f"Loss/{key}", value, step)
                elif 'kl' in key.lower():
                    self.tb_writer.add_scalar(f"KL/{key}", value, step)
                elif 'length' in key.lower():
                    self.tb_writer.add_scalar(f"Generation/{key}", value, step)
                else:
                    self.tb_writer.add_scalar(f"Training/{key}", value, step)
        
        # Store metrics history
        self.metrics_history.append({
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **{k: v for k, v in metrics_to_track.items() if v is not None}
        })
        
        # Log summary every 10 steps
        if step % 10 == 0:
            reward = logs.get('reward', 0)
            kl = logs.get('kl', 0)
            loss = logs.get('loss', 0)
            logger.info(f"Step {step}: reward={reward:.4f}, kl={kl:.4f}, loss={loss:.4f}")
    
    def on_save(self, args, state, control, **kwargs):
        # Save metrics history to JSON when checkpoint is saved
        metrics_file = f"{self.output_dir}/metrics_step_{state.global_step}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")
    
    def on_train_end(self, args, state, control, **kwargs):
        # Save final metrics
        final_metrics_file = f"{self.output_dir}/metrics_final.json"
        with open(final_metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Close TensorBoard writer
        self.tb_writer.close()
        
        # Log training summary
        end_time = datetime.now()
        duration = end_time - self.start_time
        logger.info(f"Training completed at {end_time}")
        logger.info(f"Total training time: {duration}")
        logger.info(f"Final metrics saved to {final_metrics_file}")
        
        # Print TensorBoard command
        print("\n" + "=" * 70)
        print("To visualize training metrics, run:")
        print(f"  tensorboard --logdir={self.output_dir}/logs")
        print("=" * 70 + "\n")


# =============================================================================
# SPECIAL TOKENS FOR BLOCK TAGS (Must match SFT Stage 1)
# =============================================================================

# SPECIAL_TOKENS = [
#     "<text_block>",
#     "</text_block>",
#     "<table_block>",
#     "</table_block>"
# ]

# =============================================================================
# INSTRUCTION (Must match SFT Stage 1 exactly)
# =============================================================================

INSTRUCTION = """
**Task:** Perform document layout analysis and content extraction. Extract all text from the image and organize it into logical blocks (based on both visual cues, border lines, white space and semantic meanings).
Classify each block whether it is table or not before output its content in markdown or html.

**Format:**
- Wrap each block: `<text_block>content</text_block>` or `<table_block>content</table_block>`
- Headers: If block has a header, start with `##` else leave empty.
- Tables: HTML `<table>` format
- Checkboxes: `[ ]` empty, `[x]` checked
- Special symbols: HTML entities (&copy; &reg; &diam; &cir; &deg; &amp; etc.)
- Transcribe text exactly as shown.

**Output:**
"""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def transform_markdown_to_tagged_blocks(text: str) -> str:
    """
    Transform raw markdown text with '---' separators into tagged block format.
    Must match the SFT Stage 1 format exactly.
    
    Args:
        text: Raw markdown text from annotation file
        
    Returns:
        Transformed text with <text_block> and <table_block> tags
    """
    blocks = text.split('---')
    result = []
    
    for block in blocks:
        stripped_block = block.strip()
        if not stripped_block:
            continue
        
        # Determine block type based on content
        if '<table>' in stripped_block.lower():
            result.append(f'<table_block>\n{stripped_block}\n</table_block>')
        else:
            result.append(f'<text_block>\n{stripped_block}\n</text_block>')
    
    return '\n'.join(result)  # No block_separator, match SFT format

# =============================================================================
# DATASET CLASSES
# =============================================================================

class GRPOVisionDataset(Dataset):
    """
    Dataset for GRPO training that returns prompts with images and reference answers.
    """
    
    def __init__(
        self,
        image_root: str,
        text_root: str,
        instruction: str,
        image_ext: str = '.png',
        transform_to_tags: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.instruction = instruction
        self.transform_to_tags = transform_to_tags
        self.samples = []
        
        image_path = Path(image_root)
        text_path = Path(text_root)
        
        text_files = list(text_path.glob('**/*.md'))
        
        for text_file in text_files:
            relative_path = text_file.relative_to(text_path)
            image_file = image_path / relative_path.with_suffix(image_ext)
            
            if image_file.exists():
                self.samples.append({
                    'image_path': str(image_file),
                    'text_path': str(text_file)
                })
        
        # Limit samples if specified
        if max_samples and len(self.samples) > max_samples:
            random.seed(CONFIG["seed"])
            self.samples = random.sample(self.samples, max_samples)
        
        print(f"Initialized GRPO dataset with {len(self.samples)} samples from {image_root}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(sample['image_path']).convert('RGB')
            
            # Resize to reduce context length (important for GRPO)
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.LANCZOS)
            
            # Load and transform text
            with open(sample['text_path'], 'r', encoding='utf-8') as f:
                text = f.read()
            
            if self.transform_to_tags:
                answer = transform_markdown_to_tagged_blocks(text)
            else:
                answer = text
            
            # Create prompt in the format expected by GRPO
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},  # Placeholder - will be filled by trainer
                        {"type": "text", "text": self.instruction},
                    ],
                },
            ]
            
            return {
                "prompt": prompt,
                "image": image,
                "answer": answer,  # Reference answer for reward calculation
            }
        
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return None


class BalancedGRPODataset(Dataset):
    """
    Combines real and synthetic datasets with specified ratio for GRPO.
    For GRPO, we prioritize real samples as they are more valuable for learning.
    """
    
    def __init__(
        self,
        real_dataset: GRPOVisionDataset,
        synthetic_dataset: GRPOVisionDataset,
        real_ratio: float = 0.5,  # Higher ratio of real samples for GRPO
        total_samples: Optional[int] = None,
        seed: int = 3407,
    ):
        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset
        self.seed = seed
        
        random.seed(seed)
        
        if total_samples is None:
            # Use all real samples, match with synthetic
            total_samples = int(len(real_dataset) / real_ratio)
        
        num_real = int(total_samples * real_ratio)
        num_synthetic = total_samples - num_real
        
        # Sample indices
        real_indices = list(range(len(real_dataset)))
        synthetic_indices = list(range(len(synthetic_dataset)))
        
        # Oversample real if needed
        while len(real_indices) < num_real:
            real_indices.extend(list(range(len(real_dataset))))
        
        random.shuffle(real_indices)
        random.shuffle(synthetic_indices)
        
        self.indices = []
        self.indices.extend([('real', i) for i in real_indices[:num_real]])
        self.indices.extend([('synthetic', i) for i in synthetic_indices[:num_synthetic]])
        
        random.shuffle(self.indices)
        
        print(f"Balanced GRPO Dataset: {num_real} real + {num_synthetic} synthetic = {len(self.indices)} total")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        source, actual_idx = self.indices[idx]
        
        if source == 'real':
            return self.real_dataset[actual_idx]
        else:
            return self.synthetic_dataset[actual_idx]


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    print("=" * 70)
    print("GRPO Fine-tuning Stage 2 (from SFT checkpoint)")
    print("=" * 70)
    
    # =========================================================================
    # 1. Load Model from SFT Checkpoint
    # =========================================================================
    print("\n[1/5] Loading model from SFT checkpoint...")
    print(f"       Checkpoint: {CONFIG['model_name']}")
    
    # Load the model from SFT Stage 1 checkpoint
    # The checkpoint already has the special tokens and LoRA weights
    model, processor = FastVisionModel.from_pretrained(
        CONFIG["model_name"],
        load_in_4bit=CONFIG["load_in_4bit"],
        max_seq_length=CONFIG["max_seq_length"],
        use_gradient_checkpointing="unsloth",
    )
    
    # =========================================================================
    # 2. Verify Special Tokens (should already be in checkpoint)
    # =========================================================================
    print("\n[2/5] Verifying special tokens from SFT checkpoint...")
    
    # For Vision models, we get a processor. Access the underlying tokenizer.
    # tokenizer = processor.tokenizer
    
    # # Check if special tokens already exist from SFT checkpoint
    # existing_special = tokenizer.additional_special_tokens if hasattr(tokenizer, 'additional_special_tokens') else []
    # tokens_already_exist = all(token in existing_special for token in SPECIAL_TOKENS)
    
    # if tokens_already_exist:
    #     print(f"Special tokens already present from SFT checkpoint: {SPECIAL_TOKENS}")
    # else:
    #     # Add special tokens if not present (fallback)
    #     print("Adding special tokens (not found in checkpoint)...")
    #     num_added = tokenizer.add_special_tokens({
    #         "additional_special_tokens": SPECIAL_TOKENS
    #     })
    #     print(f"Added {num_added} special tokens: {SPECIAL_TOKENS}")
    #     model.resize_token_embeddings(len(tokenizer))
    
    # # Verify tokens
    # for token in SPECIAL_TOKENS:
    #     token_id = tokenizer.convert_tokens_to_ids(token)
    #     print(f"  {token} -> token_id: {token_id}")
    
    # =========================================================================
    # 3. Continue with existing LoRA or add new adapters
    # =========================================================================
    print("\n[3/5] Checking LoRA adapters...")
    
    # Check if model already has LoRA adapters from SFT
    from peft import PeftModel
    if isinstance(model, PeftModel) or hasattr(model, 'peft_config'):
        print("LoRA adapters already present from SFT checkpoint")
        print("Continuing training with existing adapters...")
    else:
        print("Adding new LoRA adapters...")
        # IMPORTANT: Include modules_to_save for embedding layers with new tokens
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,  # Must be False for GRPO (vLLM limitation)
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=CONFIG["r"],
            lora_alpha=CONFIG["lora_alpha"],
            lora_dropout=CONFIG["lora_dropout"],
            bias="none",
            random_state=CONFIG["seed"],
            modules_to_save=["embed_tokens", "lm_head"],  # Save new token embeddings!
        )
    
    # =========================================================================
    # 4. Create Datasets
    # =========================================================================
    print("\n[4/5] Creating datasets...")
    
    # Real dataset (high value)
    real_dataset = GRPOVisionDataset(
        image_root=CONFIG["real_image_root"],
        text_root=CONFIG["real_text_root"],
        instruction=INSTRUCTION,
        image_ext='.png',
        transform_to_tags=True,
    )
    
    # Synthetic dataset
    synthetic_dataset = GRPOVisionDataset(
        image_root=CONFIG["synthetic_image_root"],
        text_root=CONFIG["synthetic_text_root"],
        instruction=INSTRUCTION,
        image_ext='.png',
        transform_to_tags=True,
        max_samples=500,  # Limit synthetic for GRPO (quality over quantity)
    )
    
    # Combined dataset - use anchor_ratio from config to match SFT distribution
    # For GRPO, we use real_ratio which is (1 - anchor_ratio) since anchor = real
    train_dataset = BalancedGRPODataset(
        real_dataset=real_dataset,
        synthetic_dataset=synthetic_dataset,
        real_ratio=CONFIG["anchor_ratio"],  # Match SFT: 30% real samples
        total_samples=2000,  # Start with smaller dataset for GRPO
        seed=CONFIG["seed"],
    )
    
    # =========================================================================
    # 5. Configure and Run GRPO Training
    # =========================================================================
    print("\n[5/5] Starting GRPO training...")
    
    training_args = GRPOConfig(
        # Optimization
        learning_rate=CONFIG["learning_rate"],
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.02,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        max_grad_norm=0.1,
        temperature=0.9,
        top_p=0.9,
        # Batch settings
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        
        # GRPO specific
        num_generations=CONFIG["num_generations"],
        max_prompt_length=CONFIG["max_prompt_length"],
        max_completion_length=CONFIG["max_completion_length"],
        
        # Training duration
        num_train_epochs=CONFIG["num_train_epochs"],
        
        # Logging and saving - Enhanced for RL metrics visualization
        logging_steps=CONFIG["logging_steps"],
        logging_dir=CONFIG["logging_dir"],
        save_steps=CONFIG["save_steps"],
        save_total_limit=CONFIG["save_total_limit"],
        output_dir=CONFIG["output_dir"],
        report_to="tensorboard",  # Enable TensorBoard logging
        
        # Advanced GRPO settings
        loss_type="dr_grpo",  # Dr. GRPO recommended for stability
        log_completions=True,  # Log generated completions for debugging
        
        # Seed
        seed=CONFIG["seed"],
    )
    
    # Create trainer with sophisticated reward functions using Table-First Matching
    # These reward functions from reward_functions.py use:
    # - Hungarian algorithm for optimal block matching
    # - Multiple similarity metrics (Jaccard, NED, token overlap)
    # - Table-First strategy focusing on minority class
    # - Asymmetric penalties for different error types
    
    # Initialize custom metrics callback for detailed RL visualization
    metrics_callback = GRPOMetricsCallback(output_dir=CONFIG["output_dir"])
    
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=processor,
        train_dataset=train_dataset,
        reward_funcs=[
            # Primary: Comprehensive block classification with Table-First Matching
            block_classification_reward_v2,
            # Secondary: Focused table classification accuracy (precision/recall)
            table_accuracy_reward,
            # Tertiary: Table structure using TEDS metric
            table_structure_reward,
            # Quaternary: Overall content similarity
            content_similarity_reward,
            # Quinary: Format compliance
            format_compliance_reward_v2,
        ],
        callbacks=[metrics_callback],  # Add custom callback for RL metrics
    )
    
    print("\nTraining configuration:")
    print(f"  - Learning rate: {CONFIG['learning_rate']}")
    print(f"  - Batch size: {CONFIG['batch_size']}")
    print(f"  - Gradient accumulation: {CONFIG['gradient_accumulation_steps']}")
    print(f"  - Num generations: {CONFIG['num_generations']}")
    print(f"  - Epochs: {CONFIG['num_train_epochs']}")
    print(f"  - Output dir: {CONFIG['output_dir']}")
    
    print("\nReward functions (using Table-First Matching with Hungarian Algorithm):")
    print("  1. block_classification_reward_v2 - Comprehensive matching with asymmetric penalties")
    print("  2. table_accuracy_reward - Focused on table precision/recall (F1.5 weighted)")
    print("  3. table_structure_reward - TEDS-based table structure matching")
    print("  4. content_similarity_reward - Overall content quality")
    print("  5. format_compliance_reward_v2 - Tag structure validation")
    
    print("\n" + "=" * 70)
    print("Starting training... (expect 0 reward for first ~50-100 steps)")
    print("=" * 70 + "\n")
    
    # Train!
    trainer_stats = trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    model.save_pretrained(f"{CONFIG['output_dir']}/final_model")
    processor.save_pretrained(f"{CONFIG['output_dir']}/final_model")
    
    print("\n" + "=" * 70)
    print("GRPO Stage 2 Training complete!")
    print(f"Model saved to: {CONFIG['output_dir']}/final_model")
    print("=" * 70)
    
    return trainer_stats


# =============================================================================
# INSTRUCTIONS FOR TWO-STAGE TRAINING
# =============================================================================

def show_instructions():
    """
    Show instructions for the two-stage training pipeline.
    """
    print("=" * 70)
    print("Two-Stage Training Pipeline: SFT → GRPO")
    print("=" * 70)
    
    print("""
[Stage 1] SFT Training (teaches format):
-----------------------------------------
  Command: python unsloth_finetuning_sft_tagged.py
  
  This creates a checkpoint at: outputs_sft_tagged/final_model
  The model learns:
    - Tagged block format (<text_block>, <table_block>)
    - Document structure extraction
    - Basic block classification

[Stage 2] GRPO Training (refines classification):
-------------------------------------------------
  Command: python unsloth_finetuning_grpo.py
  
  This script automatically loads: outputs_sft_tagged/final_model
  The model is refined with:
    - Table-First Matching rewards
    - Asymmetric penalties (Table→Text worse than Text→Table)
    - Hungarian algorithm for optimal block matching

[Evaluation]:
-------------
  Command: python infer_tagged.py --checkpoint outputs_grpo/final_model \\
           --folder /path/to/images --reference-folder /path/to/markdown

Configuration:
  - Both scripts use matching INSTRUCTION and SPECIAL_TOKENS
  - SFT: outputs_sft_tagged/, GRPO: outputs_grpo/
  - Modify CONFIG in each script to adjust paths and hyperparameters
""")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_instructions()
    else:
        main()

"""
GRPO-based Fine-tuning for Qwen3-VL on Bank Reports Document Extraction
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
        logging.FileHandler('grpo_training_bank_reports.log')
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
    
    # Dataset paths - Bank Reports
    "image_root": "/root/tungn197/idp/data/r3_bank_reports/train_data_v1/images",
    "text_root": "/root/tungn197/idp/data/r3_bank_reports/train_data_v1/markdowns",
    "instruction_prompt_file": "/root/tungn197/idp/llm/prompts/bank_report_ver_1.0.txt",
    
    # GRPO Training settings
    "batch_size": 2,  # Smaller batch for bank reports (less data)
    "gradient_accumulation_steps": 4,
    "num_generations": 5,  # Number of samples per prompt for GRPO
    "learning_rate": 5e-6,  # Lower LR for GRPO refinement
    "num_train_epochs": 3,  # More epochs for smaller dataset
    "max_prompt_length": 2048,
    "max_completion_length": 6144,
    "output_dir": "outputs_grpo_bank_reports",
    
    # Logging and checkpointing
    "save_steps": 5,  # Save checkpoint every 5 steps (smaller dataset)
    "save_total_limit": 10,  # Keep last 10 checkpoints
    "logging_steps": 1,  # Log every step
    "logging_dir": "outputs_grpo_bank_reports/logs",  # TensorBoard logs directory
    
    # Seed
    "seed": 3407,
}

# =============================================================================
# LOAD INSTRUCTION PROMPT
# =============================================================================

def load_instruction_prompt(file_path: str) -> str:
    """Load instruction prompt from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

INSTRUCTION = load_instruction_prompt(CONFIG["instruction_prompt_file"])

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
# HELPER FUNCTIONS
# =============================================================================

def json_to_tagged_blocks(json_data: dict) -> str:
    """
    Convert JSON ground truth to tagged block format.
    
    Args:
        json_data: Dictionary loaded from JSON file
        
    Returns:
        Text formatted with <text_block> and <table_block> tags
    """
    result = []
    
    def process_value(key, value, level=0):
        """Recursively process JSON values."""
        blocks = []
        
        if isinstance(value, dict):
            # Create a text block for nested objects
            content = []
            if level == 0:
                content.append(f"## {key}")
            else:
                content.append(f"**{key}:**")
            
            for k, v in value.items():
                sub_blocks = process_value(k, v, level + 1)
                blocks.extend(sub_blocks)
            
            if content:
                blocks.insert(0, ('<text_block>\n' + '\n'.join(content) + '\n</text_block>'))
                
        elif isinstance(value, list):
            # Check if it's a table (list of dicts with consistent keys)
            if value and all(isinstance(item, dict) for item in value):
                # This is a table
                headers = list(value[0].keys())
                table_html = ['<table>', '<tr>']
                
                # Add headers
                for header in headers:
                    table_html.append(f'  <th>{header}</th>')
                table_html.append('</tr>')
                
                # Add rows
                for row in value:
                    table_html.append('<tr>')
                    for header in headers:
                        cell_value = row.get(header, '')
                        table_html.append(f'  <td>{cell_value}</td>')
                    table_html.append('</tr>')
                
                table_html.append('</table>')
                
                table_content = '\n'.join(table_html)
                blocks.append(f'<table_block>\n## {key}\n{table_content}\n</table_block>')
            else:
                # List of primitives - treat as text
                content = [f"**{key}:**"]
                for item in value:
                    content.append(f"- {item}")
                blocks.append('<text_block>\n' + '\n'.join(content) + '\n</text_block>')
                
        else:
            # Primitive value
            if level == 0:
                blocks.append(f'<text_block>\n**{key}:** {value}\n</text_block>')
            else:
                blocks.append(f'<text_block>\n**{key}:** {value}\n</text_block>')
        
        return blocks
    
    # Process top-level keys
    for key, value in json_data.items():
        blocks = process_value(key, value)
        result.extend(blocks)
    
    return '\n'.join(result)


# =============================================================================
# DATASET CLASS
# =============================================================================

class GRPOVisionDatasetJSON(Dataset):
    """
    Dataset for GRPO training that loads JSON ground truth files.
    Handles mixed image extensions (.png and .jpeg).
    """
    
    def __init__(
        self,
        image_root: str,
        text_root: str,
        instruction: str,
        max_samples: Optional[int] = None,
    ):
        self.instruction = instruction
        self.samples = []
        
        image_path = Path(image_root)
        text_path = Path(text_root)
        
        # Find all JSON files
        json_files = list(text_path.glob('**/*.json'))
        
        for json_file in json_files:
            # Try different image extensions
            base_name = json_file.stem
            image_file = None
            
            for ext in ['.png', '.jpeg', '.jpg']:
                candidate = image_path / (base_name + ext)
                if candidate.exists():
                    image_file = candidate
                    break
            
            if image_file:
                self.samples.append({
                    'image_path': str(image_file),
                    'text_path': str(json_file)
                })
            else:
                logger.warning(f"No matching image found for {json_file}")
        
        # Limit samples if specified
        if max_samples and len(self.samples) > max_samples:
            random.seed(CONFIG["seed"])
            self.samples = random.sample(self.samples, max_samples)
        
        logger.info(f"Initialized GRPO dataset with {len(self.samples)} samples from {image_root}")
    
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
            
            # Load and transform JSON to tagged blocks
            with open(sample['text_path'], 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            answer = json_to_tagged_blocks(json_data)
            
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
            logger.error(f"Error loading sample {idx} ({sample['image_path']}): {e}")
            return None


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    print("=" * 70)
    print("GRPO Fine-tuning for Bank Reports - Qwen3-VL")
    print("=" * 70)
    
    print("\nConfiguration:")
    print(f"  - Image root: {CONFIG['image_root']}")
    print(f"  - Text root: {CONFIG['text_root']}")
    print(f"  - Instruction file: {CONFIG['instruction_prompt_file']}")
    print(f"  - Output dir: {CONFIG['output_dir']}")
    
    # =========================================================================
    # 1. Load Model from SFT Checkpoint
    # =========================================================================
    print("\n[1/4] Loading model from SFT checkpoint...")
    
    # The checkpoint already has the special tokens and LoRA weights
    model, processor = FastVisionModel.from_pretrained(
        CONFIG["base_model_name"],
        load_in_4bit=CONFIG["load_in_4bit"],
        max_seq_length=CONFIG["max_seq_length"],
        use_gradient_checkpointing="unsloth",
    )
    
    # =========================================================================
    # 2. Check LoRA adapters
    # =========================================================================
    print("\n[2/4] Checking LoRA adapters...")
    
    # Check if model already has LoRA adapters from SFT
    from peft import PeftModel
    if isinstance(model, PeftModel) or hasattr(model, 'peft_config'):
        print("LoRA adapters already present from SFT checkpoint")
        print("Continuing training with existing adapters...")
    else:
        print("Adding new LoRA adapters...")
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
            modules_to_save=["embed_tokens", "lm_head"],
        )
    
    # =========================================================================
    # 3. Create Dataset
    # =========================================================================
    print("\n[3/4] Creating dataset...")
    
    train_dataset = GRPOVisionDatasetJSON(
        image_root=CONFIG["image_root"],
        text_root=CONFIG["text_root"],
        instruction=INSTRUCTION,
    )
    
    print(f"Training samples: {len(train_dataset)}")
    
    # =========================================================================
    # 4. Configure and Run GRPO Training
    # =========================================================================
    print("\n[4/4] Starting GRPO training...")
    
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
        
        # Logging and saving
        logging_steps=CONFIG["logging_steps"],
        logging_dir=CONFIG["logging_dir"],
        save_steps=CONFIG["save_steps"],
        save_total_limit=CONFIG["save_total_limit"],
        output_dir=CONFIG["output_dir"],
        report_to="tensorboard",
        
        # Advanced GRPO settings
        loss_type="dr_grpo",
        log_completions=True,
        
        # Seed
        seed=CONFIG["seed"],
    )
    
    # Initialize custom metrics callback
    metrics_callback = GRPOMetricsCallback(output_dir=CONFIG["output_dir"])
    
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=processor,
        train_dataset=train_dataset,
        reward_funcs=[
            block_classification_reward_v2,
            table_accuracy_reward,
            table_structure_reward,
            content_similarity_reward,
            format_compliance_reward_v2,
        ],
        callbacks=[metrics_callback],
    )
    
    print("\nTraining configuration:")
    print(f"  - Learning rate: {CONFIG['learning_rate']}")
    print(f"  - Batch size: {CONFIG['batch_size']}")
    print(f"  - Gradient accumulation: {CONFIG['gradient_accumulation_steps']}")
    print(f"  - Num generations: {CONFIG['num_generations']}")
    print(f"  - Epochs: {CONFIG['num_train_epochs']}")
    print(f"  - Output dir: {CONFIG['output_dir']}")
    
    print("\nReward functions:")
    print("  1. block_classification_reward_v2")
    print("  2. table_accuracy_reward")
    print("  3. table_structure_reward")
    print("  4. content_similarity_reward")
    print("  5. format_compliance_reward_v2")
    
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")
    
    # Train!
    trainer_stats = trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    model.save_pretrained(f"{CONFIG['output_dir']}/final_model")
    processor.save_pretrained(f"{CONFIG['output_dir']}/final_model")
    
    print("\n" + "=" * 70)
    print("GRPO Training complete!")
    print(f"Model saved to: {CONFIG['output_dir']}/final_model")
    print("=" * 70)
    
    return trainer_stats


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()

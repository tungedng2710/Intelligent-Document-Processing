"""
GRPO Fine-tuning for Qwen3-VL on Bank Reports v1.2 (JSON Output)
==================================================================

Trains the model to extract structured JSON from bank report images using
Group Relative Policy Optimization (GRPO).

Dataset: /root/tungn197/idp/data/r3_bank_reports/train_data_v1.2
  - giay_phong_toa_tam_khoa_tai_khoan (100 samples)
  - giay_rut_tien (100 samples)
  - phieu_hach_toan (100 samples)

Output format: Pure JSON (from annotation "json" field only, ignoring layout/bbox)
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
        logging.FileHandler('grpo_training_bank_reports_v12.log')
    ]
)
logger = logging.getLogger(__name__)

# Import JSON-specific reward functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.json_reward_functions import (
    json_validity_reward,
    json_key_matching_reward,
    json_value_similarity_reward,
    json_structure_reward,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Model settings
    "base_model_name": "Qwen/Qwen3-VL-8B-Instruct",
    "load_in_4bit": False,
    "max_seq_length": 8192,

    # LoRA settings
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,

    # Dataset paths - Bank Reports v1.2
    "data_root": "/root/tungn197/idp/data/r3_bank_reports/train_data_v1.2",
    "instruction_prompt_file": str(
        Path(__file__).parent.parent.parent / "prompts" / "bank_report_ver_1.0.txt"
    ),

    # GRPO Training settings
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_generations": 5,
    "learning_rate": 5e-6,
    "num_train_epochs": 3,
    "max_prompt_length": 1024,
    "max_completion_length": 4096,
    "output_dir": "outputs_grpo_bank_reports_v12",

    # Logging and checkpointing
    "save_steps": 10,
    "save_total_limit": 10,
    "logging_steps": 1,
    "logging_dir": "outputs_grpo_bank_reports_v12/logs",

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
        self.start_time = None

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)

        self.tb_writer = SummaryWriter(log_dir=f"{output_dir}/logs/grpo_detailed")
        logger.info(f"GRPO Metrics Callback initialized. Logs at: {output_dir}/logs")

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        logger.info(f"Training started at {self.start_time}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        step = state.global_step

        metrics_to_track = {
            'reward': logs.get('reward', None),
            'reward_std': logs.get('reward_std', None),
            'rewards/json_validity_reward': logs.get('rewards/json_validity_reward', None),
            'rewards/json_key_matching_reward': logs.get('rewards/json_key_matching_reward', None),
            'rewards/json_value_similarity_reward': logs.get('rewards/json_value_similarity_reward', None),
            'rewards/json_structure_reward': logs.get('rewards/json_structure_reward', None),
            'loss': logs.get('loss', None),
            'policy_loss': logs.get('policy_loss', None),
            'kl': logs.get('kl', None),
            'kl_coef': logs.get('kl_coef', None),
            'completion_length': logs.get('completion_length', None),
            'prompt_length': logs.get('prompt_length', None),
            'learning_rate': logs.get('learning_rate', None),
            'grad_norm': logs.get('grad_norm', None),
        }

        for key, value in metrics_to_track.items():
            if value is not None:
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

        self.metrics_history.append({
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **{k: v for k, v in metrics_to_track.items() if v is not None}
        })

        if step % 10 == 0:
            reward = logs.get('reward', 0)
            kl = logs.get('kl', 0)
            loss = logs.get('loss', 0)
            logger.info(f"Step {step}: reward={reward:.4f}, kl={kl:.4f}, loss={loss:.4f}")

    def on_save(self, args, state, control, **kwargs):
        metrics_file = f"{self.output_dir}/metrics_step_{state.global_step}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")

    def on_train_end(self, args, state, control, **kwargs):
        final_metrics_file = f"{self.output_dir}/metrics_final.json"
        with open(final_metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        self.tb_writer.close()

        end_time = datetime.now()
        duration = end_time - self.start_time
        logger.info(f"Training completed at {end_time}")
        logger.info(f"Total training time: {duration}")
        logger.info(f"Final metrics saved to {final_metrics_file}")

        print("\n" + "=" * 70)
        print("To visualize training metrics, run:")
        print(f"  tensorboard --logdir={self.output_dir}/logs")
        print("=" * 70 + "\n")


# =============================================================================
# DATASET CLASS
# =============================================================================

class BankReportGRPODataset(Dataset):
    """
    Dataset for GRPO training on bank reports v1.2.
    Loads all 3 categories, using the "json" field from annotations as the
    ground truth output. Ignores layout, bboxes, and other annotation fields.
    """

    def __init__(
        self,
        data_root: str,
        instruction: str,
        max_samples: Optional[int] = None,
    ):
        self.instruction = instruction
        self.samples = []

        data_root = Path(data_root)

        # Walk all subdirectories (categories)
        for category_dir in sorted(data_root.iterdir()):
            if not category_dir.is_dir():
                continue

            annotations_dir = category_dir / "annotations"
            images_dir = category_dir / "images"

            if not annotations_dir.exists() or not images_dir.exists():
                logger.warning(f"Skipping {category_dir.name}: missing annotations/ or images/")
                continue

            for ann_file in sorted(annotations_dir.glob("*.json")):
                base_name = ann_file.stem

                # Find matching image
                image_file = None
                for ext in ['.png', '.jpg', '.jpeg']:
                    candidate = images_dir / (base_name + ext)
                    if candidate.exists():
                        image_file = candidate
                        break

                if image_file is None:
                    logger.warning(f"No image found for {ann_file}")
                    continue

                self.samples.append({
                    'image_path': str(image_file),
                    'annotation_path': str(ann_file),
                    'category': category_dir.name,
                })

        if max_samples and len(self.samples) > max_samples:
            random.seed(CONFIG["seed"])
            self.samples = random.sample(self.samples, max_samples)

        logger.info(f"Loaded {len(self.samples)} samples from {data_root}")

        # Log per-category counts
        from collections import Counter
        cat_counts = Counter(s['category'] for s in self.samples)
        for cat, cnt in sorted(cat_counts.items()):
            logger.info(f"  {cat}: {cnt} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            # Load image
            image = Image.open(sample['image_path']).convert('RGB')

            # Resize to reduce context length
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.LANCZOS)

            # Load annotation and extract only "json" field
            with open(sample['annotation_path'], 'r', encoding='utf-8') as f:
                annotation = json.load(f)

            json_content = annotation.get("json", {})
            # Serialize to string as the reference answer
            answer = json.dumps(json_content, ensure_ascii=False, indent=2)

            # Create prompt
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.instruction},
                    ],
                },
            ]

            return {
                "prompt": prompt,
                "image": image,
                "answer": answer,
            }

        except Exception as e:
            logger.error(f"Error loading sample {idx} ({sample['image_path']}): {e}")
            return None


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    print("=" * 70)
    print("GRPO Fine-tuning for Bank Reports v1.2 (JSON Output) - Qwen3-VL")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"  - Data root: {CONFIG['data_root']}")
    print(f"  - Instruction file: {CONFIG['instruction_prompt_file']}")
    print(f"  - Output dir: {CONFIG['output_dir']}")

    # =========================================================================
    # 1. Load Model
    # =========================================================================
    print("\n[1/4] Loading model...")

    model, processor = FastVisionModel.from_pretrained(
        CONFIG["base_model_name"],
        load_in_4bit=CONFIG["load_in_4bit"],
        max_seq_length=CONFIG["max_seq_length"],
        use_gradient_checkpointing="unsloth",
    )

    # =========================================================================
    # 2. Setup LoRA
    # =========================================================================
    print("\n[2/4] Setting up LoRA adapters...")

    from peft import PeftModel
    if isinstance(model, PeftModel) or hasattr(model, 'peft_config'):
        print("LoRA adapters already present")
    else:
        print("Adding new LoRA adapters...")
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=CONFIG["r"],
            lora_alpha=CONFIG["lora_alpha"],
            lora_dropout=CONFIG["lora_dropout"],
            bias="none",
            random_state=CONFIG["seed"],
        )

    # =========================================================================
    # 3. Create Dataset
    # =========================================================================
    print("\n[3/4] Creating dataset...")

    train_dataset = BankReportGRPODataset(
        data_root=CONFIG["data_root"],
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

    metrics_callback = GRPOMetricsCallback(output_dir=CONFIG["output_dir"])

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=processor,
        train_dataset=train_dataset,
        reward_funcs=[
            json_validity_reward,
            json_key_matching_reward,
            json_value_similarity_reward,
            json_structure_reward,
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

    print("\nReward functions (JSON-specific):")
    print("  1. json_validity_reward       - Valid JSON parsing check")
    print("  2. json_key_matching_reward   - Key structure F1 score")
    print("  3. json_value_similarity_reward - Value content accuracy (NED + Jaccard)")
    print("  4. json_structure_reward      - Combined structural + content score")

    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")

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


if __name__ == "__main__":
    main()

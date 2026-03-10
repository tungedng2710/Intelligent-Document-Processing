# Models Directory

This directory contains all model artifacts organized by type.

## Structure

### `checkpoints/`
Training checkpoints from fine-tuning experiments.

- **`base-sft/`** - Base supervised fine-tuning checkpoints
  - checkpoint-30, checkpoint-100, checkpoint-200, checkpoint-300, checkpoint-400, checkpoint-2000
  
- **`grpo/`** - Group Relative Policy Optimization (GRPO) training checkpoints
  - checkpoint-80 through checkpoint-170 (increments of 10)
  
- **`sft-tagged/`** - Supervised fine-tuning with tagged data
  - checkpoint-300 through checkpoint-700 (increments of 100)

### `lora/`
LoRA (Low-Rank Adaptation) adapter weights for parameter-efficient fine-tuning.

### `merged/`
Fully merged models with LoRA weights integrated into the base model.

- **`qwen3_vl_grpo_170/`** - GRPO checkpoint-170 merged with base Qwen3-VL model

## Usage

### Using Checkpoints

```bash
# With base model + LoRA checkpoint
MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct \
LORA_PATH=models/checkpoints/sft-tagged/checkpoint-300 \
./scripts/start_api.sh
```

### Using Merged Models

```bash
# Merged model (faster inference)
MODEL_PATH=models/merged/qwen3_vl_grpo_170 \
./scripts/start_api_2.sh
```

## Checkpoint Naming Convention

- `checkpoint-N` where N is the training step/epoch number
- Higher numbers indicate later training stages
- Choose checkpoints based on validation metrics

## Best Checkpoints

Based on evaluation metrics:

- **Production**: `merged/qwen3_vl_grpo_170` - Best GRPO performance
- **SFT Tagged**: `checkpoints/sft-tagged/checkpoint-300` - Stable SFT performance
- **Experimental**: `checkpoints/grpo/checkpoint-170` - Latest GRPO checkpoint

## Merging LoRA Checkpoints

To merge a checkpoint with the base model:

```bash
cd src/utils
python unsloth_convert_models.py \
  --base-model Qwen/Qwen3-VL-8B-Instruct \
  --lora-path ../../models/checkpoints/grpo/checkpoint-170 \
  --output-dir ../../models/merged/qwen3_vl_grpo_170
```

## Storage Notes

- Checkpoints can be large (several GB each)
- Consider keeping only the best performing checkpoints
- Use `.gitignore` to prevent accidentally committing large model files
- Merged models are larger than base + LoRA but faster for inference

## Cleaning Up

To remove old checkpoints and free space:

```bash
# Remove older SFT checkpoints (keep only best)
rm -rf models/checkpoints/sft-tagged/checkpoint-{400,500,600,700}

# Remove intermediate GRPO checkpoints (keep final)
rm -rf models/checkpoints/grpo/checkpoint-{80,90,100,110,120,130,140,150,160}
```

# Quick Model Reference Guide

## 📦 Available Models

### Production Models (Merged - Fastest)
- **`models/merged/qwen3_vl_grpo_170/`** (17GB)
  - Best performing GRPO checkpoint
  - LoRA weights merged into base model
  - Fastest inference, no adapter loading needed
  - **Recommended for production use**

### LoRA Checkpoints (Flexible)

#### Base SFT (`models/checkpoints/base-sft/`)
- checkpoint-30, checkpoint-100, checkpoint-200
- checkpoint-300, checkpoint-400, checkpoint-2000
- Total: 6 checkpoints

#### GRPO Training (`models/checkpoints/grpo/`)
- checkpoint-80 through checkpoint-170 (steps of 10)
- Total: 10 checkpoints
- **Best: checkpoint-170** (merged version available)

#### SFT Tagged (`models/checkpoints/sft-tagged/`)
- checkpoint-300 through checkpoint-700 (steps of 100)
- Total: 5 checkpoints
- **Recommended: checkpoint-300**

## 🚀 Quick Start Commands

### Production (Fastest)
```bash
cd /home/app/tungn197/Qwen3-VL
./scripts/start_api_2.sh
```
Uses `models/merged/qwen3_vl_grpo_170/` by default

### With Custom Checkpoint
```bash
MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct \
LORA_PATH=models/checkpoints/sft-tagged/checkpoint-300 \
./scripts/start_api.sh
```

### Environment Variables
```bash
# Override model path
export MODEL_PATH=/custom/path/to/model

# Override LoRA path
export LORA_PATH=models/checkpoints/grpo/checkpoint-170

# Then run
./scripts/start_api.sh
```

## 📊 Model Selection Guide

### Use Merged Model When:
- ✅ Need fastest inference
- ✅ Production deployment
- ✅ Don't need to switch checkpoints frequently
- ✅ Have sufficient disk space (17GB)

### Use Base + LoRA When:
- ✅ Testing different checkpoints
- ✅ Experimenting with models
- ✅ Limited disk space
- ✅ Need flexibility to swap adapters

## 💾 Storage Information

Total storage used:
- **Checkpoints**: 5.6GB (21 checkpoints)
- **Merged models**: 17GB (1 model)
- **Total**: ~22.6GB

## 🧹 Cleanup Recommendations

If storage is limited, consider removing intermediate checkpoints:

```bash
# Keep only best GRPO checkpoint
rm -rf models/checkpoints/grpo/checkpoint-{80,90,100,110,120,130,140,150,160}

# Keep only best SFT checkpoints
rm -rf models/checkpoints/sft-tagged/checkpoint-{400,500,600,700}

# This would save approximately 3-4GB
```

## 🔄 Converting Checkpoint to Merged Model

To create a merged model from a checkpoint:

```bash
cd src/utils
python unsloth_convert_models.py \
  --base-model Qwen/Qwen3-VL-8B-Instruct \
  --lora-path ../../models/checkpoints/grpo/checkpoint-170 \
  --output-dir ../../models/merged/custom_merged_model
```

## 📍 Current Default Paths

Scripts use these defaults:
- `start_api.sh`: `models/checkpoints/sft-tagged/checkpoint-300`
- `start_api_2.sh`: `models/merged/qwen3_vl_grpo_170`

Override with environment variables as shown above.

## 🎯 Checkpoint Performance Summary

| Checkpoint | Type | Notes |
|------------|------|-------|
| grpo/checkpoint-170 | GRPO | ⭐ Best overall performance |
| sft-tagged/checkpoint-300 | SFT | Stable, good baseline |
| base-sft/checkpoint-2000 | Base | Longest training |

For detailed evaluation metrics, see `docs/result.md`.

from unsloth import FastVisionModel # FastLanguageModel for LLMs
from datasets import load_dataset
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


instruction = """
**Task:** Perform document layout analysis and content extraction. Extract all text from the image and organize it into logical semantic blocks.

**Grouping Logic:**
1. **Visual Boundaries:** Use hard graphical lines and significant whitespace gaps to identify potential block separations.
2. **Semantic Coherence (Priority):**
   - **Merge:** Group spatially distinct text segments into the *same block* if they form a continuous sentence, a single multi-line address, or a single data value, regardless of soft layout breaks.
   - **Split:** Separate text into *different blocks* if they represent distinct logical entities or unrelated data fields, even if they are visually close.

**Output Format Rules:**
- **Separators:** Insert `\n---\n` between distinct blocks.
- **Headers:** If a block has a clear label, format the label as `## Label Name`.
- **Tables:** If a region contains row-column data (with or without borders), extract it as a standard HTML `<table>`.
- **Content:** Transcribe text exactly as it appears.

**Output:**
"""

model, tokenizer = FastVisionModel.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    load_in_4bit = False, 
    use_gradient_checkpointing = "unsloth"
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

# Dataset paths
image_root_path = "/home/public/ocr/idp/data-synthesizer/render/bol_synthesis/images"
text_root_path = "/home/public/ocr/idp/data-synthesizer/render/bol_synthesis/markdown"

# Custom Dataset class for lazy loading
class LazyVisionDataset(Dataset):
    """Dataset that loads images and text on-demand during training."""
    
    def __init__(self, image_root, text_root, instruction):
        self.instruction = instruction
        self.samples = []
        
        image_path = Path(image_root)
        text_path = Path(text_root)
        
        # Get all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(image_path.glob(f'**/*{ext}')))
        
        # Store only file paths
        for image_file in image_files:
            relative_path = image_file.relative_to(image_path)
            text_file = text_path / relative_path.with_suffix('.md')
            
            if text_file.exists():
                self.samples.append({
                    'image_path': str(image_file),
                    'text_path': str(text_file)
                })
        
        print(f"Initialized dataset with {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Load and return a single sample only when requested."""
        sample = self.samples[idx]
        
        try:
            # Load image on-demand
            image = Image.open(sample['image_path']).convert('RGB')
            
            # Load text on-demand
            with open(sample['text_path'], 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Return in conversation format
            conversation = [
                { "role": "user",
                  "content" : [
                    {"type" : "text",  "text"  : self.instruction},
                    {"type" : "image", "image" : image} ]
                },
                { "role" : "assistant",
                  "content" : [
                    {"type" : "text",  "text"  : text} ]
                },
            ]
            return { "messages" : conversation }
        
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a dummy sample to avoid breaking training
            return { "messages" : [] }

# Create lazy dataset - no data loaded yet!
train_dataset = LazyVisionDataset(image_root_path, text_root_path, instruction)

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = train_dataset,  # Use lazy dataset
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 30,
        # num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 8000,
    ),
)

trainer_stats = trainer.train()
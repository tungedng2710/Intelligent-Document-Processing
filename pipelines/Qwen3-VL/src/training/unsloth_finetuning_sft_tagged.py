from unsloth import FastVisionModel
from datasets import load_dataset
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

import torch
import math
import sys
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import random

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.prompt_library import get_prompt

# Load instruction from prompt library
try:
    instruction = get_prompt('training_sft')
except Exception as e:
    print(f"Warning: Failed to load prompt from library: {e}. Using fallback.")
    instruction = """
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

model, tokenizer = FastVisionModel.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    load_in_4bit = False, 
    use_gradient_checkpointing = "unsloth"
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,

    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407
)

# Dataset paths
image_root_path = "/home/app/idp/data/synthetic-bol/images"
text_root_path = "/home/app/idp/data/synthetic-bol/markdown"

anchor_image_path = "/home/app/idp/data/real-bol/bol_sribd_images"
anchor_markdown_path = "/home/app/idp/data/real-bol/bol_markdown"

def transform_markdown_to_blocks(text):
    """
    Transform markdown text by splitting on '---' separators and classifying
    each block as either text_block or table_block.
    
    Args:
        text (str): Raw markdown text from file
        
    Returns:
        str: Transformed text with blocks wrapped in appropriate tags
    """
    # Split by "---" separator
    blocks = text.split('---')
    
    # Initialize result list
    result = []
    
    for block in blocks:
        # Strip whitespace to check content
        stripped_block = block.strip()
        
        # Skip empty blocks
        if not stripped_block:
            continue
        
        # Check if block contains table tag
        if '<table>' in stripped_block:
            result.append(f'<table_block>\n{stripped_block}\n</table_block>')
        else:
            result.append(f'<text_block>\n{stripped_block}\n</text_block>')
    
    # Join all blocks with double newlines for readability
    return '\n'.join(result)


class LazyVisionDataset(Dataset):
    """Dataset that loads images and text on-demand during training."""
    
    def __init__(self, image_root, text_root, instruction, image_ext='.png'):
        self.instruction = instruction
        self.samples = []
        self.image_ext = image_ext
        
        image_path = Path(image_root)
        text_path = Path(text_root)
        
        # Get all markdown files first (ensures both image and label exist)
        text_files = list(text_path.glob('**/*.md'))
        
        # For each markdown file, check if corresponding image exists
        for text_file in text_files:
            relative_path = text_file.relative_to(text_path)
            # Change extension to the specified image extension
            image_file = image_path / relative_path.with_suffix(image_ext)
            
            if image_file.exists():
                self.samples.append({
                    'image_path': str(image_file),
                    'text_path': str(text_file)
                })
        
        print(f"Initialized dataset with {len(self.samples)} samples from {image_root}")
    
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
            text = transform_markdown_to_blocks(text)
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
            return { "messages" : [] }


class BalancedMixedDataset(Dataset):
    """
    Dataset that combines main and anchor datasets ensuring each batch
    has approximately the specified anchor ratio.
    
    Creates a structured dataset where samples are organized to maintain
    the desired ratio within each batch_size window.
    """
    
    def __init__(self, main_dataset, anchor_dataset, batch_size=16, anchor_ratio=0.3, seed=3407):
        self.main_dataset = main_dataset
        self.anchor_dataset = anchor_dataset
        self.batch_size = batch_size
        self.anchor_ratio = anchor_ratio
        self.seed = seed
        
        # Calculate samples per batch
        self.anchor_per_batch = math.ceil(batch_size * anchor_ratio)
        self.main_per_batch = batch_size - self.anchor_per_batch
        
        # Calculate how many complete batches we can create
        self.num_batches = len(main_dataset) // self.main_per_batch
        self.total_length = self.num_batches * batch_size
        
        # Pre-generate the sampling order
        self._generate_sampling_order()
        
        print(f"Balanced Mixed Dataset:")
        print(f"  Main dataset: {len(main_dataset)} samples")
        print(f"  Anchor dataset: {len(anchor_dataset)} samples")
        print(f"  Batch size: {batch_size}")
        print(f"  Per batch: {self.main_per_batch} main + {self.anchor_per_batch} anchor")
        print(f"  Total batches: {self.num_batches}")
        print(f"  Total samples used: {self.total_length}")
    
    def _generate_sampling_order(self):
        """Pre-generate which samples to use and in what order."""
        rng = random.Random(self.seed)
        
        self.sample_indices = []
        
        # Create indices for both datasets
        main_indices = list(range(len(self.main_dataset)))
        anchor_indices = list(range(len(self.anchor_dataset)))
        
        rng.shuffle(main_indices)
        rng.shuffle(anchor_indices)
        
        # Build sample order batch by batch
        for batch_idx in range(self.num_batches):
            batch_samples = []
            
            # Add main samples
            main_start = batch_idx * self.main_per_batch
            for i in range(self.main_per_batch):
                idx = main_start + i
                if idx >= len(main_indices):
                    idx = idx % len(main_indices)
                batch_samples.append(('main', main_indices[idx]))
            
            # Add anchor samples
            anchor_start = (batch_idx * self.anchor_per_batch) % len(anchor_indices)
            for i in range(self.anchor_per_batch):
                idx = (anchor_start + i) % len(anchor_indices)
                batch_samples.append(('anchor', anchor_indices[idx]))
            
            # Shuffle within batch
            rng.shuffle(batch_samples)
            self.sample_indices.extend(batch_samples)
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        """Return sample based on pre-generated sampling order."""
        if idx >= len(self.sample_indices):
            idx = idx % len(self.sample_indices)
        
        dataset_type, actual_idx = self.sample_indices[idx]
        
        if dataset_type == 'main':
            return self.main_dataset[actual_idx]
        else:  # anchor
            return self.anchor_dataset[actual_idx]


# Create both datasets with lazy loading
# For main dataset: check all image extensions
main_dataset = LazyVisionDataset(image_root_path, text_root_path, instruction)

# For anchor dataset: start from markdown files and look for .png images
anchor_dataset = LazyVisionDataset(
    image_root=anchor_image_path,
    text_root=anchor_markdown_path,
    instruction=instruction,
    image_ext='.png'
)

# Create balanced mixed dataset
train_dataset = BalancedMixedDataset(
    main_dataset=main_dataset,
    anchor_dataset=anchor_dataset,
    batch_size=16,
    anchor_ratio=0.3,
    seed=3407
)

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = train_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 4,
        warmup_steps = 30,
        max_steps = 1000,          
        save_steps = 100,
        save_total_limit = 5,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",

        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 8192,
    ),
)

trainer_stats = trainer.train()
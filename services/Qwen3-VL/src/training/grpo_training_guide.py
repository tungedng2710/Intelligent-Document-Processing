"""
===============================================================================
GRPO (Group Relative Policy Optimization) Training Guide for Qwen3-VL Models
===============================================================================

This comprehensive guide covers GRPO training with Unsloth's FastVisionModel
for Qwen3-VL models. Based on official Unsloth documentation and notebooks.

Resources:
- https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation
- https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/vision-reinforcement-learning-vlm-rl
- https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo
- Qwen3_VL GRPO notebook: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision-GRPO.ipynb
"""

# =============================================================================
# SECTION 1: IMPORTS AND SETUP
# =============================================================================

# Core imports for GRPO training
from unsloth import FastVisionModel
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import re


# =============================================================================
# SECTION 2: MODEL LOADING
# =============================================================================

def load_model_for_grpo(
    model_name: str = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    max_seq_length: int = 16384,  # Must be long for VLMs
    load_in_4bit: bool = True,
    fast_inference: bool = False,  # Set True to enable vLLM fast inference
    gpu_memory_utilization: float = 0.8,
    lora_rank: int = 16,
):
    """
    Load a Qwen3-VL model for GRPO training with LoRA.
    
    Parameters:
    -----------
    model_name : str
        The model to load. Supported Qwen3-VL options:
        - "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit" (4-bit quantized)
        - "Qwen/Qwen3-VL-8B-Instruct" (full precision)
        - "unsloth/Qwen3-VL-8B-Instruct" (unsloth optimized)
        
    max_seq_length : int
        Maximum sequence length. VLMs need longer sequences (16384+ recommended)
        
    load_in_4bit : bool
        Use 4-bit quantization to reduce memory. Set False for 16-bit LoRA.
        
    fast_inference : bool
        Enable vLLM fast inference for generation during GRPO.
        Note: vLLM does not support LoRA on vision layers yet.
        
    gpu_memory_utilization : float
        GPU memory fraction to use (0.0-1.0). Reduce if OOM.
        
    lora_rank : int
        LoRA rank. Higher = smarter but slower. 16-32 recommended for GRPO.
    
    Returns:
    --------
    model, tokenizer : tuple
        The loaded model and tokenizer/processor
    """
    
    # Load base model
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    
    # Add LoRA adapters for GRPO training
    # IMPORTANT: For vision GRPO, finetune_vision_layers must be False
    # because vLLM doesn't support LoRA on vision layers yet
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # Must be False for GRPO with vLLM
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_rank,
        lora_alpha=lora_rank,  # Recommended: alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        use_gradient_checkpointing="unsloth",  # Reduces memory usage
    )
    
    return model, tokenizer


# =============================================================================
# SECTION 3: DATASET FORMATTING FOR GRPO
# =============================================================================

# Define reasoning tokens for structured output
REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"
SOLUTION_START = "<answer>"
SOLUTION_END = "</answer>"


def format_prompt_for_grpo(question: str, include_reasoning: bool = True) -> str:
    """
    Format a question into a prompt that encourages structured reasoning.
    
    Parameters:
    -----------
    question : str
        The question to ask about the image
    include_reasoning : bool
        Whether to request reasoning before the answer
        
    Returns:
    --------
    str : The formatted prompt text
    """
    if include_reasoning:
        text_content = (
            f"{question}. Also first provide your reasoning or working out "
            f"on how you would go about solving the question between "
            f"{REASONING_START} and {REASONING_END} "
            f"and then your final answer between "
            f"{SOLUTION_START} and (put a single float here) {SOLUTION_END}"
        )
    else:
        text_content = question
    
    return text_content


def create_conversation_format(example: dict) -> dict:
    """
    Convert a dataset example into the correct conversation format for GRPO.
    
    For GRPO with vision models, the dataset must have:
    - 'prompt': List of messages in chat format (with images as placeholders)
    - 'image': The actual PIL Image
    - Optional metadata for reward functions (e.g., 'answer')
    
    The prompt format for vision models:
    [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # Placeholder for the image
                {"type": "text", "text": "Your question here"},
            ],
        },
    ]
    
    Parameters:
    -----------
    example : dict
        Raw dataset example with 'question', 'decoded_image'/'image', 'answer'
        
    Returns:
    --------
    dict : Formatted example with 'prompt', 'image', 'answer'
    """
    text_content = format_prompt_for_grpo(example.get('question', ''))
    
    # Construct the prompt in multi-modal format
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # Placeholder for the image
                {"type": "text", "text": text_content},
            ],
        },
    ]
    
    # Get the image (handle both 'image' and 'decoded_image' column names)
    image = example.get("decoded_image", example.get("image"))
    
    return {
        "prompt": prompt,
        "image": image,
        "answer": example.get("answer", ""),
    }


def prepare_dataset_for_grpo(
    dataset_name: str = "AI4Math/MathVista",
    split: str = "testmini",
    filter_numeric: bool = True,
    resize_images: bool = True,
    image_size: tuple = (512, 512),
):
    """
    Prepare a dataset for GRPO training with vision models.
    
    Parameters:
    -----------
    dataset_name : str
        HuggingFace dataset name
    split : str
        Dataset split to use
    filter_numeric : bool
        Filter to keep only numeric answers (for math tasks)
    resize_images : bool
        Resize images to reduce context length
    image_size : tuple
        Target image size (width, height)
        
    Returns:
    --------
    Dataset : Prepared dataset for GRPO
    """
    from PIL import Image
    
    dataset = load_dataset(dataset_name, split=split)
    
    # Filter for numeric answers if specified
    if filter_numeric:
        def is_numeric_answer(example):
            try:
                float(example.get("answer", ""))
                return True
            except (ValueError, TypeError):
                return False
        
        dataset = dataset.filter(is_numeric_answer)
    
    # Resize images to reduce context length
    if resize_images:
        def resize_example(example):
            image = example.get("decoded_image", example.get("image"))
            if image is not None:
                image = image.resize(image_size)
            example["decoded_image"] = image
            return example
        
        dataset = dataset.map(resize_example)
    
    # Convert to RGB (required for training)
    def convert_to_rgb(example):
        image = example.get("decoded_image", example.get("image"))
        if image is not None and hasattr(image, 'mode'):
            if image.mode != 'RGB':
                image = image.convert('RGB')
        example["decoded_image"] = image
        return example
    
    dataset = dataset.map(convert_to_rgb)
    
    # Apply conversation formatting
    dataset = dataset.map(create_conversation_format)
    
    return dataset


def apply_chat_template_to_dataset(dataset, tokenizer):
    """
    Apply the chat template to convert prompts to tokenized format.
    
    Parameters:
    -----------
    dataset : Dataset
        Dataset with 'prompt' column in chat format
    tokenizer : PreTrainedTokenizer
        The tokenizer/processor from the model
        
    Returns:
    --------
    Dataset : Dataset with tokenized prompts
    """
    def apply_template(example):
        templated_prompt = tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=False,
            add_generation_prompt=True,  # Must add for GRPO generation
        )
        return {"prompt": templated_prompt}
    
    return dataset.map(apply_template)


# =============================================================================
# SECTION 4: REWARD FUNCTIONS
# =============================================================================

def formatting_reward_func(completions: list, **kwargs) -> list[float]:
    """
    Reward function that checks if the completion follows the expected format.
    
    This function rewards completions that have:
    - Exactly one <reasoning>...</reasoning> block (+1.0)
    - Exactly one <answer>...</answer> block (+1.0)
    
    Also penalizes Qwen-specific issues like excessive 'addCriterion' tokens.
    
    Parameters:
    -----------
    completions : list
        List of completion strings from the model
    **kwargs : dict
        Additional arguments (not used but required by TRL)
        
    Returns:
    --------
    list[float] : Reward scores for each completion
    """
    thinking_pattern = f'{REASONING_START}(.*?){REASONING_END}'
    answer_pattern = f'{SOLUTION_START}(.*?){SOLUTION_END}'

    scores = []
    for completion in completions:
        score = 0.0
        
        # Check for reasoning block
        thinking_matches = re.findall(thinking_pattern, completion, re.DOTALL)
        if len(thinking_matches) == 1:
            score += 1.0
        
        # Check for answer block
        answer_matches = re.findall(answer_pattern, completion, re.DOTALL)
        if len(answer_matches) == 1:
            score += 1.0

        # Fix up addCriterion issues (Qwen-specific quirk)
        # See: https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl#qwen-2.5-vl-vision-rl-issues-and-quirks
        # Penalize excessive addCriterion and newlines
        if len(completion) != 0:
            removal = completion.replace("addCriterion", "").replace("\n", "")
            if (len(completion) - len(removal)) / len(completion) >= 0.5:
                score -= 2.0

        scores.append(score)
    
    return scores


def correctness_reward_func(
    prompts: list,
    completions: list,
    answer: list,
    **kwargs
) -> list[float]:
    """
    Reward function that checks if the extracted answer matches the ground truth.
    
    Extracts the answer from between <answer>...</answer> tags and compares
    with the ground truth.
    
    Parameters:
    -----------
    prompts : list
        List of prompts (used for debugging output)
    completions : list
        List of completion strings from the model
    answer : list
        List of ground truth answers
    **kwargs : dict
        Additional arguments
        
    Returns:
    --------
    list[float] : Reward scores (2.0 for correct, 0.0 for incorrect)
    """
    answer_pattern = f'{SOLUTION_START}(.*?){SOLUTION_END}'

    responses = [
        re.findall(answer_pattern, completion, re.DOTALL)
        for completion in completions
    ]
    
    # Debug print (optional)
    if prompts:
        q = prompts[0]
        print('-' * 20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", 
              f"\nResponse:{completions[0]}")
    
    scores = []
    for r, a in zip(responses, answer):
        if len(r) == 1 and a == r[0].replace('\n', ''):
            scores.append(2.0)  # Correct answer
        else:
            scores.append(0.0)  # Incorrect or missing answer
    
    return scores


def length_penalty_reward(
    completions: list,
    max_length: int = 500,
    penalty_factor: float = 0.001,
    **kwargs
) -> list[float]:
    """
    Reward function that penalizes overly long completions.
    
    Parameters:
    -----------
    completions : list
        List of completion strings
    max_length : int
        Target maximum length (tokens are approximated by characters)
    penalty_factor : float
        Penalty per character over max_length
        
    Returns:
    --------
    list[float] : Penalty scores (negative for long completions)
    """
    scores = []
    for completion in completions:
        length = len(completion)
        if length > max_length:
            penalty = -penalty_factor * (length - max_length)
            scores.append(penalty)
        else:
            scores.append(0.0)
    return scores


def custom_multi_reward_func(
    prompts: list,
    completions: list,
    answer: list = None,
    **kwargs
) -> list[float]:
    """
    Example of combining multiple reward signals.
    
    This demonstrates how to create a composite reward function
    that weighs multiple aspects of the completion.
    
    Parameters:
    -----------
    prompts, completions, answer : lists
        Standard reward function inputs
        
    Returns:
    --------
    list[float] : Combined reward scores
    """
    # Get individual reward components
    format_scores = formatting_reward_func(completions)
    
    if answer is not None:
        correctness_scores = correctness_reward_func(prompts, completions, answer)
    else:
        correctness_scores = [0.0] * len(completions)
    
    length_scores = length_penalty_reward(completions)
    
    # Combine with weights
    combined_scores = []
    for f, c, l in zip(format_scores, correctness_scores, length_scores):
        # Weight: correctness is most important, then format, then length
        combined = 0.3 * f + 0.6 * c + 0.1 * l
        combined_scores.append(combined)
    
    return combined_scores


# =============================================================================
# SECTION 5: GRPO TRAINER CONFIGURATION
# =============================================================================

def create_grpo_config(
    output_dir: str = "outputs",
    learning_rate: float = 5e-6,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    num_generations: int = 2,
    max_prompt_length: int = 1024,
    max_completion_length: int = 1024,
    num_train_epochs: float = 0.5,
    use_gspo: bool = True,  # Enable GSPO variant
    loss_type: str = "dr_grpo",  # Dr. GRPO recommended
    **kwargs
) -> GRPOConfig:
    """
    Create a GRPOConfig for vision model training.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save checkpoints
    learning_rate : float
        Learning rate (5e-6 recommended for GRPO)
    per_device_train_batch_size : int
        Batch size per GPU (1 recommended due to memory)
    gradient_accumulation_steps : int
        Gradient accumulation (increase to 4 for smoother training)
    num_generations : int
        Number of completions to generate per prompt (decrease if OOM)
    max_prompt_length : int
        Maximum prompt token length
    max_completion_length : int
        Maximum completion token length
    num_train_epochs : float
        Number of training epochs
    use_gspo : bool
        Enable GSPO (Group Sequence Policy Optimization) variant
    loss_type : str
        GRPO loss variant: "grpo", "bnpo", "dr_grpo", "dapo"
        - "grpo": Original GRPO
        - "bnpo": Batch-normalized policy optimization
        - "dr_grpo": Dr. GRPO (recommended, per-token loss)
        - "dapo": DAPO variant
        
    Returns:
    --------
    GRPOConfig : Configuration for GRPOTrainer
    """
    config_kwargs = {
        # Optimizer settings
        "learning_rate": learning_rate,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "weight_decay": 0.1,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",
        
        # Logging
        "logging_steps": 1,
        "log_completions": False,  # Set True to see completions in logs
        "report_to": "none",  # Can use "wandb" for Weights & Biases
        
        # Batch settings
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_generations": num_generations,
        
        # Sequence lengths
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
        
        # Training duration
        "num_train_epochs": num_train_epochs,
        
        # Checkpointing
        "save_steps": 60,
        "output_dir": output_dir,
        
        # Gradient clipping
        "max_grad_norm": 0.1,
        
        # Loss configuration
        "loss_type": loss_type,
    }
    
    # Enable GSPO if requested
    if use_gspo:
        config_kwargs.update({
            "importance_sampling_level": "sequence",
            "mask_truncated_completions": False,
        })
    
    # Add any additional kwargs
    config_kwargs.update(kwargs)
    
    return GRPOConfig(**config_kwargs)


def create_grpo_trainer(
    model,
    tokenizer,
    train_dataset,
    training_args: GRPOConfig,
    reward_funcs: list = None,
) -> GRPOTrainer:
    """
    Create a GRPOTrainer instance for vision model training.
    
    Parameters:
    -----------
    model : PreTrainedModel
        The model with LoRA adapters
    tokenizer : PreTrainedTokenizer
        The tokenizer/processor
    train_dataset : Dataset
        The prepared training dataset
    training_args : GRPOConfig
        The training configuration
    reward_funcs : list
        List of reward functions to use
        
    Returns:
    --------
    GRPOTrainer : The configured trainer
    """
    if reward_funcs is None:
        reward_funcs = [formatting_reward_func, correctness_reward_func]
    
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,  # Use processing_class for vision models
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
    )
    
    return trainer


# =============================================================================
# SECTION 6: COMPLETE TRAINING EXAMPLE
# =============================================================================

def run_grpo_training_example():
    """
    Complete example of GRPO training with Qwen3-VL.
    
    This function demonstrates the full pipeline:
    1. Load model with LoRA
    2. Prepare dataset
    3. Configure trainer
    4. Run training
    5. Save model
    """
    print("=" * 60)
    print("GRPO Training Example for Qwen3-VL")
    print("=" * 60)
    
    # Step 1: Load model
    print("\n[1/5] Loading model...")
    model, tokenizer = load_model_for_grpo(
        model_name="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
        max_seq_length=16384,
        load_in_4bit=True,
        lora_rank=16,
    )
    print("Model loaded successfully!")
    
    # Step 2: Prepare dataset
    print("\n[2/5] Preparing dataset...")
    train_dataset = prepare_dataset_for_grpo(
        dataset_name="AI4Math/MathVista",
        split="testmini",
        filter_numeric=True,
        resize_images=True,
        image_size=(512, 512),
    )
    
    # Apply chat template
    train_dataset = apply_chat_template_to_dataset(train_dataset, tokenizer)
    print(f"Dataset prepared with {len(train_dataset)} examples")
    
    # Step 3: Configure trainer
    print("\n[3/5] Configuring trainer...")
    training_args = create_grpo_config(
        output_dir="grpo_outputs",
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_prompt_length=1024,
        max_completion_length=1024,
        num_train_epochs=0.5,
        use_gspo=True,
        loss_type="dr_grpo",
    )
    
    trainer = create_grpo_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        training_args=training_args,
        reward_funcs=[formatting_reward_func, correctness_reward_func],
    )
    print("Trainer configured!")
    
    # Step 4: Run training
    print("\n[4/5] Starting training...")
    print("Note: It may take 150-200 steps before seeing reward improvements.")
    print("The first ~100 steps often show 0 reward - this is normal!")
    trainer.train()
    print("Training complete!")
    
    # Step 5: Save model
    print("\n[5/5] Saving model...")
    model.save_lora("grpo_lora")
    print("LoRA saved to 'grpo_lora'")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return model, tokenizer, trainer


# =============================================================================
# SECTION 7: INFERENCE AFTER GRPO TRAINING
# =============================================================================

def run_inference(model, tokenizer, image, prompt: str):
    """
    Run inference with a GRPO-trained model.
    
    Parameters:
    -----------
    model : PreTrainedModel
        The trained model
    tokenizer : PreTrainedTokenizer
        The tokenizer/processor
    image : PIL.Image
        The input image
    prompt : str
        The text prompt
        
    Returns:
    --------
    str : The model's response
    """
    from transformers import TextStreamer
    
    # Enable inference mode
    FastVisionModel.for_inference(model)
    
    # Prepare inputs
    inputs = tokenizer(
        image,
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    
    # Generate with streaming
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    outputs = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=1024,
        use_cache=True,
        temperature=1.0,
        min_p=0.1,
    )
    
    # Decode and return
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def load_grpo_trained_model(lora_path: str = "grpo_lora"):
    """
    Load a GRPO-trained LoRA model for inference.
    
    Parameters:
    -----------
    lora_path : str
        Path to the saved LoRA adapter
        
    Returns:
    --------
    model, tokenizer : tuple
    """
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=lora_path,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer


# =============================================================================
# SECTION 8: QWEN3-VL SPECIFIC CONSIDERATIONS
# =============================================================================

"""
QWEN3-VL SPECIFIC NOTES:

1. addCriterion Issue:
   - Qwen VL models may output "addCriterion" tokens during inference
   - This is an inherent model quirk and can be ignored
   - The formatting_reward_func penalizes excessive occurrences
   - See: https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl#qwen-2.5-vl-vision-rl-issues-and-quirks

2. Sequence Length:
   - VLMs require longer max_seq_length (16384+ recommended)
   - Images consume significant token space
   - Resize images to 512x512 to reduce context length

3. Vision Layers:
   - For GRPO with vLLM, finetune_vision_layers must be False
   - vLLM doesn't support LoRA on vision layers yet
   - Vision GRPO still works effectively on language layers only

4. Memory Management:
   - Use load_in_4bit=True to reduce memory
   - Use gradient_checkpointing="unsloth" for 2x longer contexts
   - Reduce num_generations if OOM
   - Reduce batch_size to 1 if needed

5. Training Dynamics:
   - GRPO training may show 0 reward for first ~100 steps
   - This is normal - the model is exploring
   - Patience is key - improvements come after 150-200 steps
   - Monitor the reward column in logs

6. Best Practices:
   - Use dr_grpo loss_type for per-token loss
   - Enable GSPO with importance_sampling_level="sequence"
   - Use small learning rate (5e-6)
   - Use adamw_8bit optimizer
   - Set max_grad_norm=0.1 for stability
"""


# =============================================================================
# SECTION 9: ADVANCED REWARD FUNCTIONS
# =============================================================================

def create_llm_as_judge_reward(judge_model, judge_tokenizer):
    """
    Create a reward function that uses another LLM as a judge.
    
    This is an advanced technique where a larger model evaluates
    the quality of completions.
    
    Parameters:
    -----------
    judge_model : PreTrainedModel
        The judge model
    judge_tokenizer : PreTrainedTokenizer
        The judge tokenizer
        
    Returns:
    --------
    Callable : A reward function
    """
    def llm_judge_reward(
        prompts: list,
        completions: list,
        **kwargs
    ) -> list[float]:
        scores = []
        
        for prompt, completion in zip(prompts, completions):
            # Create judge prompt
            judge_prompt = f"""Rate the following response on a scale of 0-10.
            
Question: {prompt}
Response: {completion}

Score (0-10):"""
            
            # Get judge response
            inputs = judge_tokenizer(judge_prompt, return_tensors="pt").to("cuda")
            outputs = judge_model.generate(**inputs, max_new_tokens=10)
            response = judge_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse score
            try:
                score = float(response.strip().split()[-1]) / 10.0
            except:
                score = 0.5  # Default neutral score
            
            scores.append(score)
        
        return scores
    
    return llm_judge_reward


def create_rule_based_math_reward():
    """
    Create a more sophisticated math reward function.
    
    This checks for:
    - Proper equation formatting
    - Step-by-step reasoning
    - Final answer validity
    
    Returns:
    --------
    Callable : A reward function
    """
    def math_reward(
        prompts: list,
        completions: list,
        answer: list = None,
        **kwargs
    ) -> list[float]:
        scores = []
        
        for i, completion in enumerate(completions):
            score = 0.0
            
            # Check for step markers
            step_pattern = r'(Step \d+|First|Then|Next|Finally|Therefore)'
            steps = re.findall(step_pattern, completion, re.IGNORECASE)
            score += min(len(steps) * 0.2, 1.0)  # Up to 1.0 for steps
            
            # Check for mathematical expressions
            math_pattern = r'[\d\+\-\*\/\=\(\)]+'
            math_exprs = re.findall(math_pattern, completion)
            score += min(len(math_exprs) * 0.1, 0.5)  # Up to 0.5 for math
            
            # Check answer correctness
            if answer and i < len(answer):
                answer_pattern = f'{SOLUTION_START}(.*?){SOLUTION_END}'
                found = re.findall(answer_pattern, completion, re.DOTALL)
                if found:
                    try:
                        extracted = float(found[0].strip())
                        expected = float(answer[i])
                        if abs(extracted - expected) < 0.01:
                            score += 3.0  # Big bonus for correct answer
                        elif abs(extracted - expected) < 0.1:
                            score += 1.5  # Partial credit for close
                    except:
                        pass
            
            scores.append(score)
        
        return scores
    
    return math_reward


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
    print("\nThis module provides utilities for GRPO training with Qwen3-VL.")
    print("\nKey functions:")
    print("- load_model_for_grpo(): Load and configure model with LoRA")
    print("- prepare_dataset_for_grpo(): Prepare dataset in correct format")
    print("- formatting_reward_func(): Check output format structure")
    print("- correctness_reward_func(): Check answer correctness")
    print("- create_grpo_config(): Create training configuration")
    print("- create_grpo_trainer(): Create the GRPO trainer")
    print("- run_grpo_training_example(): Complete training example")
    print("\nRun run_grpo_training_example() to see a complete training pipeline.")

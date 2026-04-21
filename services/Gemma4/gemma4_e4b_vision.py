# -*- coding: utf-8 -*-
import torch
from datasets import load_dataset
from transformers import TextStreamer
from unsloth import FastVisionModel, get_chat_template
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# --- Model ---

model, processor = FastVisionModel.from_pretrained(
    "unsloth/gemma-4-E4B-it",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=32,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
    target_modules="all-linear",
)

processor = get_chat_template(processor, "gemma-4")

# --- Data ---

dataset = load_dataset("unsloth/LaTeX_OCR", split="train")

instruction = "Write the LaTeX representation for this image."


def convert_to_conversation(sample):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": sample["text"]}]},
        ]
    }


converted_dataset = [convert_to_conversation(sample) for sample in dataset]

# --- Training ---

trainer = SFTTrainer(
    model=model,
    train_dataset=converted_dataset,
    processing_class=processor.tokenizer,
    data_collator=UnslothVisionDataCollator(model, processor),
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="steps",
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=2048,
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {round(used_memory / max_memory * 100, 3)} %.")
print(f"Peak reserved memory for training % of max memory = {round(used_memory_for_lora / max_memory * 100, 3)} %.")

# --- Inference ---

image = dataset[10]["image"]
messages = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": instruction}],
    }
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

text_streamer = TextStreamer(processor, skip_prompt=True)
result = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128,
                        use_cache=True, temperature=1.0, top_p=0.95, top_k=64)

# --- Save ---

model.save_pretrained("gemma_4_lora")
processor.save_pretrained("gemma_4_lora")

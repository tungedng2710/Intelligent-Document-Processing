from unsloth import FastVisionModel
from transformers import TextStreamer
from PIL import Image
import torch
import argparse

# Load the fine-tuned model
def load_model(model_path="outputs", base_model="Qwen/Qwen3-VL-8B-Instruct"):
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading model from {model_path}...")
    
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,  # Load from fine-tuned checkpoint
        load_in_4bit = False,
    )
    
    FastVisionModel.for_inference(model)  # Enable for inference!
    print("Model loaded successfully!")
    
    return model, tokenizer

def infer(model, tokenizer, image_path, instruction=None, max_new_tokens=8192, temperature=0.9, min_p=0.1):
    """Run inference on a single image."""
    
    # Default instruction if not provided
    if instruction is None:
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
    
    # Load image
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path).convert('RGB')
    
    # Prepare messages
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    
    # Prepare inputs
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=False,  # Don't truncate to avoid image token mismatch
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate with streaming output
    print("\n" + "="*80)
    print("GENERATED OUTPUT:")
    print("="*80)
    
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    output = model.generate(
        **inputs, 
        streamer=text_streamer, 
        max_new_tokens=max_new_tokens,
        use_cache=True, 
        temperature=temperature, 
        min_p=min_p
    )
    
    print("="*80)
    
    # Decode the full output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Run inference with fine-tuned Qwen3-VL model')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model_path', type=str, default='outputs', help='Path to the fine-tuned model')
    parser.add_argument('--max_tokens', type=int, default=8192, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.01, help='Sampling temperature')
    parser.add_argument('--min_p', type=float, default=0.1, help='Minimum probability for sampling')
    parser.add_argument('--output', type=str, help='Optional output file to save results')
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    # Run inference
    result = infer(
        model, 
        tokenizer, 
        args.image,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        min_p=args.min_p
    )
    
    # Save output if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
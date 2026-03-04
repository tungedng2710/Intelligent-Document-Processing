import gradio as gr
from unsloth import FastVisionModel
from PIL import Image
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.prompt_library import get_prompt

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model(model_path="outputs/checkpoint-30"):
    """Load the fine-tuned model and tokenizer."""
    global model, tokenizer
    
    print(f"Loading model from {model_path}...")
    
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=False,
    )
    
    FastVisionModel.for_inference(model)
    print("Model loaded successfully!")
    
    return "Model loaded successfully!"

def process_image(image, temperature=1.0, max_tokens=2048):
    """Process uploaded image and return extracted text."""
    
    if model is None or tokenizer is None:
        return "Error: Model not loaded. Please load the model first."
    
    # Load instruction from prompt library
    try:
        instruction = get_prompt('api_semantic')
    except Exception as e:
        print(f"Warning: Failed to load prompt from library: {e}. Using fallback.")
        instruction = """
**Task:** Perform document layout analysis and content extraction. Extract all text from the image and organize it into logical semantic blocks.

**Grouping Logic:**
1. **Visual Boundaries:** Use hard graphical lines and significant whitespace gaps to identify potential block separations.
2. **Semantic Coherence (Priority):**
   - **Merge:** Group spatially distinct text segments into the *same block* if they form a continuous sentence, a single multi-line address, or a single data value, regardless of soft layout breaks.
   - **Split:** Separate text into *different blocks* if they represent distinct logical entities or unrelated data fields, even if they are visually close.

**Output Format Rules:**
- **Separators:** Insert `\\n---\\n` between distinct blocks.
- **Headers:** If a block has a clear label, format the label as `## Label Name`.
- **Tables:** If a region contains row-column data (with or without borders), extract it as a standard HTML `<table>`.
- **Content:** Transcribe text exactly as it appears.

**Output:**
"""
    
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
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
            truncation=False,
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            use_cache=True,
            temperature=float(temperature),
            min_p=0.1
        )
        
        # Decode output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|im_start|>assistant" in generated_text:
            result = generated_text.split("<|im_start|>assistant")[-1].strip()
        else:
            result = generated_text
        
        return result
        
    except Exception as e:
        return f"Error during processing: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Document Layout Analysis & OCR") as demo:
    gr.Markdown("# 📄 Document Layout Analysis & OCR")
    gr.Markdown("Upload a document image to extract and organize text into semantic blocks.")
    
    with gr.Tab("Inference"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Document Image")
                
                with gr.Row():
                    temperature_slider = gr.Slider(
                        minimum=0.1, 
                        maximum=2.0, 
                        value=1.0, 
                        step=0.1,
                        label="Temperature"
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=512,
                        maximum=8000,
                        value=2048,
                        step=256,
                        label="Max Tokens"
                    )
                
                submit_btn = gr.Button("Extract Text", variant="primary")
                
            with gr.Column():
                output_text = gr.Textbox(
                    label="Extracted Text",
                    lines=25,
                    max_lines=50
                )
        
        submit_btn.click(
            fn=process_image,
            inputs=[image_input, temperature_slider, max_tokens_slider],
            outputs=output_text
        )
        
        gr.Examples(
            examples=[
                ["/root/tungn197/license_plate_recognition/data/test_samples/bol1.png", 1.0, 2048],
            ],
            inputs=[image_input, temperature_slider, max_tokens_slider],
            label="Example Documents"
        )
    
    with gr.Tab("Model Management"):
        with gr.Row():
            model_path_input = gr.Textbox(
                value="outputs/checkpoint-30",
                label="Model Path"
            )
            load_btn = gr.Button("Load Model", variant="primary")
        
        load_status = gr.Textbox(label="Status", interactive=False)
        
        load_btn.click(
            fn=load_model,
            inputs=model_path_input,
            outputs=load_status
        )
    
    gr.Markdown("""
    ## Usage Instructions:
    1. **Load Model**: Go to the "Model Management" tab and click "Load Model" (uses checkpoint-30 by default)
    2. **Upload Image**: Click or drag an image to the upload area
    3. **Adjust Parameters** (optional):
       - Temperature: Controls randomness (lower = more deterministic)
       - Max Tokens: Maximum length of generated output
    4. **Extract Text**: Click the button to process the image
    
    ## Output Format:
    - Blocks separated by `---`
    - Headers formatted as `## Header Name`
    - Tables in HTML format
    """)

if __name__ == "__main__":
    # Auto-load model on startup
    print("Starting application...")
    print("Loading model automatically...")
    load_model()
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

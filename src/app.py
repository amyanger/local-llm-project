"""
Gradio Web UI for fine-tuned LLM chat.

Usage:
    python src/app.py
    python src/app.py --model models/openhermes-chat --port 7860 --share

The app will be available at http://localhost:7860
"""

import os
from pathlib import Path

# Configure HuggingFace to use local cache (D: drive) and disable hf_transfer
os.environ.setdefault("HF_HOME", str(Path(__file__).parent.parent / ".cache"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import argparse
import gradio as gr
import torch

# Import model loading from inference module
from inference import load_model


# Global model and tokenizer (loaded once at startup)
model = None
tokenizer = None


def respond(message, history):
    """Generate a response to the user message."""
    global model, tokenizer

    if model is None or tokenizer is None:
        return "Error: Model not loaded. Please wait for the model to load."

    # Build messages list
    system_prompt = "You are a helpful, friendly assistant."
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history
    for item in history:
        if isinstance(item, dict):
            # New Gradio format: {"role": "...", "content": "..."}
            role = item.get("role", "user")
            content = item.get("content", "")
            # Content might be a list in some cases, extract text
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            messages.append({"role": role, "content": str(content)})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            # Old format: (user_msg, assistant_msg)
            user_msg, assistant_msg = item
            messages.append({"role": "user", "content": str(user_msg)})
            if assistant_msg:
                messages.append({"role": "assistant", "content": str(assistant_msg)})

    # Add current message
    messages.append({"role": "user", "content": str(message)})

    # Format prompt using chat template
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback to manual ChatML format
        formatted_prompt = ""
        for msg in messages:
            formatted_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        formatted_prompt += "<|im_start|>assistant\n"

    # Tokenize and generate
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=40,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response.strip()


def main():
    global model, tokenizer

    parser = argparse.ArgumentParser(description="Launch Gradio chat interface")
    parser.add_argument(
        "--model",
        type=str,
        default="models/openhermes-chat",
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link",
    )

    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    print("This may take a few minutes on first run...")
    model, tokenizer = load_model(args.model)
    print("Model loaded successfully!")

    # Create simple chat interface
    demo = gr.ChatInterface(
        fn=respond,
        title="Local LLM Chat",
        description="Chat with your fine-tuned OpenHermes model (QLoRA fine-tuned)",
        examples=["What is Python?", "Explain recursion", "Write a haiku about coding"],
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()

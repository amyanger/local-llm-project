"""
Run inference with fine-tuned model (ChatML format).

Usage:
    python src/inference.py --model models/openhermes-chat --prompt "What is Python?"
"""

import os
from pathlib import Path

# Configure HuggingFace to use local cache (D: drive) and disable hf_transfer
os.environ.setdefault("HF_HOME", str(Path(__file__).parent.parent / ".cache"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import argparse
import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_path: str, max_seq_length: int = 2048):
    """Load fine-tuned LoRA model for inference."""
    # Check if this is a LoRA adapter by looking for adapter_config.json
    adapter_config_path = Path(model_path) / "adapter_config.json"

    if adapter_config_path.exists():
        # Load adapter config to get base model
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "teknium/OpenHermes-2.5-Mistral-7B")
        print(f"Loading base model: {base_model_name}")

        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Load tokenizer from adapter first (has the special tokens)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Resize embeddings if tokenizer has more tokens than model
        if len(tokenizer) > model.config.vocab_size:
            print(f"Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))

        # Load LoRA adapter
        print(f"Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model.eval()
    else:
        # Direct model loading (not a LoRA adapter)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: str = "You are a helpful, friendly assistant.",
):
    """Generate response for a prompt using ChatML format."""
    # Build messages for chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    # Use the tokenizer's chat template if available
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback to manual ChatML format
        formatted_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=40,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    # Decode only the new tokens (skip the input)
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response.strip()


def interactive_mode(model, tokenizer):
    """Run interactive chat loop."""
    print("\nInteractive mode. Type 'quit' to exit.\n")

    while True:
        prompt = input("You: ").strip()

        if prompt.lower() in ["quit", "exit", "q"]:
            break

        if not prompt:
            continue

        response = generate(model, tokenizer, prompt)
        print(f"\nModel: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/openhermes-chat",
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to process (omit for interactive mode)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )

    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    model, tokenizer = load_model(args.model)
    print("Model loaded successfully!\n")

    if args.prompt:
        # Single prompt mode
        response = generate(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"Response: {response}")
    else:
        # Interactive mode
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()

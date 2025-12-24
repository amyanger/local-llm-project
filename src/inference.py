"""
Run inference with fine-tuned model.

Usage:
    python src/inference.py --model models/checkpoints --prompt "What is Python?"
"""

import argparse
import torch
from unsloth import FastLanguageModel


def load_model(model_path: str, max_seq_length: int = 2048):
    """Load fine-tuned model for inference."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Enable faster inference
    FastLanguageModel.for_inference(model)

    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
):
    """Generate response for a prompt."""
    formatted_prompt = f"""### Instruction:
{prompt}

### Response:
"""

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    return response


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
        default="models/checkpoints",
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

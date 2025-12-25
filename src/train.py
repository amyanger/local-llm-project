"""
Local LLM Fine-tuning Script
Uses QLoRA for memory-efficient training on RTX 5090

Usage:
    python src/train.py --model teknium/OpenHermes-2.5-Mistral-7B --dataset mlabonne/ultrachat_200k
"""

import os
from pathlib import Path

# Configure HuggingFace to use local cache (D: drive) and disable hf_transfer
os.environ.setdefault("HF_HOME", str(Path(__file__).parent.parent / ".cache"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def verify_gpu():
    """Verify GPU is available and print info."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Check PyTorch installation.")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()


def load_model(model_name: str, max_seq_length: int = 2048):
    """Load model with 4-bit quantization for memory efficiency."""
    print(f"Loading model: {model_name}")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def format_chatml(example):
    """Format dataset example into ChatML format."""
    # Handle conversational format (messages list)
    if "messages" in example:
        formatted = ""
        for msg in example["messages"]:
            role = msg["role"]
            content = msg["content"]
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return formatted
    # Handle instruction/response format
    elif "instruction" in example and "response" in example:
        return f"""<|im_start|>user
{example['instruction']}<|im_end|>
<|im_start|>assistant
{example['response']}<|im_end|>"""
    elif "text" in example:
        return example["text"]
    elif "prompt" in example and "completion" in example:
        return f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['completion']}<|im_end|>"
    else:
        return example.get("text", str(example))


def train(
    model_name: str,
    dataset_path: str,
    output_dir: str = "models/checkpoints",
    max_seq_length: int = 2048,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 2e-5,
    max_samples: int = 10000,
):
    """Run fine-tuning."""
    verify_gpu()

    # Load model
    model, tokenizer = load_model(model_name, max_seq_length)

    # Add ChatML special tokens if not present
    special_tokens = ["<|im_start|>", "<|im_end|>"]
    tokens_to_add = [t for t in special_tokens if t not in tokenizer.get_vocab()]
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added special tokens: {tokens_to_add}")

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    if dataset_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        dataset = load_dataset(dataset_path, split="train_sft")

    # Limit dataset size for faster training
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))
        print(f"Using {max_samples} samples for training")

    # Format dataset to ChatML
    print("Formatting dataset to ChatML...")
    dataset = dataset.map(lambda x: {"text": format_chatml(x)}, remove_columns=dataset.column_names)

    print(f"Dataset size: {len(dataset)} examples")

    # Training config using SFTConfig
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=10,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        max_length=max_seq_length,
        packing=False,
        dataset_text_field="text",
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save model
    print(f"Saving model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with QLoRA")
    parser.add_argument(
        "--model",
        type=str,
        default="teknium/OpenHermes-2.5-Mistral-7B",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="HuggingFace dataset or path to JSONL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/openhermes-chat",
        help="Output directory for checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=10000, help="Max training samples (0 for all)")

    args = parser.parse_args()

    train(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples if args.max_samples > 0 else None,
    )


if __name__ == "__main__":
    main()

"""
Local LLM Fine-tuning Script
Uses QLoRA for memory-efficient training on RTX 5090

Usage:
    python src/train.py --model mistralai/Mistral-7B-v0.1 --dataset data/processed/train.jsonl
"""

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


def format_instruction(example):
    """Format dataset example into instruction format."""
    # Handle different dataset formats
    if "instruction" in example and "response" in example:
        return f"""### Instruction:
{example['instruction']}

### Response:
{example['response']}"""
    elif "text" in example:
        return example["text"]
    elif "prompt" in example and "completion" in example:
        return f"{example['prompt']}{example['completion']}"
    else:
        # For openassistant-guanaco format
        return example.get("text", str(example))


def train(
    model_name: str,
    dataset_path: str,
    output_dir: str = "models/checkpoints",
    max_seq_length: int = 2048,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
):
    """Run fine-tuning."""
    verify_gpu()

    # Load model
    model, tokenizer = load_model(model_name, max_seq_length)

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    if dataset_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        dataset = load_dataset(dataset_path, split="train")

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
        max_seq_length=max_seq_length,
        packing=False,
        dataset_text_field="text",
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
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
        default="mistralai/Mistral-7B-v0.1",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to training dataset (JSONL)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")

    args = parser.parse_args()

    train(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()

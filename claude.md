# Local LLM Training Project

## Project Overview

This project is for training/fine-tuning a local LLM on an RTX 5090 (32GB VRAM). Based on research, **fine-tuning an existing model is the recommended approach** over training from scratch.

## Recommendation: Fine-Tune, Don't Train From Scratch

### Why Fine-Tuning Wins

| Aspect | Fine-Tuning | Training from Scratch |
|--------|-------------|----------------------|
| **Cost** | Low (single GPU feasible) | Extremely high (GPU clusters, weeks of compute) |
| **Data Required** | Thousands of examples | Terabytes of text data |
| **Time** | Hours to days | Weeks to months |
| **Performance** | Can outperform larger general models on specific tasks | Requires massive resources to match existing models |
| **Complexity** | Moderate | Very high |

Training from scratch only makes sense for rare/underrepresented languages or completely novel domains where pre-trained models have zero knowledge. For most use cases, fine-tuning is far more practical.

## Recommended Base Models for RTX 5090 (32GB)

### Tier 1: Best for Fine-Tuning

| Model | Parameters | VRAM (Fine-tuning) | Best For |
|-------|------------|-------------------|----------|
| **Mistral 7B** | 7B | ~16GB with QLoRA | General tasks, instruction-following |
| **Llama 3.1 8B** | 8B | ~18GB with QLoRA | General purpose, large ecosystem |
| **Qwen 2.5 7B Coder** | 7B | ~16GB with QLoRA | Code generation and reasoning |
| **CodeLlama 7B** | 7B | ~16GB with QLoRA | Code-specific tasks |

### Tier 2: With Optimization

| Model | Parameters | VRAM (Fine-tuning) | Notes |
|-------|------------|-------------------|-------|
| **Llama 3.1 13B** | 13B | ~24GB with QLoRA | Needs careful memory management |
| **Mistral Small 3** | 24B | ~32GB with aggressive quantization | Cutting-edge performance |

### Tier 3: Inference Only (Quantized)

| Model | Parameters | VRAM (Inference) | Notes |
|-------|------------|-----------------|-------|
| **Llama 3.1 70B** | 70B | ~28GB (4-bit) | Inference only on RTX 5090 |
| **DeepSeek Coder 33B** | 33B | ~20GB (4-bit) | Strong coding capabilities |

## Recommended Approach

**Start with: Mistral 7B or Llama 3.1 8B + QLoRA fine-tuning**

Reasons:
- Well within RTX 5090's capabilities
- Massive community support and documentation
- QLoRA reduces memory from ~60GB to ~16GB
- Training time: hours, not days
- Many successful fine-tuned derivatives exist (OpenHermes, Zephyr)

## PyTorch Setup for RTX 5090 (Blackwell Architecture)

### Critical: Use PyTorch Nightly

The RTX 5090 uses Blackwell architecture (sm_120) which requires CUDA 12.8+. The stable PyTorch only supports up to sm_90, causing errors like:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install PyTorch nightly with CUDA 12.8
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Environment Variables (if needed)

```bash
# Linux/Mac
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
export CUDA_VISIBLE_DEVICES=0
export TORCH_USE_CUDA_DSA=1

# Windows PowerShell
$env:TORCH_CUDA_ARCH_LIST="12.0"
$env:FORCE_CUDA=1
$env:CUDA_VISIBLE_DEVICES=0
```

### Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Training Stack

### Recommended Tools

1. **Unsloth** - 2.5x faster training, memory optimized
   ```bash
   pip install unsloth
   ```

2. **Hugging Face Transformers + PEFT**
   ```bash
   pip install transformers peft accelerate bitsandbytes
   ```

3. **Axolotl** - YAML-based fine-tuning orchestrator
   ```bash
   pip install axolotl
   ```

### Key Techniques

- **QLoRA**: Quantized LoRA - enables 13B model training on consumer GPUs
- **LoRA**: Low-Rank Adaptation - only trains small adapter matrices
- **Flash Attention**: Faster attention computation
- **Gradient Checkpointing**: Trade compute for memory

## Project Structure

```
local-llm-project/
├── claude.md              # This file - project documentation
├── requirements.txt       # Python dependencies
├── setup.py              # Installation script
├── config/
│   └── training_config.yaml
├── data/
│   ├── raw/              # Raw training data
│   └── processed/        # Processed datasets
├── src/
│   ├── data_prep.py      # Data preprocessing
│   ├── train.py          # Training script
│   └── inference.py      # Run inference
├── models/
│   └── checkpoints/      # Saved model checkpoints
└── notebooks/
    └── exploration.ipynb # Experimentation
```

## Quick Start Guide

### 1. Setup Environment
```bash
cd local-llm-project
python -m venv venv
venv\Scripts\activate  # Windows

# Install PyTorch nightly for RTX 5090
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install training dependencies
pip install transformers peft accelerate bitsandbytes datasets unsloth
```

### 2. Prepare Dataset
Create instruction-following pairs in JSONL format:
```json
{"instruction": "What is Python?", "response": "Python is a high-level programming language..."}
{"instruction": "Write a function to add two numbers", "response": "def add(a, b):\n    return a + b"}
```

### 3. Train with QLoRA
```python
from unsloth import FastLanguageModel
from trl import SFTTrainer

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mistralai/Mistral-7B-v0.1",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    max_seq_length=2048,
    # ... additional config
)
trainer.train()
```

## Hardware Considerations

### RTX 5090 Specs
- **VRAM**: 32GB GDDR7
- **Architecture**: Blackwell (sm_120)
- **TDP**: Up to 575W under AI workloads

### Power Supply Requirements
Ensure your PSU can deliver sustained 575W+ to the GPU. Blackwell GPUs consume substantial power under AI workloads.

## Resources

### Documentation
- [Hugging Face Fine-tuning Guide 2025](https://www.philschmid.de/fine-tune-llms-in-2025)
- [Unsloth Documentation](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide)
- [PyTorch RTX 5090 Support](https://discuss.pytorch.org/t/nvidia-geforce-rtx-5090/218954)

### Models
- [Mistral 7B on HuggingFace](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [Llama 3.1 8B on HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B)
- [Qwen 2.5 Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-7B)

### Fine-tuned Examples
- OpenHermes 2.5 (Mistral fine-tune, improved HumanEval 43% → 50.7%)
- Zephyr 7B (Mistral + DPO)
- Solar 10.7B (Llama 2 + Mistral weights)

## Next Steps

1. [ ] Set up Python environment with PyTorch nightly
2. [ ] Download base model (Mistral 7B recommended)
3. [ ] Prepare training dataset (instruction-response pairs)
4. [ ] Run initial training with small dataset
5. [ ] Evaluate and iterate

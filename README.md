# Local LLM Fine-Tuning Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)

A memory-efficient framework for fine-tuning large language models on consumer GPUs using QLoRA. Designed for NVIDIA RTX 5090 (Blackwell architecture) but compatible with RTX 30/40 series.

## Features

- **Memory Efficient**: Fine-tune 7B parameter models with only ~16GB VRAM using 4-bit quantization
- **QLoRA Training**: Parameter-efficient fine-tuning with Low-Rank Adaptation
- **RTX 5090 Ready**: Full support for Blackwell architecture (sm_120) via PyTorch nightly
- **Flexible**: Works with Mistral, Llama, Qwen, and other HuggingFace models
- **Easy to Use**: Simple CLI interface for training and inference

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 16GB+ VRAM
- CUDA 12.8+ (for RTX 50 series) or CUDA 12.1+ (for RTX 30/40 series)

### Installation

```bash
git clone https://github.com/amyanger/local-llm-project.git
cd local-llm-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install PyTorch (RTX 5090 / Blackwell)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Fine-tune Mistral 7B on OpenAssistant dataset
python src/train.py \
    --model mistralai/Mistral-7B-v0.1 \
    --dataset timdettmers/openassistant-guanaco \
    --epochs 3

# Or use your own dataset
python src/train.py \
    --model mistralai/Mistral-7B-v0.1 \
    --dataset data/raw/your_dataset.jsonl \
    --epochs 5
```

### Inference

```bash
# Interactive chat mode
python src/inference.py --model models/checkpoints

# Single prompt
python src/inference.py --model models/checkpoints --prompt "Explain quantum computing"
```

## Project Structure

```
local-llm-project/
├── src/
│   ├── train.py          # QLoRA fine-tuning script
│   └── inference.py      # Model inference and chat
├── data/
│   ├── raw/              # Training datasets
│   └── processed/        # Preprocessed data
├── models/
│   └── checkpoints/      # Saved model weights
├── config/               # Training configurations
├── requirements.txt      # Python dependencies
└── README.md
```

## Supported Models

| Model | Parameters | VRAM Required | Recommended For |
|-------|------------|---------------|-----------------|
| Mistral 7B | 7B | ~16GB | General tasks, instruction-following |
| Llama 3.1 8B | 8B | ~18GB | General purpose, large ecosystem |
| CodeLlama 7B | 7B | ~16GB | Code generation |
| Qwen 2.5 Coder 7B | 7B | ~16GB | Code + reasoning |

## Dataset Format

Training data should be in JSONL format with instruction-response pairs:

```json
{"instruction": "What is machine learning?", "response": "Machine learning is a subset of artificial intelligence..."}
{"instruction": "Write a Python function to sort a list", "response": "def sort_list(lst):\n    return sorted(lst)"}
```

Or use HuggingFace datasets with a `text` field directly.

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `mistralai/Mistral-7B-v0.1` | Base model from HuggingFace |
| `--dataset` | Required | Path to JSONL or HuggingFace dataset |
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 2 | Per-device batch size |
| `--lr` | 2e-4 | Learning rate |
| `--output` | `models/checkpoints` | Output directory |

## Technical Details

### QLoRA Configuration

- **LoRA Rank (r)**: 16
- **LoRA Alpha**: 16
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Quantization**: 4-bit NF4 with double quantization
- **Compute dtype**: BFloat16

### Stack

- **Transformers**: Model loading and tokenization
- **PEFT**: Parameter-efficient fine-tuning (LoRA)
- **BitsAndBytes**: 4-bit quantization
- **TRL**: Supervised fine-tuning trainer
- **Datasets**: Data loading and processing

## Hardware Requirements

### Minimum
- NVIDIA GPU with 16GB VRAM (RTX 4080, 3090, etc.)
- 32GB System RAM
- 50GB Storage

### Recommended
- NVIDIA RTX 5090 (32GB VRAM)
- 64GB System RAM
- 100GB+ NVMe Storage

### RTX 5090 Notes
The RTX 5090 uses Blackwell architecture (sm_120) which requires:
- PyTorch nightly build with CUDA 12.8
- Latest NVIDIA drivers (560+)

## Results

Fine-tuning Mistral 7B on 10K instruction pairs:

| Metric | Before | After |
|--------|--------|-------|
| Training Loss | 2.1 | 0.8 |
| Perplexity | 8.2 | 2.2 |

*Results vary based on dataset quality and training duration.*

## Roadmap

- [ ] Add DPO (Direct Preference Optimization) support
- [ ] Implement evaluation benchmarks
- [ ] Add multi-GPU training with DeepSpeed
- [ ] Create web UI for inference
- [ ] Support for vision-language models

## References

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face TRL](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built for the AI/ML community. Star this repo if you find it useful!**

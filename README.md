# LLM Inference Engine in C++

This repository contains a CPU-first LLM inference engine written in modern C++. It runs GPT-2 and SmolLM2-360M-Instruct without depending on Python for inference. The codebase focuses on clarity and directness, while still using fast CPU kernels where available.

The current implementation uses:
- A simple `Tensor` type with contiguous `float32` storage in [include/tensor.h](include/tensor.h).
- Core ops such as matmul, softmax, GELU, layer norm, and RMSNorm in [src/ops.cpp](src/ops.cpp).
- Two model implementations:
  - GPT-2 in [src/gpt2.cpp](src/gpt2.cpp).
  - SmolLM2 in [src/smollm2.cpp](src/smollm2.cpp).

CPU performance is primarily driven by BLAS (Accelerate on macOS) through `Tensor::matmul` and `Tensor::matmul_ex`. Some attention loops are still implemented in C++ for clarity and can be optimized further. GPU execution is not implemented yet.

## Supported Models

| Model | Architecture | Parameters | Features |
|-------|-------------|------------|----------|
| GPT-2 | Transformer | 124M | Token + Position embeddings, GELU, LayerNorm |
| SmolLM2-360M-Instruct | LLaMA | 360M | RoPE, GQA, SwiGLU, RMSNorm, Chat Template |

## What’s Implemented

- Weight loading from a custom binary format (see [src/model_loader.cpp](src/model_loader.cpp)).
- Tokenization:
  - GPT-2 BPE tokenizer in [src/tokenizer.cpp](src/tokenizer.cpp).
  - SmolLM2 tokenizer and chat template in [src/smollm2_tokenizer.cpp](src/smollm2_tokenizer.cpp).
- Transformer blocks:
  - GPT-2 attention and MLP in [src/gpt2.cpp](src/gpt2.cpp).
  - SmolLM2 GQA attention, RoPE, RMSNorm, and SwiGLU in [src/smollm2.cpp](src/smollm2.cpp).
- Autoregressive generation with sampling settings in the model main programs:
  - GPT-2 entry point in [src/main.cpp](src/main.cpp).
  - SmolLM2 entry point in [src/smollm2_main.cpp](src/smollm2_main.cpp).

## Project Structure

```
.
├── include/
│   ├── tensor.h              # Basic tensor data structure
│   ├── ops.h                 # Neural network operations
│   ├── model_loader.h        # Weight loading utilities
│   ├── tokenizer.h           # GPT-2 tokenizer
│   ├── gpt2.h                # GPT-2 model architecture
│   ├── smollm2.h             # SmolLM2 model architecture
│   └── smollm2_tokenizer.h   # SmolLM2 tokenizer
├── src/
│   ├── main.cpp              # GPT-2 main inference loop
│   ├── model_loader.cpp
│   ├── ops.cpp
│   ├── tokenizer.cpp
│   ├── gpt2.cpp
│   ├── smollm2.cpp           # SmolLM2 implementation
│   ├── smollm2_tokenizer.cpp # SmolLM2 tokenizer
│   └── smollm2_main.cpp      # SmolLM2 main inference
├── scripts/
│   ├── export_weights.py         # Export GPT-2 weights
│   └── export_smollm2_weights.py # Export SmolLM2 weights
└── weights/                  # Directory for model weights
```

## Quick Start: SmolLM2-360M-Instruct

### Prerequisites

- CMake 3.15+
- A C++17 compiler (Apple Clang, Clang, or GCC)
- Python 3.x with `transformers` and `torch` only for weight export

On macOS, you will get the fastest CPU performance by linking to Accelerate (already wired in CMake). Optional OpenMP support can be enabled via Homebrew `libomp`.

### Steps

1) Export SmolLM2 weights (requires Python with transformers):

```bash
pip install transformers torch
python scripts/export_smollm2_weights.py
```

This will download SmolLM2-360M-Instruct from HuggingFace and save weights to `weights/smollm2/smollm2_weights.bin` (~1.4GB).

2) Build the C++ inference engine:

```bash
mkdir -p build
cd build
cmake ..
make
```

3) Run inference:

```bash
# Single prompt
./smollm2_inference --prompt "What is the capital of France?"

# Interactive chat mode
./smollm2_inference --interactive

# With custom parameters
./smollm2_inference --prompt "Explain quantum computing" \
  --temperature 0.7 \
  --max-tokens 256 \
  --top-k 50 \
  --top-p 0.9
```

### SmolLM2 Options

```
Options:
  --weights <path>     Path to weights file (default: weights/smollm2/smollm2_weights.bin)
  --tokenizer <path>   Path to tokenizer directory (default: weights/smollm2)
  --prompt <text>      Input prompt for generation
  --max-tokens <n>     Maximum new tokens to generate (default: 256)
  --temperature <f>    Sampling temperature (default: 0.7)
  --top-k <n>          Top-k sampling (default: 50)
  --top-p <f>          Top-p (nucleus) sampling (default: 0.9)
  --repetition-penalty <f>  Repetition penalty (default: 1.1)
  --interactive        Interactive chat mode
```

## GPT-2 Usage

1) Export GPT-2 weights:

```bash
python scripts/export_weights.py
```

2) Build the C++ inference engine (same as above):

```bash
mkdir -p build
cd build
cmake ..
make
```

3) Run GPT-2 inference:

```bash
./gpt2_inference "Your prompt here"
```

Or use the default prompt:

```bash
./gpt2_inference
```

## SmolLM2 Architecture

SmolLM2-360M uses the LLaMA architecture with the following key components:

### Model Configuration
```
vocab_size: 49152
hidden_size: 960
intermediate_size: 2560
num_hidden_layers: 32
num_attention_heads: 15
num_key_value_heads: 5 (Grouped Query Attention)
head_dim: 64
max_position_embeddings: 8192
rope_theta: 100000
rms_norm_eps: 1e-5
```

### Key Components

1. **RMSNorm** (vs LayerNorm in GPT-2): Root Mean Square normalization without mean centering
2. **Rotary Position Embeddings (RoPE)**: Position encoding applied to Q and K in attention
3. **Grouped Query Attention (GQA)**: 15 query heads share 5 KV heads (3:1 ratio)
4. **SwiGLU Activation**: gate_proj * SiLU(up_proj) in the MLP
5. **KV-Cache**: Efficient autoregressive generation by caching key/value states

## How It Works

### 1. Weight Export (scripts/export_smollm2_weights.py)

The Python script loads SmolLM2 from HuggingFace and exports all weights to a binary format:
- Each tensor is stored with: name length, name, shape dimensions, shape values, and float32 data
- Total: 199 tensors (embedding, 32 transformer layers, final RMSNorm)

### 2. Model Loading (src/model_loader.cpp)

Reads the binary weight file and loads all tensors into memory:
- Parses tensor names and shapes
- Allocates memory for each weight matrix
- Stores in a hashmap for fast lookup

### 3. Tokenization (src/smollm2_tokenizer.cpp)

Downloads SmolLM2 tokenizer from HuggingFace:
- 49,152 token vocabulary
- Byte-Pair Encoding (BPE)
- Special tokens: `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`
- Chat template support for instruct models

### 4. SmolLM2 Architecture (src/smollm2.cpp)

Implements the full LLaMA-style architecture:
- **Token Embeddings**: Converts token IDs to 960-dimensional vectors
- **32 Transformer Blocks**: Each with:
  - RMSNorm (pre-norm)
  - Grouped Query Attention with RoPE (15 heads, 5 KV heads, 64 dims each)
  - Residual connections
  - SwiGLU MLP (960 → 2560 → 960)
- **Final RMSNorm**
- **LM Head**: Projects to vocabulary size (tied with embeddings)

### 5. Text Generation

- Temperature sampling with configurable temperature
- Top-k and top-p (nucleus) filtering
- Repetition penalty
- KV-cache for efficient token-by-token generation
- Stops on `<|im_end|>` token or max tokens

## Current Limitations

1. **Performance**: Fast CPU matmul via BLAS is in place, but attention and other ops are still mostly hand-written loops.
2. **Memory**: All weights are loaded into RAM (~1.4GB for SmolLM2-360M).
3. **Precision**: Float32 only (no quantization yet).
4. **GPU**: No GPU backend (Metal/CUDA/Vulkan) yet.

## Future Improvements

- SIMD optimizations (AVX2, NEON)
- More multi-threading or batched attention
- Weight quantization (int8, int4)
- Flash Attention
- Metal/CUDA support
- Support for larger SmolLM2 variants (1.7B)

## Performance Notes

On macOS, the code uses Accelerate BLAS for matrix multiplication, which is the main speedup path. If you built with OpenMP (via libomp), some attention loops run in parallel. Performance depends heavily on sequence length and model size, and is best measured with a consistent prompt length and token count.

## License

This is a learning project. Feel free to use and modify as needed.

## Acknowledgments

- SmolLM2 by HuggingFace: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct
- Based on LLaMA architecture by Meta
- Inspired by llama.cpp and other minimal inference engines

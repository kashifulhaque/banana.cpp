# LLM Inference Engine in C++

This repository contains a CPU-first LLM inference engine written in modern C++. It runs SmolLM2-360M-Instruct without depending on Python at all. The codebase focuses on clarity and directness, while still using fast CPU kernels where available.

The current implementation uses:
- A simple `Tensor` type with support for `float32`, `float16`, and `bfloat16` storage in [include/tensor.h](include/tensor.h).
- Core ops such as matmul, softmax, GELU, layer norm, and RMSNorm in [src/ops.cpp](src/ops.cpp).
- SmolLM2 model implementation in [src/smollm2.cpp](src/smollm2.cpp).
- Built-in weight downloader that fetches models from HuggingFace in [src/weight_downloader.cpp](src/weight_downloader.cpp).

CPU performance is primarily driven by BLAS (Accelerate on macOS) through `Tensor::matmul` and `Tensor::matmul_ex`. Some attention loops are still implemented in C++ for clarity and can be optimized further. GPU execution is not implemented yet.

## Supported Models

| Model | Architecture | Parameters | Features |
|-------|-------------|------------|----------|
| SmolLM2-360M-Instruct | LLaMA | 360M | RoPE, GQA, SwiGLU, RMSNorm, Chat Template |

## What's Implemented

- Weight loading from a custom binary format (see [src/model_loader.cpp](src/model_loader.cpp)).
- Tokenization:
  - SmolLM2 tokenizer and chat template in [src/smollm2_tokenizer.cpp](src/smollm2_tokenizer.cpp).
- Transformer blocks:
  - SmolLM2 GQA attention, RoPE, RMSNorm, and SwiGLU in [src/smollm2.cpp](src/smollm2.cpp).
- Autoregressive generation with sampling settings in [src/smollm2_main.cpp](src/smollm2_main.cpp).

## Project Structure

```
.
├── include/
│   ├── tensor.h              # Basic tensor data structure (fp32/fp16/bf16)
│   ├── ops.h                 # Neural network operations
│   ├── model_loader.h        # Weight loading utilities
│   ├── weight_downloader.h   # HuggingFace weight downloader
│   ├── smollm2.h             # SmolLM2 model architecture
│   └── smollm2_tokenizer.h   # SmolLM2 tokenizer
├── src/
│   ├── model_loader.cpp
│   ├── weight_downloader.cpp # Download & convert weights from HF
│   ├── ops.cpp
│   ├── smollm2.cpp           # SmolLM2 implementation
│   ├── smollm2_tokenizer.cpp # SmolLM2 tokenizer
│   └── smollm2_main.cpp      # SmolLM2 main inference
└── weights/                  # Directory for model weights
```

## Quick Start: SmolLM2-360M-Instruct

### Prerequisites

- CMake 3.15+
- A C++17 compiler (Apple Clang, Clang, or GCC)
- `curl` for downloading weights from HuggingFace

On macOS, you will get the fastest CPU performance by linking to Accelerate (already wired in CMake). Optional OpenMP support can be enabled via Homebrew `libomp`.

### Steps

1) Build the C++ inference engine:

```bash
mkdir -p build
cd build
cmake ..
make
```

2) Download weights and run inference (weights are downloaded automatically if not present):

```bash
# Download weights with BF16 precision (default, ~720MB)
./smollm2_inference --download

# Or download with FP16 precision
./smollm2_inference --download --precision fp16

# Or download with FP32 precision (~1.4GB)
./smollm2_inference --download --precision fp32
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
  --download           Download model weights from HuggingFace
  --precision <type>   Weight precision: fp32, fp16, bf16 (default: bf16)
```

### Weight Precision

The engine supports three precision levels for weight storage:

| Precision | Size (360M model) | Memory Usage | Quality |
|-----------|-------------------|--------------|---------|
| FP32      | ~1.4 GB           | Higher       | Best    |
| FP16      | ~720 MB           | Lower        | Good    |
| BF16      | ~720 MB           | Lower        | Good (recommended) |

BF16 (bfloat16) is recommended as it maintains the same dynamic range as FP32 while halving memory usage. All computations are performed in FP32 for maximum accuracy.

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

1. **RMSNorm**: Root Mean Square normalization without mean centering
2. **Rotary Position Embeddings (RoPE)**: Position encoding applied to Q and K in attention
3. **Grouped Query Attention (GQA)**: 15 query heads share 5 KV heads (3:1 ratio)
4. **SwiGLU Activation**: gate_proj * SiLU(up_proj) in the MLP
5. **KV-Cache**: Efficient autoregressive generation by caching key/value states

## How It Works

### 1. Weight Download (src/weight_downloader.cpp)

Downloads SmolLM2 weights directly from HuggingFace and converts to an optimized binary format:
- Downloads `model.safetensors` from HuggingFace Hub
- Parses safetensors format and extracts tensor metadata
- Converts to custom binary format with configurable precision (FP32/FP16/BF16)
- Total: 199 tensors (embedding, 32 transformer layers, final RMSNorm)

Binary format (v2):
- Magic number (`SML2`) + version + precision byte
- For each tensor: name length, name, shape dimensions, shape values, data

### 2. Model Loading (src/model_loader.cpp)

Reads the binary weight file and loads all tensors into memory:
- Detects format version and precision from header
- Parses tensor names and shapes
- Loads FP16/BF16 data and converts to FP32 for computation
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
2. **Memory**: All weights are loaded into RAM (~720MB for BF16, ~1.4GB for FP32).
3. **GPU**: No GPU backend (Metal/CUDA/Vulkan) yet.

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

# LLM Inference Engine in C++

A barebones LLM inference engine implemented in pure C++ without dependencies on PyTorch, TensorFlow, or transformers library. Supports GPT-2 and SmolLM2-360M-Instruct. This is similar to projects like llama.cpp but focused on simplicity for learning purposes.

## Supported Models

| Model | Architecture | Parameters | Features |
|-------|-------------|------------|----------|
| GPT-2 | Transformer | 124M | Token + Position embeddings, GELU, LayerNorm |
| **SmolLM2-360M-Instruct** | LLaMA | 360M | RoPE, GQA, SwiGLU, RMSNorm, Chat Template |

## Features

- ✅ Pure C++ implementation (no Python dependencies for inference)
- ✅ Loads model weights from HuggingFace
- ✅ Custom tensor operations (matmul, GELU, SiLU, softmax, layer norm, RMSNorm)
- ✅ Full transformer architecture implementation
- ✅ SmolLM2 with Grouped Query Attention (GQA) and Rotary Position Embeddings (RoPE)
- ✅ Text generation with temperature, top-k, and top-p sampling
- ✅ KV-cache for efficient autoregressive generation
- ✅ Interactive chat mode
- ✅ Automatic tokenizer download

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
- C++17 compatible compiler (GCC, Clang, or MSVC)
- Python 3.x with transformers library (only for weight export)

### Steps

1. **Export SmolLM2 weights** (requires Python with transformers):

```bash
pip install transformers torch
python scripts/export_smollm2_weights.py
```

This will download SmolLM2-360M-Instruct from HuggingFace and save weights to `weights/smollm2/smollm2_weights.bin` (~1.4GB).

2. **Build the C++ inference engine**:

```bash
mkdir -p build
cd build
cmake ..
make
```

3. **Run inference**:

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

1. **Export GPT-2 weights**:

```bash
python scripts/export_weights.py
```

2. **Run GPT-2 inference**:

```bash
./build/gpt2_inference "Your prompt here"
```

This will download GPT-2 from HuggingFace and save weights to `weights/gpt2_weights.bin` (~650MB).

2. **Build the C++ inference engine**:

```bash
mkdir -p build
cd build
cmake ..
make
```

3. **Run inference**:

```bash
./build/gpt2_inference "Your prompt here"
```

Or use the default prompt:

```bash
./build/gpt2_inference
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

### 1. Weight Export (`export_smollm2_weights.py`)

The Python script loads SmolLM2 from HuggingFace and exports all weights to a binary format:
- Each tensor is stored with: name length, name, shape dimensions, shape values, and float32 data
- Total: 199 tensors (embedding, 32 transformer layers, final RMSNorm)

### 2. Model Loading (`model_loader.cpp`)

Reads the binary weight file and loads all tensors into memory:
- Parses tensor names and shapes
- Allocates memory for each weight matrix
- Stores in a hashmap for fast lookup

### 3. Tokenization (`smollm2_tokenizer.cpp`)

Downloads SmolLM2 tokenizer from HuggingFace:
- 49,152 token vocabulary
- Byte-Pair Encoding (BPE)
- Special tokens: `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`
- Chat template support for instruct models

### 4. SmolLM2 Architecture (`smollm2.cpp`)

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

1. **Performance**: Naive matrix multiplication (no SIMD, threading, or GPU)
2. **Memory**: All weights loaded into RAM (~1.4GB for SmolLM2-360M)
3. **Precision**: Using float32 (no quantization)

## Future Improvements

- [ ] SIMD optimizations (AVX2, NEON)
- [ ] Multi-threading for matrix operations
- [ ] Weight quantization (int8, int4)
- [ ] Flash Attention
- [ ] Metal/CUDA support
- [ ] Support for larger SmolLM2 variants (1.7B)

## Performance

On a modern CPU:
- Load time: ~3-5 seconds
- Generation speed: ~0.5-2 tokens/second (naive implementation)

Much faster performance is possible with optimizations (SIMD, quantization, etc.)

## License

This is a learning project. Feel free to use and modify as needed.

## Acknowledgments

- SmolLM2 by HuggingFace: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct
- Based on LLaMA architecture by Meta
- Inspired by llama.cpp and other minimal inference engines

# GPT-2 Inference Engine in C++

A barebones GPT-2 inference engine implemented in pure C++ without dependencies on PyTorch, TensorFlow, or transformers library. This is similar to projects like llama.cpp but focused on simplicity for learning purposes.

## Features

- ✅ Pure C++ implementation (no Python dependencies for inference)
- ✅ Loads GPT-2 weights from HuggingFace
- ✅ Custom tensor operations (matmul, GELU, softmax, layer norm)
- ✅ Full transformer architecture implementation
- ✅ Text generation with greedy sampling
- ✅ Automatic tokenizer download

## Project Structure

```
.
├── include/
│   ├── tensor.h        # Basic tensor data structure
│   ├── ops.h           # Neural network operations
│   ├── model_loader.h  # Weight loading utilities
│   ├── tokenizer.h     # GPT-2 tokenizer
│   └── gpt2.h          # GPT-2 model architecture
├── src/
│   ├── main.cpp        # Main inference loop
│   ├── model_loader.cpp
│   ├── ops.cpp
│   ├── tokenizer.cpp
│   └── gpt2.cpp
├── scripts/
│   └── export_weights.py  # Python script to export GPT-2 weights
└── weights/            # Directory for model weights and tokenizer
```

## Building

### Prerequisites

- CMake 3.15+
- C++17 compatible compiler (GCC, Clang, or MSVC)
- Python 3.x with transformers library (only for weight export)

### Steps

1. **Export GPT-2 weights** (requires Python with transformers):

```bash
pip install transformers torch
python scripts/export_weights.py
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

## How It Works

### 1. Weight Export (`export_weights.py`)

The Python script loads GPT-2 from HuggingFace and exports all weights to a binary format:
- Each tensor is stored with: name length, name, shape dimensions, shape values, and float32 data
- Total: 149 tensors (embeddings, 12 transformer layers, final layer norm, LM head)

### 2. Model Loading (`model_loader.cpp`)

Reads the binary weight file and loads all tensors into memory:
- Parses tensor names and shapes
- Allocates memory for each weight matrix
- Stores in a hashmap for fast lookup

### 3. Tokenization (`tokenizer.cpp`)

Downloads GPT-2 tokenizer from HuggingFace (vocab.json + merges.txt):
- 50,257 token vocabulary
- Byte-Pair Encoding (BPE) with 50,000 merges
- Handles special tokens and space encoding

Note: The current tokenization is simplified and may not perfectly match GPT-2's original tokenizer. This is a known limitation for this barebones implementation.

### 4. GPT-2 Architecture (`gpt2.cpp`)

Implements the full GPT-2 architecture:
- **Token + Position Embeddings**: Converts token IDs to 768-dimensional vectors
- **12 Transformer Blocks**: Each with:
  - Layer Normalization
  - Multi-Head Self-Attention (12 heads, 64 dims each)
  - Residual connections
  - MLP with GELU activation (768 → 3072 → 768)
- **Final Layer Norm**
- **LM Head**: Projects to vocabulary size for next-token prediction

### 5. Text Generation

- Greedy sampling: Picks the token with highest probability
- Autoregressive generation: Each new token is fed back as input
- Stops after N tokens or when end-of-text token is generated

## Current Limitations

1. **Tokenization**: The BPE implementation is simplified and may not match OpenAI's exact tokenization
2. **Sampling**: Only greedy sampling (no temperature, top-k, or top-p)
3. **Performance**: Naive matrix multiplication (no SIMD, threading, or GPU)
4. **Memory**: All weights loaded into RAM, no quantization
5. **Context Length**: Limited to 1024 tokens (GPT-2's context window)

## Future Improvements

- [ ] Proper BPE tokenization matching GPT-2
- [ ] KV caching for faster generation
- [ ] Temperature and top-k/top-p sampling
- [ ] Multi-threading for matrix operations
- [ ] SIMD optimizations (AVX2, NEON)
- [ ] Weight quantization (int8, int4)
- [ ] Batch processing
- [ ] Support for GPT-2 Medium/Large/XL

## Performance

On a modern CPU (M-series Apple Silicon or recent Intel/AMD):
- Load time: ~2-3 seconds
- Generation speed: ~1-2 tokens/second (naive implementation)

Much faster performance is possible with optimizations (KV cache, SIMD, etc.)

## License

This is a learning project. Feel free to use and modify as needed.

## Acknowledgments

- Based on GPT-2 by OpenAI
- Inspired by llama.cpp and other minimal inference engines
- Weights from HuggingFace transformers library

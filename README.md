# LLM Inference Engine

A modular, pure C++ implementation of a small language model inference engine supporting multiple model architectures.

## Supported Models

- **SmolLM2**: 135M, 360M, 1.7B variants
- **Llama 3.2**: 1B, 3B variants  
- **Qwen**: 2.5-0.5B, 3-0.6B variants

## Project Structure

```
banana.cpp/
├── include/
│   ├── config/               # Configuration structures
│   │   ├── llm_config.h      # Model configuration and presets
│   │   └── tokenizer_config.h # Tokenizer configuration and presets
│   ├── core/                 # Core data structures
│   │   ├── tensor.h          # Tensor class with FP16/BF16 support
│   │   └── ops.h             # Basic tensor operations
│   ├── layers/               # Neural network layers
│   │   ├── attention.h       # GQA/MHA/MQA attention
│   │   ├── mlp.h             # SwiGLU and standard MLP
│   │   ├── normalization.h   # RMSNorm and LayerNorm
│   │   └── position_embedding.h # RoPE implementation
│   ├── models/               # Model implementations
│   │   ├── base_model.h      # Abstract base class
│   │   └── llm_model.h       # Unified LLM model
│   ├── tokenizers/           # Tokenization
│   │   ├── bpe_tokenizer.h   # BPE encoding/decoding
│   │   ├── chat_template.h   # Chat template handlers
│   │   └── tokenizer.h       # Unified tokenizer interface
│   ├── registry/             # Model registration
│   │   └── model_registry.h  # Model auto-detection
│   └── utils/                # Utilities
│       ├── model_loader.h    # Weight loading
│       └── weight_downloader.h # HuggingFace download
├── src/                      # Implementation files (mirrors include/)
│   ├── config/
│   ├── core/
│   ├── layers/
│   ├── models/
│   ├── tokenizers/
│   ├── registry/
│   ├── utils/
│   └── main.cpp              # Entry point
└── CMakeLists.txt
```

## Building

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

```bash
# List supported models
./llm_inference --list-models

# Download and run a model
./llm_inference --model smollm2-360m --download
./llm_inference --model smollm2-360m --prompt "What is the capital of France?"

# Interactive mode
./llm_inference --model smollm2-360m --interactive

# Custom settings
./llm_inference --model llama-3.2-1b \
    --temperature 0.8 \
    --top-k 40 \
    --top-p 0.95 \
    --max-tokens 512
```

## Adding a New Model

1. Add model config preset to `include/config/llm_config.h`:
   ```cpp
   inline LLMConfig my_new_model() {
     LLMConfig cfg;
     cfg.model_name = "MyModel";
     // ... set dimensions, etc.
     return cfg;
   }
   ```

2. Add tokenizer preset to `include/config/tokenizer_config.h` if needed

3. Add detection logic in `src/registry/model_registry.cpp`

4. If the model uses novel layer types:
   - Add new layer class in `include/layers/`
   - Implement in `src/layers/`

## Architecture

The engine uses a modular layer-based architecture:

- **Layers**: Reusable components (Attention, MLP, Normalization, RoPE)
- **Models**: Compose layers based on configuration
- **Tokenizers**: Separate BPE logic from chat template handling
- **Registry**: Auto-detect model type from config.json

This design makes it easy to:
- Add new model architectures by composing existing layers
- Add new layer types without modifying models
- Support multiple chat templates and tokenizer formats

## License

MIT

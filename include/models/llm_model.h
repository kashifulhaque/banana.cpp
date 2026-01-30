#ifndef MODELS_LLM_MODEL_H
#define MODELS_LLM_MODEL_H

#include "config/llm_config.h"
#include "models/base_model.h"
#include "layers/attention.h"
#include "layers/mlp.h"
#include "layers/normalization.h"
#include "layers/position_embedding.h"
#include "utils/model_loader.h"
#include <memory>
#include <string>
#include <vector>

// Forward declaration
class ModelLoader;

namespace models {

/// Generic LLM Model supporting multiple architectures
/// Supports: LLaMA, SmolLM2, Qwen, Mistral, and similar transformer-based LLMs
class LLMModel : public BaseModel {
public:
  LLMModel(ModelLoader& loader, const LLMConfig& config);
  virtual ~LLMModel() = default;

  /// Forward pass: given input token IDs, return logits for next token
  Tensor forward(const std::vector<int>& input_ids) override;

  /// Forward pass with KV cache for efficient generation
  Tensor forward_with_cache(const std::vector<int>& input_ids, layers::KVCache& cache);

  /// Generate text given a prompt
  std::vector<int> generate(
    const std::vector<int>& input_ids,
    const SamplingConfig& config = SamplingConfig(),
    TokenCallback token_callback = nullptr
  ) override;

  /// Access configuration
  const LLMConfig& get_config() const override { return config_; }

  /// Check if a token is an EOS token
  bool is_eos_token(int token_id) const override;

protected:
  ModelLoader& loader_;
  LLMConfig config_;

  // Layer components
  std::unique_ptr<layers::RoPE> rope_;
  std::unique_ptr<layers::GroupedQueryAttention> attention_;
  std::unique_ptr<layers::SwiGLUMLP> mlp_;
  std::unique_ptr<layers::RMSNorm> rms_norm_;
  std::unique_ptr<layers::LayerNorm> layer_norm_;

  // Embedding layer
  Tensor embedding(const std::vector<int>& input_ids);

  // Transformer block
  Tensor transformer_block(const Tensor& x, int layer_idx, int start_pos);
  Tensor transformer_block_with_cache(const Tensor& x, int layer_idx, layers::KVCache& cache);

  // Weight access helpers
  const Tensor* get_weight(const std::string& name, bool warn = true);
  std::string layer_weight_name(int layer_idx, const std::string& suffix);

  // Sampling
  int sample_token(const Tensor& logits, const SamplingConfig& config,
                   const std::vector<int>& generated_tokens);
};

/// Factory function to create model from config
std::unique_ptr<LLMModel> create_model(ModelLoader& loader, const LLMConfig& config);

} // namespace models

#endif // MODELS_LLM_MODEL_H

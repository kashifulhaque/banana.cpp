#ifndef LLM_MODEL_H
#define LLM_MODEL_H

#include "llm_config.h"
#include "model_loader.h"
#include "tensor.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

/// Callback type for streaming token generation
/// Called with each new token ID as it's generated
/// Return false to stop generation early
using TokenCallback = std::function<bool(int token_id)>;

/// KV Cache for efficient autoregressive generation
struct KVCache {
  std::vector<Tensor> key_cache;    // [num_layers] of [max_seq, num_kv_heads, head_dim]
  std::vector<Tensor> value_cache;  // [num_layers] of [max_seq, num_kv_heads, head_dim]
  int current_length = 0;

  void init(const LLMConfig& config, int max_seq_len);
  void clear();
};

/// Generic LLM Model supporting multiple architectures
class LLMModel {
public:
  LLMModel(ModelLoader& loader, const LLMConfig& config);
  virtual ~LLMModel() = default;

  /// Forward pass: given input token IDs, return logits for next token
  Tensor forward(const std::vector<int>& input_ids);

  /// Forward pass with KV cache for efficient generation
  Tensor forward_with_cache(const std::vector<int>& input_ids, KVCache& cache);

  /// Generate text given a prompt
  /// If token_callback is provided, it will be called for each generated token
  std::vector<int> generate(
    const std::vector<int>& input_ids,
    const SamplingConfig& config = SamplingConfig(),
    TokenCallback token_callback = nullptr
  );

  /// Access configuration
  const LLMConfig& get_config() const { return config_; }

  /// Check if a token is an EOS token
  bool is_eos_token(int token_id) const;

protected:
  ModelLoader& loader_;
  LLMConfig config_;

  // Precomputed RoPE frequencies
  std::vector<float> cos_cached_;
  std::vector<float> sin_cached_;
  int max_cached_positions_ = 0;

  // Initialization
  void init_rope_cache(int max_seq_len);

  // Core layers
  Tensor embedding(const std::vector<int>& input_ids);
  Tensor transformer_block(const Tensor& x, int layer_idx, int start_pos);
  Tensor transformer_block_with_cache(const Tensor& x, int layer_idx, KVCache& cache);

  // Attention with GQA and RoPE
  Tensor attention(const Tensor& x, int layer_idx, int start_pos);
  Tensor attention_with_cache(const Tensor& x, int layer_idx, KVCache& cache);

  // MLP (SwiGLU for LLaMA-style, standard for others)
  Tensor mlp(const Tensor& x, int layer_idx);

  // Normalization
  Tensor rms_norm(const Tensor& x, const Tensor& weight);
  Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor* bias = nullptr);

  // Apply RoPE to query and key tensors
  void apply_rope(Tensor& q, Tensor& k, int start_pos);

  // Apply Q/K normalization (Qwen3)
  void apply_qk_norm(Tensor& q, Tensor& k, int layer_idx);

  // Linear projections
  Tensor linear(const Tensor& x, const Tensor& weight);
  Tensor linear_with_bias(const Tensor& x, const Tensor& weight, const Tensor& bias);

  // Weight access helpers
  const Tensor* get_weight(const std::string& name, bool warn = true);
  std::string layer_weight_name(int layer_idx, const std::string& suffix);

  // Sampling
  int sample_token(const Tensor& logits, const SamplingConfig& config,
                   const std::vector<int>& generated_tokens);
};

/// Factory function to create model from config
std::unique_ptr<LLMModel> create_model(ModelLoader& loader, const LLMConfig& config);

#endif // LLM_MODEL_H

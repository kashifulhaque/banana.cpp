#ifndef LAYERS_POSITION_EMBEDDING_H
#define LAYERS_POSITION_EMBEDDING_H

#include "core/tensor.h"
#include <vector>

namespace layers {

/// Abstract base class for position embeddings
class PositionEmbedding {
public:
  virtual ~PositionEmbedding() = default;
  
  /// Initialize position embedding cache
  /// @param max_seq_len Maximum sequence length to precompute
  virtual void init(int max_seq_len) = 0;
  
  /// Apply position embedding to query and key tensors
  /// @param q Query tensor of shape [seq_len, num_heads, head_dim]
  /// @param k Key tensor of shape [seq_len, num_kv_heads, head_dim]
  /// @param start_pos Starting position for the sequence
  virtual void apply(Tensor& q, Tensor& k, int start_pos) const = 0;
};

/// Rotary Position Embedding (RoPE)
/// Used by LLaMA, Qwen, Mistral, SmolLM, and most modern LLMs
class RoPE : public PositionEmbedding {
public:
  /// Construct RoPE with given parameters
  /// @param head_dim Dimension of each attention head
  /// @param theta Base theta for frequency computation (default 10000)
  /// @param scaling_factor Scaling factor for extended context (default 1.0)
  RoPE(int head_dim, float theta = 10000.0f, float scaling_factor = 1.0f);
  
  void init(int max_seq_len) override;
  void apply(Tensor& q, Tensor& k, int start_pos) const override;
  
  // Accessors
  int head_dim() const { return head_dim_; }
  float theta() const { return theta_; }
  float scaling_factor() const { return scaling_factor_; }
  int max_cached_positions() const { return max_cached_positions_; }

private:
  int head_dim_;
  float theta_;
  float scaling_factor_;
  int max_cached_positions_ = 0;
  
  // Precomputed cos and sin values for each position
  std::vector<float> cos_cached_;
  std::vector<float> sin_cached_;
};

} // namespace layers

#endif // LAYERS_POSITION_EMBEDDING_H

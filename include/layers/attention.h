#ifndef LAYERS_ATTENTION_H
#define LAYERS_ATTENTION_H

#include "core/tensor.h"
#include "layers/normalization.h"
#include "layers/position_embedding.h"
#include <memory>
#include <vector>

namespace layers {

/// KV Cache for efficient autoregressive generation
struct KVCache {
  std::vector<Tensor> key_cache;    // [num_layers] of [max_seq, num_kv_heads, head_dim]
  std::vector<Tensor> value_cache;  // [num_layers] of [max_seq, num_kv_heads, head_dim]
  int current_length = 0;

  /// Initialize cache for given model dimensions
  void init(int num_layers, int max_seq_len, int num_kv_heads, int head_dim);
  
  /// Clear all cached values
  void clear();
};

/// Configuration for attention layer
struct AttentionConfig {
  int hidden_size = 2048;
  int num_attention_heads = 32;
  int num_key_value_heads = 32;    // Set < num_attention_heads for GQA
  int head_dim = 0;                // 0 = auto-compute from hidden_size / num_heads
  bool use_qkv_bias = false;       // Whether Q/K/V projections have bias
  bool use_o_bias = false;         // Whether output projection has bias
  bool use_qk_norm = false;        // Whether to apply RMSNorm to Q/K (Qwen3)
  float rms_norm_eps = 1e-5f;      // Epsilon for Q/K normalization
  
  int get_head_dim() const {
    return head_dim > 0 ? head_dim : hidden_size / num_attention_heads;
  }
  
  int num_groups() const {
    return num_attention_heads / num_key_value_heads;
  }
  
  bool uses_gqa() const {
    return num_key_value_heads < num_attention_heads;
  }
};

/// Grouped Query Attention (GQA) layer
/// Supports Multi-Head Attention (MHA), Multi-Query Attention (MQA), 
/// and Grouped Query Attention (GQA) through configuration
class GroupedQueryAttention {
public:
  explicit GroupedQueryAttention(const AttentionConfig& config);
  
  /// Forward pass without KV cache
  /// @param x Input tensor [seq_len, hidden_size]
  /// @param q_weight Query projection weight [num_heads * head_dim, hidden_size]
  /// @param k_weight Key projection weight [num_kv_heads * head_dim, hidden_size]
  /// @param v_weight Value projection weight [num_kv_heads * head_dim, hidden_size]
  /// @param o_weight Output projection weight [hidden_size, num_heads * head_dim]
  /// @param pos_embedding Position embedding to apply
  /// @param start_pos Starting position in sequence
  /// @param q_bias Optional query bias
  /// @param k_bias Optional key bias
  /// @param v_bias Optional value bias
  /// @param q_norm_weight Optional Q normalization weight (Qwen3)
  /// @param k_norm_weight Optional K normalization weight (Qwen3)
  /// @return Output tensor [seq_len, hidden_size]
  Tensor forward(
    const Tensor& x,
    const Tensor& q_weight, const Tensor& k_weight,
    const Tensor& v_weight, const Tensor& o_weight,
    const PositionEmbedding& pos_embedding,
    int start_pos,
    const Tensor* q_bias = nullptr,
    const Tensor* k_bias = nullptr,
    const Tensor* v_bias = nullptr,
    const Tensor* q_norm_weight = nullptr,
    const Tensor* k_norm_weight = nullptr
  );
  
  /// Forward pass with KV cache for efficient generation
  Tensor forward_with_cache(
    const Tensor& x,
    const Tensor& q_weight, const Tensor& k_weight,
    const Tensor& v_weight, const Tensor& o_weight,
    const PositionEmbedding& pos_embedding,
    KVCache& cache,
    int layer_idx,
    const Tensor* q_bias = nullptr,
    const Tensor* k_bias = nullptr,
    const Tensor* v_bias = nullptr,
    const Tensor* q_norm_weight = nullptr,
    const Tensor* k_norm_weight = nullptr
  );
  
  const AttentionConfig& config() const { return config_; }

private:
  AttentionConfig config_;
  
  // Linear projection helper
  Tensor linear(const Tensor& x, const Tensor& weight);
  Tensor linear_with_bias(const Tensor& x, const Tensor& weight, const Tensor& bias);
  
  // Q/K normalization helper (Qwen3)
  void apply_qk_norm(Tensor& q, Tensor& k, 
                     const Tensor* q_norm_weight, 
                     const Tensor* k_norm_weight);
};

} // namespace layers

#endif // LAYERS_ATTENTION_H

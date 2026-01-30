#include "layers/position_embedding.h"
#include <cmath>

namespace layers {

// ============================================================================
// RoPE Implementation
// ============================================================================

RoPE::RoPE(int head_dim, float theta, float scaling_factor)
    : head_dim_(head_dim), theta_(theta), scaling_factor_(scaling_factor) {}

void RoPE::init(int max_seq_len) {
  // Precompute inverse frequencies
  std::vector<float> inv_freq(head_dim_ / 2);
  for (int i = 0; i < head_dim_ / 2; ++i) {
    inv_freq[i] = 1.0f / std::pow(theta_, static_cast<float>(2 * i) / head_dim_);
  }

  // Apply scaling if configured
  if (scaling_factor_ != 1.0f) {
    for (auto& freq : inv_freq) {
      freq /= scaling_factor_;
    }
  }

  // Compute cos and sin for all positions
  cos_cached_.resize(max_seq_len * head_dim_ / 2);
  sin_cached_.resize(max_seq_len * head_dim_ / 2);

  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int i = 0; i < head_dim_ / 2; ++i) {
      float angle = pos * inv_freq[i];
      cos_cached_[pos * (head_dim_ / 2) + i] = std::cos(angle);
      sin_cached_[pos * (head_dim_ / 2) + i] = std::sin(angle);
    }
  }

  max_cached_positions_ = max_seq_len;
}

void RoPE::apply(Tensor& q, Tensor& k, int start_pos) const {
  int seq_len = q.shape[0];
  int num_heads_q = q.shape[1];
  int num_heads_k = k.shape[1];
  int half_dim = head_dim_ / 2;

  // Optimized RoPE with pointer-based access for better cache locality
  for (int pos = 0; pos < seq_len; ++pos) {
    int abs_pos = start_pos + pos;
    const float* cos_ptr = cos_cached_.data() + abs_pos * half_dim;
    const float* sin_ptr = sin_cached_.data() + abs_pos * half_dim;

    // Apply RoPE to query heads
    for (int h = 0; h < num_heads_q; ++h) {
      float* q_ptr = q.data.data() + (pos * num_heads_q + h) * head_dim_;
      
      for (int i = 0; i < half_dim; ++i) {
        float q1 = q_ptr[i];
        float q2 = q_ptr[i + half_dim];
        float cos_val = cos_ptr[i];
        float sin_val = sin_ptr[i];
        q_ptr[i] = q1 * cos_val - q2 * sin_val;
        q_ptr[i + half_dim] = q1 * sin_val + q2 * cos_val;
      }
    }

    // Apply RoPE to key heads
    for (int h = 0; h < num_heads_k; ++h) {
      float* k_ptr = k.data.data() + (pos * num_heads_k + h) * head_dim_;
      
      for (int i = 0; i < half_dim; ++i) {
        float k1 = k_ptr[i];
        float k2 = k_ptr[i + half_dim];
        float cos_val = cos_ptr[i];
        float sin_val = sin_ptr[i];
        k_ptr[i] = k1 * cos_val - k2 * sin_val;
        k_ptr[i + half_dim] = k1 * sin_val + k2 * cos_val;
      }
    }
  }
}

} // namespace layers

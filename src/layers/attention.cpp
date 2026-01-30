#include "layers/attention.h"
#include "core/ops.h"
#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

namespace layers {

// ============================================================================
// KV Cache Implementation
// ============================================================================

void KVCache::init(int num_layers, int max_seq_len, int num_kv_heads, int head_dim) {
  key_cache.clear();
  value_cache.clear();

  for (int l = 0; l < num_layers; ++l) {
    key_cache.push_back(Tensor({max_seq_len, num_kv_heads, head_dim}));
    value_cache.push_back(Tensor({max_seq_len, num_kv_heads, head_dim}));
  }
  current_length = 0;
}

void KVCache::clear() {
  for (auto& k : key_cache) {
    std::fill(k.data.begin(), k.data.end(), 0.0f);
  }
  for (auto& v : value_cache) {
    std::fill(v.data.begin(), v.data.end(), 0.0f);
  }
  current_length = 0;
}

// ============================================================================
// GroupedQueryAttention Implementation
// ============================================================================

GroupedQueryAttention::GroupedQueryAttention(const AttentionConfig& config)
    : config_(config) {}

Tensor GroupedQueryAttention::linear(const Tensor& x, const Tensor& weight) {
  int in_features = weight.shape[1];
  int out_features = weight.shape[0];
  int batch = x.data.size() / in_features;

  Tensor output = ops::matmul_ex(x, weight, false, true);
  if (batch == 1) {
    output.shape = {out_features};
  }

  if (x.shape.size() == 2 && x.shape[0] > 1) {
    output.shape = {x.shape[0], out_features};
  }

  return output;
}

Tensor GroupedQueryAttention::linear_with_bias(const Tensor& x, const Tensor& weight, 
                                                const Tensor& bias) {
  Tensor output = linear(x, weight);
  
  int out_features = weight.shape[0];
  int batch = output.data.size() / out_features;
  
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < out_features; ++j) {
      output.data[i * out_features + j] += bias.data[j];
    }
  }
  
  return output;
}

void GroupedQueryAttention::apply_qk_norm(Tensor& q, Tensor& k,
                                          const Tensor* q_norm_weight,
                                          const Tensor* k_norm_weight) {
  if (!q_norm_weight || !k_norm_weight) {
    return;
  }
  
  int seq_len = q.shape[0];
  int num_heads_q = q.shape[1];
  int num_heads_k = k.shape[1];
  int head_dim = config_.get_head_dim();
  float eps = config_.rms_norm_eps;
  
#if defined(__APPLE__)
  // Vectorized Q/K normalization using Accelerate vDSP
  
  // Apply RMSNorm to each query head
  for (int pos = 0; pos < seq_len; ++pos) {
    for (int h = 0; h < num_heads_q; ++h) {
      float* head_ptr = q.data.data() + (pos * num_heads_q + h) * head_dim;
      
      float sum_sq;
      vDSP_dotpr(head_ptr, 1, head_ptr, 1, &sum_sq, head_dim);
      
      float rms = std::sqrt(sum_sq / head_dim + eps);
      float scale = 1.0f / rms;
      
      vDSP_vsmul(head_ptr, 1, &scale, head_ptr, 1, head_dim);
      vDSP_vmul(head_ptr, 1, q_norm_weight->data.data(), 1, head_ptr, 1, head_dim);
    }
  }
  
  // Apply RMSNorm to each key head
  for (int pos = 0; pos < seq_len; ++pos) {
    for (int h = 0; h < num_heads_k; ++h) {
      float* head_ptr = k.data.data() + (pos * num_heads_k + h) * head_dim;
      
      float sum_sq;
      vDSP_dotpr(head_ptr, 1, head_ptr, 1, &sum_sq, head_dim);
      
      float rms = std::sqrt(sum_sq / head_dim + eps);
      float scale = 1.0f / rms;
      
      vDSP_vsmul(head_ptr, 1, &scale, head_ptr, 1, head_dim);
      vDSP_vmul(head_ptr, 1, k_norm_weight->data.data(), 1, head_ptr, 1, head_dim);
    }
  }
#else
  // Fallback scalar implementation
  for (int pos = 0; pos < seq_len; ++pos) {
    for (int h = 0; h < num_heads_q; ++h) {
      float sum_sq = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        int idx = (pos * num_heads_q + h) * head_dim + d;
        sum_sq += q.data[idx] * q.data[idx];
      }
      float rms = std::sqrt(sum_sq / head_dim + eps);
      float scale = 1.0f / rms;
      
      for (int d = 0; d < head_dim; ++d) {
        int idx = (pos * num_heads_q + h) * head_dim + d;
        q.data[idx] = q.data[idx] * scale * q_norm_weight->data[d];
      }
    }
  }
  
  for (int pos = 0; pos < seq_len; ++pos) {
    for (int h = 0; h < num_heads_k; ++h) {
      float sum_sq = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        int idx = (pos * num_heads_k + h) * head_dim + d;
        sum_sq += k.data[idx] * k.data[idx];
      }
      float rms = std::sqrt(sum_sq / head_dim + eps);
      float scale = 1.0f / rms;
      
      for (int d = 0; d < head_dim; ++d) {
        int idx = (pos * num_heads_k + h) * head_dim + d;
        k.data[idx] = k.data[idx] * scale * k_norm_weight->data[d];
      }
    }
  }
#endif
}

Tensor GroupedQueryAttention::forward(
    const Tensor& x,
    const Tensor& q_weight, const Tensor& k_weight,
    const Tensor& v_weight, const Tensor& o_weight,
    const PositionEmbedding& pos_embedding,
    int start_pos,
    const Tensor* q_bias,
    const Tensor* k_bias,
    const Tensor* v_bias,
    const Tensor* q_norm_weight,
    const Tensor* k_norm_weight) {
  
  int seq_len = x.shape[0];
  int num_heads = config_.num_attention_heads;
  int num_kv_heads = config_.num_key_value_heads;
  int head_dim = config_.get_head_dim();
  int num_groups = config_.num_groups();
  int hidden_size = config_.hidden_size;

  // Project Q, K, V
  Tensor q = (config_.use_qkv_bias && q_bias) ? linear_with_bias(x, q_weight, *q_bias) 
                                              : linear(x, q_weight);
  Tensor k = (config_.use_qkv_bias && k_bias) ? linear_with_bias(x, k_weight, *k_bias) 
                                              : linear(x, k_weight);
  Tensor v = (config_.use_qkv_bias && v_bias) ? linear_with_bias(x, v_weight, *v_bias) 
                                              : linear(x, v_weight);

  // Reshape for attention
  q.shape = {seq_len, num_heads, head_dim};
  k.shape = {seq_len, num_kv_heads, head_dim};
  v.shape = {seq_len, num_kv_heads, head_dim};

  // Apply Q/K normalization if configured (Qwen3)
  if (config_.use_qk_norm) {
    apply_qk_norm(q, k, q_norm_weight, k_norm_weight);
  }

  // Apply RoPE
  Tensor q_mutable = q;
  Tensor k_mutable = k;
  pos_embedding.apply(q_mutable, k_mutable, start_pos);
  q = q_mutable;
  k = k_mutable;

  // Compute attention
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  int attn_output_dim = num_heads * head_dim;
  Tensor attn_output({seq_len, attn_output_dim});
  std::fill(attn_output.data.begin(), attn_output.data.end(), 0.0f);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int h = 0; h < num_heads; ++h) {
    int kv_head = h / num_groups;

    std::vector<float> scores(seq_len * seq_len);

    for (int i = 0; i < seq_len; ++i) {
      const float* q_ptr = q.data.data() + (i * num_heads + h) * head_dim;
      
      for (int j = 0; j < seq_len; ++j) {
        // Causal mask: skip future positions
        if (j > i + start_pos) {
          scores[i * seq_len + j] = -1e10f;
          continue;
        }
        
        const float* k_ptr = k.data.data() + (j * num_kv_heads + kv_head) * head_dim;
        
#if defined(__APPLE__)
        float score;
        vDSP_dotpr(q_ptr, 1, k_ptr, 1, &score, head_dim);
        scores[i * seq_len + j] = score * scale;
#else
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          score += q_ptr[d] * k_ptr[d];
        }
        scores[i * seq_len + j] = score * scale;
#endif
      }
    }

    // Softmax per row
    for (int i = 0; i < seq_len; ++i) {
      float* row = scores.data() + i * seq_len;
      int valid_len = i + 1;
      
#if defined(__APPLE__)
      float max_score;
      vDSP_maxv(row, 1, &max_score, valid_len);
      
      float neg_max = -max_score;
      vDSP_vsadd(row, 1, &neg_max, row, 1, valid_len);
      
      int n = valid_len;
      vvexpf(row, row, &n);
      
      float sum;
      vDSP_sve(row, 1, &sum, valid_len);
      
      float inv_sum = 1.0f / (sum + 1e-10f);
      vDSP_vsmul(row, 1, &inv_sum, row, 1, valid_len);
      
      for (int j = valid_len; j < seq_len; ++j) {
        row[j] = 0.0f;
      }
#else
      float max_score = -1e10f;
      for (int j = 0; j <= i; ++j) {
        max_score = std::max(max_score, scores[i * seq_len + j]);
      }

      float sum = 0.0f;
      for (int j = 0; j < seq_len; ++j) {
        if (j <= i) {
          scores[i * seq_len + j] = std::exp(scores[i * seq_len + j] - max_score);
          sum += scores[i * seq_len + j];
        } else {
          scores[i * seq_len + j] = 0.0f;
        }
      }

      for (int j = 0; j < seq_len; ++j) {
        scores[i * seq_len + j] /= (sum + 1e-10f);
      }
#endif
    }

    // Apply attention weights to values
    for (int i = 0; i < seq_len; ++i) {
      for (int d = 0; d < head_dim; ++d) {
        float sum = 0.0f;
        for (int j = 0; j <= i; ++j) {
          int v_idx = (j * num_kv_heads + kv_head) * head_dim + d;
          sum += scores[i * seq_len + j] * v.data[v_idx];
        }
        attn_output.data[i * attn_output_dim + h * head_dim + d] = sum;
      }
    }
  }

  // Output projection
  Tensor output = linear(attn_output, o_weight);
  output.shape = {seq_len, hidden_size};

  return output;
}

Tensor GroupedQueryAttention::forward_with_cache(
    const Tensor& x,
    const Tensor& q_weight, const Tensor& k_weight,
    const Tensor& v_weight, const Tensor& o_weight,
    const PositionEmbedding& pos_embedding,
    KVCache& cache,
    int layer_idx,
    const Tensor* q_bias,
    const Tensor* k_bias,
    const Tensor* v_bias,
    const Tensor* q_norm_weight,
    const Tensor* k_norm_weight) {
  
  int seq_len = x.shape[0];
  int num_heads = config_.num_attention_heads;
  int num_kv_heads = config_.num_key_value_heads;
  int head_dim = config_.get_head_dim();
  int num_groups = config_.num_groups();
  int hidden_size = config_.hidden_size;
  int start_pos = cache.current_length;

  // Project Q, K, V
  Tensor q = (config_.use_qkv_bias && q_bias) ? linear_with_bias(x, q_weight, *q_bias) 
                                              : linear(x, q_weight);
  Tensor k = (config_.use_qkv_bias && k_bias) ? linear_with_bias(x, k_weight, *k_bias) 
                                              : linear(x, k_weight);
  Tensor v = (config_.use_qkv_bias && v_bias) ? linear_with_bias(x, v_weight, *v_bias) 
                                              : linear(x, v_weight);

  q.shape = {seq_len, num_heads, head_dim};
  k.shape = {seq_len, num_kv_heads, head_dim};
  v.shape = {seq_len, num_kv_heads, head_dim};

  // Apply Q/K normalization if configured (Qwen3)
  if (config_.use_qk_norm) {
    apply_qk_norm(q, k, q_norm_weight, k_norm_weight);
  }

  // Apply RoPE
  Tensor q_mutable = q;
  Tensor k_mutable = k;
  pos_embedding.apply(q_mutable, k_mutable, start_pos);
  q = q_mutable;
  k = k_mutable;

  // Store in cache
  Tensor& k_cache = cache.key_cache[layer_idx];
  Tensor& v_cache = cache.value_cache[layer_idx];
  
  int kv_stride = num_kv_heads * head_dim;
  for (int pos = 0; pos < seq_len; ++pos) {
    int cache_pos = start_pos + pos;
    std::memcpy(
      k_cache.data.data() + cache_pos * kv_stride,
      k.data.data() + pos * kv_stride,
      kv_stride * sizeof(float)
    );
    std::memcpy(
      v_cache.data.data() + cache_pos * kv_stride,
      v.data.data() + pos * kv_stride,
      kv_stride * sizeof(float)
    );
  }

  int total_len = start_pos + seq_len;
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  int attn_output_dim = num_heads * head_dim;
  Tensor attn_output({seq_len, attn_output_dim});
  std::fill(attn_output.data.begin(), attn_output.data.end(), 0.0f);

#if defined(__APPLE__)
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int h = 0; h < num_heads; ++h) {
    int kv_head = h / num_groups;

    for (int i = 0; i < seq_len; ++i) {
      int abs_i = start_pos + i;
      int valid_len = abs_i + 1;

      thread_local std::vector<float> scores;
      if (scores.size() < static_cast<size_t>(total_len)) {
        scores.resize(total_len);
      }
      
      const float* q_ptr = q.data.data() + (i * num_heads + h) * head_dim;

      int k_stride = num_kv_heads * head_dim;
      const float* k_head_base = k_cache.data.data() + kv_head * head_dim;
      for (int j = 0; j < valid_len; ++j) {
        const float* k_ptr = k_head_base + j * k_stride;
        float score;
        vDSP_dotpr(q_ptr, 1, k_ptr, 1, &score, head_dim);
        scores[j] = score * scale;
      }
      
      float max_score;
      vDSP_maxv(scores.data(), 1, &max_score, valid_len);
      
      float neg_max = -max_score;
      vDSP_vsadd(scores.data(), 1, &neg_max, scores.data(), 1, valid_len);
      
      int n = valid_len;
      vvexpf(scores.data(), scores.data(), &n);
      
      float sum;
      vDSP_sve(scores.data(), 1, &sum, valid_len);
      
      float inv_sum = 1.0f / (sum + 1e-10f);
      vDSP_vsmul(scores.data(), 1, &inv_sum, scores.data(), 1, valid_len);

      float* out_ptr = attn_output.data.data() + i * attn_output_dim + h * head_dim;
      int v_stride = num_kv_heads * head_dim;
      
      std::memset(out_ptr, 0, head_dim * sizeof(float));
      
      const float* v_head_base = v_cache.data.data() + kv_head * head_dim;
      for (int j = 0; j < valid_len; ++j) {
        const float* v_row = v_head_base + j * v_stride;
        float score_j = scores[j];
        vDSP_vsma(v_row, 1, &score_j, out_ptr, 1, out_ptr, 1, head_dim);
      }
    }
  }
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int h = 0; h < num_heads; ++h) {
    int kv_head = h / num_groups;

    for (int i = 0; i < seq_len; ++i) {
      int abs_i = start_pos + i;
      int valid_len = abs_i + 1;

      std::vector<float> scores(total_len);
      const float* q_ptr = q.data.data() + (i * num_heads + h) * head_dim;

      for (int j = 0; j < total_len; ++j) {
        if (j > abs_i) {
          scores[j] = -1e10f;
        } else {
          const float* k_ptr = k_cache.data.data() + (j * num_kv_heads + kv_head) * head_dim;
          float score = 0.0f;
          for (int d = 0; d < head_dim; ++d) {
            score += q_ptr[d] * k_ptr[d];
          }
          scores[j] = score * scale;
        }
      }

      float max_score = *std::max_element(scores.begin(), scores.begin() + valid_len);
      float sum = 0.0f;
      for (int j = 0; j <= abs_i; ++j) {
        scores[j] = std::exp(scores[j] - max_score);
        sum += scores[j];
      }
      for (int j = 0; j <= abs_i; ++j) {
        scores[j] /= (sum + 1e-10f);
      }

      for (int d = 0; d < head_dim; ++d) {
        float weighted_sum = 0.0f;
        for (int j = 0; j <= abs_i; ++j) {
          int v_idx = (j * num_kv_heads + kv_head) * head_dim + d;
          weighted_sum += scores[j] * v_cache.data[v_idx];
        }
        attn_output.data[i * attn_output_dim + h * head_dim + d] = weighted_sum;
      }
    }
  }
#endif

  Tensor output = linear(attn_output, o_weight);
  output.shape = {seq_len, hidden_size};

  return output;
}

} // namespace layers

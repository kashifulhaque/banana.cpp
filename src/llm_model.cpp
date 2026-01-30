#include "llm_model.h"
#include "ops.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <set>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

// ============================================================================
// KV Cache Implementation
// ============================================================================

void KVCache::init(const LLMConfig& config, int max_seq_len) {
  key_cache.clear();
  value_cache.clear();

  int num_kv_heads = config.num_key_value_heads;
  int head_dim = config.head_dim();

  for (int l = 0; l < config.num_hidden_layers; ++l) {
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
// LLM Model Implementation
// ============================================================================

LLMModel::LLMModel(ModelLoader& loader, const LLMConfig& config)
    : loader_(loader), config_(config) {
  std::cout << "Initializing LLM model: " << config_.model_name << std::endl;
  std::cout << "  Architecture: ";
  switch (config_.architecture) {
    case ModelArchitecture::LLAMA: std::cout << "LLaMA"; break;
    case ModelArchitecture::QWEN: std::cout << "Qwen"; break;
    case ModelArchitecture::MISTRAL: std::cout << "Mistral"; break;
    case ModelArchitecture::PHI: std::cout << "Phi"; break;
    default: std::cout << "Unknown"; break;
  }
  std::cout << std::endl;
  std::cout << "  hidden_size: " << config_.hidden_size << std::endl;
  std::cout << "  num_layers: " << config_.num_hidden_layers << std::endl;
  std::cout << "  num_attention_heads: " << config_.num_attention_heads << std::endl;
  std::cout << "  num_kv_heads: " << config_.num_key_value_heads << std::endl;
  std::cout << "  head_dim: " << config_.head_dim() << std::endl;
  std::cout << "  vocab_size: " << config_.vocab_size << std::endl;

  init_rope_cache(config_.max_position_embeddings);
}

const Tensor* LLMModel::get_weight(const std::string& name, bool warn) {
  const Tensor* weight = loader_.get(name);
  if (!weight && warn) {
    std::cerr << "Warning: Weight '" << name << "' not found!" << std::endl;
  }
  return weight;
}

std::string LLMModel::layer_weight_name(int layer_idx, const std::string& suffix) {
  return config_.layer_prefix + std::to_string(layer_idx) + "." + suffix;
}

bool LLMModel::is_eos_token(int token_id) const {
  if (token_id == config_.eos_token_id) return true;
  for (int eos : config_.eos_token_ids) {
    if (token_id == eos) return true;
  }
  return false;
}

// ============================================================================
// RoPE (Rotary Position Embedding) Implementation
// ============================================================================

void LLMModel::init_rope_cache(int max_seq_len) {
  int head_dim = config_.head_dim();
  float theta = config_.rope_theta;

  // Precompute inverse frequencies
  std::vector<float> inv_freq(head_dim / 2);
  for (int i = 0; i < head_dim / 2; ++i) {
    inv_freq[i] = 1.0f / std::pow(theta, static_cast<float>(2 * i) / head_dim);
  }

  // Apply scaling if configured
  if (config_.rope_scaling_factor != 1.0f) {
    for (auto& freq : inv_freq) {
      freq /= config_.rope_scaling_factor;
    }
  }

  // Compute cos and sin for all positions
  cos_cached_.resize(max_seq_len * head_dim / 2);
  sin_cached_.resize(max_seq_len * head_dim / 2);

  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int i = 0; i < head_dim / 2; ++i) {
      float angle = pos * inv_freq[i];
      cos_cached_[pos * (head_dim / 2) + i] = std::cos(angle);
      sin_cached_[pos * (head_dim / 2) + i] = std::sin(angle);
    }
  }

  max_cached_positions_ = max_seq_len;
}

void LLMModel::apply_rope(Tensor& q, Tensor& k, int start_pos) {
  int seq_len = q.shape[0];
  int num_heads_q = q.shape[1];
  int num_heads_k = k.shape[1];
  int head_dim = q.shape[2];
  int half_dim = head_dim / 2;

  // Optimized RoPE with pointer-based access for better cache locality
  for (int pos = 0; pos < seq_len; ++pos) {
    int abs_pos = start_pos + pos;
    const float* cos_ptr = cos_cached_.data() + abs_pos * half_dim;
    const float* sin_ptr = sin_cached_.data() + abs_pos * half_dim;

    // Apply RoPE to query heads
    for (int h = 0; h < num_heads_q; ++h) {
      float* q_ptr = q.data.data() + (pos * num_heads_q + h) * head_dim;
      
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
      float* k_ptr = k.data.data() + (pos * num_heads_k + h) * head_dim;
      
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

// ============================================================================
// Normalization Implementations
// ============================================================================

Tensor LLMModel::rms_norm(const Tensor& x, const Tensor& weight) {
  int features = x.shape.back();
  int outer_dim = x.data.size() / features;

  Tensor result = x;
  float eps = config_.rms_norm_eps;

#if defined(__APPLE__)
  // Vectorized RMSNorm using Accelerate vDSP
  for (int i = 0; i < outer_dim; ++i) {
    const float* row = x.data.data() + i * features;
    float* out_row = result.data.data() + i * features;
    
    // Compute sum of squares using vectorized dot product
    float sum_sq;
    vDSP_dotpr(row, 1, row, 1, &sum_sq, features);
    
    float rms = std::sqrt(sum_sq / features + eps);
    float scale = 1.0f / rms;
    
    // Scale the input: out = row * scale
    vDSP_vsmul(row, 1, &scale, out_row, 1, features);
    
    // Element-wise multiply with weight: out = out * weight
    vDSP_vmul(out_row, 1, weight.data.data(), 1, out_row, 1, features);
  }
#else
  // Fallback scalar implementation
  for (int i = 0; i < outer_dim; ++i) {
    float sum_sq = 0.0f;
    for (int j = 0; j < features; ++j) {
      float val = x.data[i * features + j];
      sum_sq += val * val;
    }
    float rms = std::sqrt(sum_sq / features + eps);
    float scale = 1.0f / rms;

    for (int j = 0; j < features; ++j) {
      result.data[i * features + j] = x.data[i * features + j] * scale * weight.data[j];
    }
  }
#endif

  return result;
}

Tensor LLMModel::layer_norm(const Tensor& x, const Tensor& weight, const Tensor* bias) {
  int features = x.shape.back();
  int outer_dim = x.data.size() / features;

  Tensor result = x;
  float eps = config_.layer_norm_eps;

#if defined(__APPLE__)
  // Vectorized LayerNorm using Accelerate vDSP
  // Temporary buffer for centered values
  std::vector<float> centered(features);
  
  for (int i = 0; i < outer_dim; ++i) {
    const float* row = x.data.data() + i * features;
    float* out_row = result.data.data() + i * features;
    
    // Calculate mean using vDSP
    float mean;
    vDSP_meanv(row, 1, &mean, features);
    
    // Center the data: centered = row - mean
    float neg_mean = -mean;
    vDSP_vsadd(row, 1, &neg_mean, centered.data(), 1, features);
    
    // Calculate variance using dot product: variance = sum(centered^2) / features
    float variance;
    vDSP_dotpr(centered.data(), 1, centered.data(), 1, &variance, features);
    variance /= features;
    
    // Normalize: out = centered * std_inv
    float std_inv = 1.0f / std::sqrt(variance + eps);
    vDSP_vsmul(centered.data(), 1, &std_inv, out_row, 1, features);
    
    // Scale by weight: out = out * weight
    vDSP_vmul(out_row, 1, weight.data.data(), 1, out_row, 1, features);
    
    // Add bias if present
    if (bias) {
      vDSP_vadd(out_row, 1, bias->data.data(), 1, out_row, 1, features);
    }
  }
#else
  // Fallback scalar implementation
  for (int i = 0; i < outer_dim; ++i) {
    // Calculate mean
    float mean = 0.0f;
    for (int j = 0; j < features; ++j) {
      mean += x.data[i * features + j];
    }
    mean /= features;

    // Calculate variance
    float variance = 0.0f;
    for (int j = 0; j < features; ++j) {
      float diff = x.data[i * features + j] - mean;
      variance += diff * diff;
    }
    variance /= features;

    // Normalize
    float std_inv = 1.0f / std::sqrt(variance + eps);
    for (int j = 0; j < features; ++j) {
      float normalized = (x.data[i * features + j] - mean) * std_inv;
      result.data[i * features + j] = weight.data[j] * normalized;
      if (bias) {
        result.data[i * features + j] += bias->data[j];
      }
    }
  }
#endif

  return result;
}

// ============================================================================
// Linear Projections
// ============================================================================

Tensor LLMModel::linear(const Tensor& x, const Tensor& weight) {
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

Tensor LLMModel::linear_with_bias(const Tensor& x, const Tensor& weight, const Tensor& bias) {
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

// ============================================================================
// Q/K Normalization (Qwen3)
// ============================================================================

void LLMModel::apply_qk_norm(Tensor& q, Tensor& k, int layer_idx) {
  std::string prefix = config_.layer_prefix + std::to_string(layer_idx) + ".self_attn.";
  const Tensor* q_norm = get_weight(prefix + "q_norm.weight", false);
  const Tensor* k_norm = get_weight(prefix + "k_norm.weight", false);
  
  if (!q_norm || !k_norm) {
    return;
  }
  
  int seq_len = q.shape[0];
  int num_heads_q = q.shape[1];
  int num_heads_k = k.shape[1];
  int head_dim = q.shape[2];
  float eps = config_.rms_norm_eps;
  
#if defined(__APPLE__)
  // Vectorized Q/K normalization using Accelerate vDSP
  
  // Apply RMSNorm to each query head
  for (int pos = 0; pos < seq_len; ++pos) {
    for (int h = 0; h < num_heads_q; ++h) {
      float* head_ptr = q.data.data() + (pos * num_heads_q + h) * head_dim;
      
      // Compute sum of squares using vectorized dot product
      float sum_sq;
      vDSP_dotpr(head_ptr, 1, head_ptr, 1, &sum_sq, head_dim);
      
      float rms = std::sqrt(sum_sq / head_dim + eps);
      float scale = 1.0f / rms;
      
      // Scale in place
      vDSP_vsmul(head_ptr, 1, &scale, head_ptr, 1, head_dim);
      
      // Multiply by weight
      vDSP_vmul(head_ptr, 1, q_norm->data.data(), 1, head_ptr, 1, head_dim);
    }
  }
  
  // Apply RMSNorm to each key head
  for (int pos = 0; pos < seq_len; ++pos) {
    for (int h = 0; h < num_heads_k; ++h) {
      float* head_ptr = k.data.data() + (pos * num_heads_k + h) * head_dim;
      
      // Compute sum of squares using vectorized dot product
      float sum_sq;
      vDSP_dotpr(head_ptr, 1, head_ptr, 1, &sum_sq, head_dim);
      
      float rms = std::sqrt(sum_sq / head_dim + eps);
      float scale = 1.0f / rms;
      
      // Scale in place
      vDSP_vsmul(head_ptr, 1, &scale, head_ptr, 1, head_dim);
      
      // Multiply by weight
      vDSP_vmul(head_ptr, 1, k_norm->data.data(), 1, head_ptr, 1, head_dim);
    }
  }
#else
  // Fallback scalar implementation
  
  // Apply RMSNorm to each query head
  for (int pos = 0; pos < seq_len; ++pos) {
    for (int h = 0; h < num_heads_q; ++h) {
      // Compute RMS for this head
      float sum_sq = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        int idx = (pos * num_heads_q + h) * head_dim + d;
        sum_sq += q.data[idx] * q.data[idx];
      }
      float rms = std::sqrt(sum_sq / head_dim + eps);
      float scale = 1.0f / rms;
      
      // Apply normalization with weight
      for (int d = 0; d < head_dim; ++d) {
        int idx = (pos * num_heads_q + h) * head_dim + d;
        q.data[idx] = q.data[idx] * scale * q_norm->data[d];
      }
    }
  }
  
  // Apply RMSNorm to each key head
  for (int pos = 0; pos < seq_len; ++pos) {
    for (int h = 0; h < num_heads_k; ++h) {
      // Compute RMS for this head
      float sum_sq = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        int idx = (pos * num_heads_k + h) * head_dim + d;
        sum_sq += k.data[idx] * k.data[idx];
      }
      float rms = std::sqrt(sum_sq / head_dim + eps);
      float scale = 1.0f / rms;
      
      // Apply normalization with weight
      for (int d = 0; d < head_dim; ++d) {
        int idx = (pos * num_heads_k + h) * head_dim + d;
        k.data[idx] = k.data[idx] * scale * k_norm->data[d];
      }
    }
  }
#endif
}

// ============================================================================
// Embedding Layer
// ============================================================================

Tensor LLMModel::embedding(const std::vector<int>& input_ids) {
  const Tensor* embed_tokens = get_weight(config_.embed_tokens_pattern);

  if (!embed_tokens) {
    std::cerr << "Failed to load embedding weights!" << std::endl;
    return Tensor();
  }

  int seq_len = input_ids.size();
  int hidden_size = config_.hidden_size;

  Tensor output({seq_len, hidden_size});

  for (int pos = 0; pos < seq_len; ++pos) {
    int token_id = input_ids[pos];
    for (int i = 0; i < hidden_size; ++i) {
      output.data[pos * hidden_size + i] = embed_tokens->data[token_id * hidden_size + i];
    }
  }

  return output;
}

// ============================================================================
// Attention with GQA and RoPE
// ============================================================================

Tensor LLMModel::attention(const Tensor& x, int layer_idx, int start_pos) {
  const Tensor* q_proj = get_weight(layer_weight_name(layer_idx, config_.q_proj));
  const Tensor* k_proj = get_weight(layer_weight_name(layer_idx, config_.k_proj));
  const Tensor* v_proj = get_weight(layer_weight_name(layer_idx, config_.v_proj));
  const Tensor* o_proj = get_weight(layer_weight_name(layer_idx, config_.o_proj));

  if (!q_proj || !k_proj || !v_proj || !o_proj) {
    return x;
  }

  // Optional biases (for Qwen-style models)
  const Tensor* q_bias = nullptr;
  const Tensor* k_bias = nullptr;
  const Tensor* v_bias = nullptr;
  
  if (config_.use_qkv_bias) {
    std::string prefix = config_.layer_prefix + std::to_string(layer_idx) + ".self_attn.";
    q_bias = get_weight(prefix + "q_proj.bias", false);
    k_bias = get_weight(prefix + "k_proj.bias", false);
    v_bias = get_weight(prefix + "v_proj.bias", false);
  }

  int seq_len = x.shape[0];
  int hidden_size = config_.hidden_size;
  int num_heads = config_.num_attention_heads;
  int num_kv_heads = config_.num_key_value_heads;
  int head_dim = config_.head_dim();
  int num_groups = num_heads / num_kv_heads;

  // Project Q, K, V
  Tensor q = config_.use_qkv_bias && q_bias ? linear_with_bias(x, *q_proj, *q_bias) : linear(x, *q_proj);
  Tensor k = config_.use_qkv_bias && k_bias ? linear_with_bias(x, *k_proj, *k_bias) : linear(x, *k_proj);
  Tensor v = config_.use_qkv_bias && v_bias ? linear_with_bias(x, *v_proj, *v_bias) : linear(x, *v_proj);

  // Reshape for RoPE
  q.shape = {seq_len, num_heads, head_dim};
  k.shape = {seq_len, num_kv_heads, head_dim};
  v.shape = {seq_len, num_kv_heads, head_dim};

  // Apply Q/K normalization (Qwen3)
  if (config_.use_qk_norm) {
    apply_qk_norm(q, k, layer_idx);
  }

  // Apply RoPE
  apply_rope(q, k, start_pos);

  // Compute attention
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  int attn_output_dim = num_heads * head_dim;  // This may differ from hidden_size
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
        // Vectorized dot product for Q*K
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
      int valid_len = i + 1;  // Causal: only positions 0..i are valid
      
#if defined(__APPLE__)
      // Find max using vDSP
      float max_score;
      vDSP_maxv(row, 1, &max_score, valid_len);
      
      // Subtract max and exponentiate
      float neg_max = -max_score;
      vDSP_vsadd(row, 1, &neg_max, row, 1, valid_len);
      
      // Vectorized exp using vForce
      int n = valid_len;
      vvexpf(row, row, &n);
      
      // Sum for normalization
      float sum;
      vDSP_sve(row, 1, &sum, valid_len);
      
      // Divide by sum
      float inv_sum = 1.0f / (sum + 1e-10f);
      vDSP_vsmul(row, 1, &inv_sum, row, 1, valid_len);
      
      // Zero out future positions (already -1e10 but exp made them tiny, be safe)
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
  Tensor output = linear(attn_output, *o_proj);
  output.shape = {seq_len, hidden_size};

  return output;
}

Tensor LLMModel::attention_with_cache(const Tensor& x, int layer_idx, KVCache& cache) {
  const Tensor* q_proj = get_weight(layer_weight_name(layer_idx, config_.q_proj));
  const Tensor* k_proj = get_weight(layer_weight_name(layer_idx, config_.k_proj));
  const Tensor* v_proj = get_weight(layer_weight_name(layer_idx, config_.v_proj));
  const Tensor* o_proj = get_weight(layer_weight_name(layer_idx, config_.o_proj));

  if (!q_proj || !k_proj || !v_proj || !o_proj) {
    return x;
  }

  const Tensor* q_bias = nullptr;
  const Tensor* k_bias = nullptr;
  const Tensor* v_bias = nullptr;
  
  if (config_.use_qkv_bias) {
    std::string prefix = config_.layer_prefix + std::to_string(layer_idx) + ".self_attn.";
    q_bias = get_weight(prefix + "q_proj.bias", false);
    k_bias = get_weight(prefix + "k_proj.bias", false);
    v_bias = get_weight(prefix + "v_proj.bias", false);
  }

  int seq_len = x.shape[0];
  int hidden_size = config_.hidden_size;
  int num_heads = config_.num_attention_heads;
  int num_kv_heads = config_.num_key_value_heads;
  int head_dim = config_.head_dim();
  int num_groups = num_heads / num_kv_heads;
  int start_pos = cache.current_length;

  // Project Q, K, V
  Tensor q = config_.use_qkv_bias && q_bias ? linear_with_bias(x, *q_proj, *q_bias) : linear(x, *q_proj);
  Tensor k = config_.use_qkv_bias && k_bias ? linear_with_bias(x, *k_proj, *k_bias) : linear(x, *k_proj);
  Tensor v = config_.use_qkv_bias && v_bias ? linear_with_bias(x, *v_proj, *v_bias) : linear(x, *v_proj);

  q.shape = {seq_len, num_heads, head_dim};
  k.shape = {seq_len, num_kv_heads, head_dim};
  v.shape = {seq_len, num_kv_heads, head_dim};

  // Apply Q/K normalization (Qwen3)
  if (config_.use_qk_norm) {
    apply_qk_norm(q, k, layer_idx);
  }

  apply_rope(q, k, start_pos);

  // Store in cache (optimized with memcpy for contiguous data)
  Tensor& k_cache = cache.key_cache[layer_idx];
  Tensor& v_cache = cache.value_cache[layer_idx];
  
  int kv_stride = num_kv_heads * head_dim;
  for (int pos = 0; pos < seq_len; ++pos) {
    int cache_pos = start_pos + pos;
    // Copy entire KV head row at once (num_kv_heads * head_dim floats)
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

  int attn_output_dim = num_heads * head_dim;  // This may differ from hidden_size
  Tensor attn_output({seq_len, attn_output_dim});
  std::fill(attn_output.data.begin(), attn_output.data.end(), 0.0f);

#if defined(__APPLE__)
  // Highly optimized attention using BLAS operations
  // For decode (seq_len=1), this uses cblas_sgemv for Q*K and score*V
  
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int h = 0; h < num_heads; ++h) {
    int kv_head = h / num_groups;

    for (int i = 0; i < seq_len; ++i) {
      int abs_i = start_pos + i;
      int valid_len = abs_i + 1;

      // Thread-local score buffer - pre-allocate to avoid repeated allocations
      thread_local std::vector<float> scores;
      if (scores.size() < static_cast<size_t>(total_len)) {
        scores.resize(total_len);
      }
      
      const float* q_ptr = q.data.data() + (i * num_heads + h) * head_dim;

      // Compute Q*K scores using vectorized dot product
      // K is strided but vDSP_dotpr handles each row efficiently
      int k_stride = num_kv_heads * head_dim;
      const float* k_head_base = k_cache.data.data() + kv_head * head_dim;
      for (int j = 0; j < valid_len; ++j) {
        const float* k_ptr = k_head_base + j * k_stride;
        float score;
        vDSP_dotpr(q_ptr, 1, k_ptr, 1, &score, head_dim);
        scores[j] = score * scale;
      }
      
      // Zero out future positions (not needed since we only process valid_len)
      
      // Softmax over valid positions using vDSP
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

      // Compute weighted sum of values: out = sum_j(scores[j] * V[j,:])
      // Use vectorized operations: iterate over valid_len, for each j:
      //   out += scores[j] * V[j, kv_head, :]
      
      float* out_ptr = attn_output.data.data() + i * attn_output_dim + h * head_dim;
      int v_stride = num_kv_heads * head_dim;
      
      // Zero output first
      std::memset(out_ptr, 0, head_dim * sizeof(float));
      
      // Accumulate: out += scores[j] * V_row for each j
      const float* v_head_base = v_cache.data.data() + kv_head * head_dim;
      for (int j = 0; j < valid_len; ++j) {
        const float* v_row = v_head_base + j * v_stride;
        float score_j = scores[j];
        // out += score_j * v_row (vDSP_vsma: out = out + scalar * v_row)
        vDSP_vsma(v_row, 1, &score_j, out_ptr, 1, out_ptr, 1, head_dim);
      }
    }
  }
#else
  // Fallback non-Apple implementation
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

  Tensor output = linear(attn_output, *o_proj);
  output.shape = {seq_len, hidden_size};

  return output;
}

// ============================================================================
// MLP (SwiGLU for LLaMA-style)
// ============================================================================

Tensor LLMModel::mlp(const Tensor& x, int layer_idx) {
  const Tensor* gate_proj = get_weight(layer_weight_name(layer_idx, config_.gate_proj));
  const Tensor* up_proj = get_weight(layer_weight_name(layer_idx, config_.up_proj));
  const Tensor* down_proj = get_weight(layer_weight_name(layer_idx, config_.down_proj));

  if (!gate_proj || !up_proj || !down_proj) {
    return x;
  }

  int seq_len = x.shape[0];
  int intermediate_size = config_.intermediate_size;

  Tensor gate = linear(x, *gate_proj);
  Tensor up = linear(x, *up_proj);

  Tensor hidden({seq_len, intermediate_size});
  
#if defined(__APPLE__)
  // Vectorized SiLU activation: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
  // Compute for all elements at once using vDSP
  int total = seq_len * intermediate_size;
  
  if (config_.hidden_act == ActivationType::SILU || config_.hidden_act == ActivationType::SWIGLU) {
    // Compute sigmoid(gate): 1 / (1 + exp(-gate))
    // Step 1: negate gate
    std::vector<float> neg_gate(total);
    float neg_one = -1.0f;
    vDSP_vsmul(gate.data.data(), 1, &neg_one, neg_gate.data(), 1, total);
    
    // Step 2: exp(-gate)
    vvexpf(neg_gate.data(), neg_gate.data(), &total);
    
    // Step 3: 1 + exp(-gate)
    float one = 1.0f;
    vDSP_vsadd(neg_gate.data(), 1, &one, neg_gate.data(), 1, total);
    
    // Step 4: gate / (1 + exp(-gate))
    vDSP_vdiv(neg_gate.data(), 1, gate.data.data(), 1, hidden.data.data(), 1, total);
    
    // Step 5: silu(gate) * up
    vDSP_vmul(hidden.data.data(), 1, up.data.data(), 1, hidden.data.data(), 1, total);
  } else {
    // Fallback for other activation types
    for (int i = 0; i < seq_len; ++i) {
      for (int j = 0; j < intermediate_size; ++j) {
        float g = gate.data[i * intermediate_size + j];
        float u = up.data[i * intermediate_size + j];
        
        float activated;
        switch (config_.hidden_act) {
          case ActivationType::GELU:
          case ActivationType::GELU_NEW:
            activated = 0.5f * g * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (g + 0.044715f * g * g * g)));
            break;
          case ActivationType::RELU:
            activated = std::max(0.0f, g);
            break;
          default:
            activated = g / (1.0f + std::exp(-g));
            break;
        }
        hidden.data[i * intermediate_size + j] = activated * u;
      }
    }
  }
#else
  // Scalar implementation
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < intermediate_size; ++j) {
      float g = gate.data[i * intermediate_size + j];
      float u = up.data[i * intermediate_size + j];
      
      float activated;
      switch (config_.hidden_act) {
        case ActivationType::SILU:
        case ActivationType::SWIGLU:
          activated = g / (1.0f + std::exp(-g));
          break;
        case ActivationType::GELU:
        case ActivationType::GELU_NEW:
          activated = 0.5f * g * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (g + 0.044715f * g * g * g)));
          break;
        case ActivationType::RELU:
          activated = std::max(0.0f, g);
          break;
        default:
          activated = g / (1.0f + std::exp(-g));
          break;
      }
      hidden.data[i * intermediate_size + j] = activated * u;
    }
  }
#endif

  Tensor output = linear(hidden, *down_proj);
  output.shape = {seq_len, config_.hidden_size};

  return output;
}

// ============================================================================
// Transformer Block
// ============================================================================

Tensor LLMModel::transformer_block(const Tensor& x, int layer_idx, int start_pos) {
  const Tensor* input_ln = get_weight(layer_weight_name(layer_idx, config_.input_layernorm));
  const Tensor* post_attn_ln = get_weight(layer_weight_name(layer_idx, config_.post_attention_layernorm));

  if (!input_ln || !post_attn_ln) {
    return x;
  }

  // Pre-norm attention
  Tensor normed = (config_.norm_type == NormType::RMS_NORM) 
                  ? rms_norm(x, *input_ln) 
                  : layer_norm(x, *input_ln);
  Tensor attn_output = attention(normed, layer_idx, start_pos);

  // Residual
  Tensor h({x.shape[0], x.shape[1]});
  for (size_t i = 0; i < x.data.size(); ++i) {
    h.data[i] = x.data[i] + attn_output.data[i];
  }

  // Pre-norm MLP
  Tensor normed2 = (config_.norm_type == NormType::RMS_NORM) 
                   ? rms_norm(h, *post_attn_ln) 
                   : layer_norm(h, *post_attn_ln);
  Tensor mlp_output = mlp(normed2, layer_idx);

  // Residual
  Tensor output({x.shape[0], x.shape[1]});
  for (size_t i = 0; i < h.data.size(); ++i) {
    output.data[i] = h.data[i] + mlp_output.data[i];
  }

  return output;
}

Tensor LLMModel::transformer_block_with_cache(const Tensor& x, int layer_idx, KVCache& cache) {
  const Tensor* input_ln = get_weight(layer_weight_name(layer_idx, config_.input_layernorm));
  const Tensor* post_attn_ln = get_weight(layer_weight_name(layer_idx, config_.post_attention_layernorm));

  if (!input_ln || !post_attn_ln) {
    return x;
  }

  Tensor normed = (config_.norm_type == NormType::RMS_NORM) 
                  ? rms_norm(x, *input_ln) 
                  : layer_norm(x, *input_ln);
  Tensor attn_output = attention_with_cache(normed, layer_idx, cache);

  // Residual connection: h = x + attn_output
  Tensor h({x.shape[0], x.shape[1]});
#if defined(__APPLE__)
  vDSP_vadd(x.data.data(), 1, attn_output.data.data(), 1, h.data.data(), 1, x.data.size());
#else
  for (size_t i = 0; i < x.data.size(); ++i) {
    h.data[i] = x.data[i] + attn_output.data[i];
  }
#endif

  Tensor normed2 = (config_.norm_type == NormType::RMS_NORM) 
                   ? rms_norm(h, *post_attn_ln) 
                   : layer_norm(h, *post_attn_ln);
  Tensor mlp_output = mlp(normed2, layer_idx);

  // Residual connection: output = h + mlp_output
  Tensor output({x.shape[0], x.shape[1]});
#if defined(__APPLE__)
  vDSP_vadd(h.data.data(), 1, mlp_output.data.data(), 1, output.data.data(), 1, h.data.size());
#else
  for (size_t i = 0; i < h.data.size(); ++i) {
    output.data[i] = h.data[i] + mlp_output.data[i];
  }
#endif

  return output;
}

// ============================================================================
// Forward Pass
// ============================================================================

Tensor LLMModel::forward(const std::vector<int>& input_ids) {
  Tensor hidden_states = embedding(input_ids);

  for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
    hidden_states = transformer_block(hidden_states, layer, 0);
  }

  // Final norm
  const Tensor* final_norm = get_weight(config_.norm_pattern);
  if (final_norm) {
    hidden_states = (config_.norm_type == NormType::RMS_NORM) 
                    ? rms_norm(hidden_states, *final_norm) 
                    : layer_norm(hidden_states, *final_norm);
  }

  // LM head
  const Tensor* lm_head = get_weight(config_.lm_head_pattern, false);
  if (!lm_head && config_.tie_word_embeddings) {
    lm_head = get_weight(config_.embed_tokens_pattern);
  }

  if (!lm_head) {
    std::cerr << "Failed to load LM head weights!" << std::endl;
    return Tensor();
  }

  int seq_len = input_ids.size();
  int hidden_size = config_.hidden_size;
  int vocab_size = config_.vocab_size;

  // Get pointer to last hidden state (no copy needed for single token)
  const float* last_hidden_ptr = hidden_states.data.data() + (seq_len - 1) * hidden_size;

  Tensor logits({vocab_size});

#if defined(__APPLE__)
  // Use BLAS sgemv: logits = lm_head @ last_hidden
  // lm_head is [vocab_size, hidden_size], we want [vocab_size] output
  // y = alpha * A * x + beta * y, where A is row-major [vocab_size x hidden_size]
  cblas_sgemv(
    CblasRowMajor, CblasNoTrans,
    vocab_size, hidden_size,  // M, N
    1.0f,                     // alpha
    lm_head->data.data(), hidden_size,  // A, lda
    last_hidden_ptr, 1,       // x, incx
    0.0f,                     // beta
    logits.data.data(), 1     // y, incy
  );
#else
  // Fallback scalar implementation
  for (int v = 0; v < vocab_size; ++v) {
    float sum = 0.0f;
    for (int h = 0; h < hidden_size; ++h) {
      sum += last_hidden_ptr[h] * lm_head->data[v * hidden_size + h];
    }
    logits.data[v] = sum;
  }
#endif

  return logits;
}

Tensor LLMModel::forward_with_cache(const std::vector<int>& input_ids, KVCache& cache) {
  Tensor hidden_states = embedding(input_ids);

  for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
    hidden_states = transformer_block_with_cache(hidden_states, layer, cache);
  }

  cache.current_length += input_ids.size();

  const Tensor* final_norm = get_weight(config_.norm_pattern);
  if (final_norm) {
    hidden_states = (config_.norm_type == NormType::RMS_NORM) 
                    ? rms_norm(hidden_states, *final_norm) 
                    : layer_norm(hidden_states, *final_norm);
  }

  const Tensor* lm_head = get_weight(config_.lm_head_pattern, false);
  if (!lm_head && config_.tie_word_embeddings) {
    lm_head = get_weight(config_.embed_tokens_pattern);
  }

  if (!lm_head) {
    std::cerr << "Failed to load LM head weights!" << std::endl;
    return Tensor();
  }

  int seq_len = input_ids.size();
  int hidden_size = config_.hidden_size;
  int vocab_size = config_.vocab_size;

  // Get pointer to last hidden state (no copy needed for single token)
  const float* last_hidden_ptr = hidden_states.data.data() + (seq_len - 1) * hidden_size;

  Tensor logits({vocab_size});

#if defined(__APPLE__)
  // Use BLAS sgemv: logits = lm_head @ last_hidden
  // lm_head is [vocab_size, hidden_size], we want [vocab_size] output
  // y = alpha * A * x + beta * y, where A is row-major [vocab_size x hidden_size]
  cblas_sgemv(
    CblasRowMajor, CblasNoTrans,
    vocab_size, hidden_size,  // M, N
    1.0f,                     // alpha
    lm_head->data.data(), hidden_size,  // A, lda
    last_hidden_ptr, 1,       // x, incx
    0.0f,                     // beta
    logits.data.data(), 1     // y, incy
  );
#else
  // Fallback scalar implementation
  for (int v = 0; v < vocab_size; ++v) {
    float sum = 0.0f;
    for (int h = 0; h < hidden_size; ++h) {
      sum += last_hidden_ptr[h] * lm_head->data[v * hidden_size + h];
    }
    logits.data[v] = sum;
  }
#endif

  return logits;
}

// ============================================================================
// Sampling
// ============================================================================

int LLMModel::sample_token(const Tensor& logits, const SamplingConfig& config,
                           const std::vector<int>& generated_tokens) {
  int vocab_size = logits.shape[0];
  std::vector<float> probs = logits.data;

  // Repetition penalty
  if (config.repetition_penalty != 1.0f) {
    for (int token : generated_tokens) {
      if (token >= 0 && token < vocab_size) {
        if (probs[token] > 0) {
          probs[token] /= config.repetition_penalty;
        } else {
          probs[token] *= config.repetition_penalty;
        }
      }
    }
  }

  // Temperature
  if (config.temperature > 0.0f) {
    for (float& p : probs) {
      p /= config.temperature;
    }
  }

  float max_logit = *std::max_element(probs.begin(), probs.end());

  // Top-k
  if (config.top_k > 0 && config.top_k < vocab_size) {
    std::vector<std::pair<float, int>> indexed_probs;
    for (int i = 0; i < vocab_size; ++i) {
      indexed_probs.push_back({probs[i], i});
    }
    std::partial_sort(indexed_probs.begin(), indexed_probs.begin() + config.top_k,
                      indexed_probs.end(), std::greater<std::pair<float, int>>());

    float threshold = indexed_probs[config.top_k - 1].first;
    for (int i = 0; i < vocab_size; ++i) {
      if (probs[i] < threshold) {
        probs[i] = -1e10f;
      }
    }
  }

  // Softmax
  float sum = 0.0f;
  for (int i = 0; i < vocab_size; ++i) {
    probs[i] = std::exp(probs[i] - max_logit);
    sum += probs[i];
  }
  for (float& p : probs) {
    p /= sum;
  }

  // Top-p
  if (config.top_p < 1.0f) {
    std::vector<std::pair<float, int>> indexed_probs;
    for (int i = 0; i < vocab_size; ++i) {
      indexed_probs.push_back({probs[i], i});
    }
    std::sort(indexed_probs.begin(), indexed_probs.end(),
              std::greater<std::pair<float, int>>());

    float cumsum = 0.0f;
    int cutoff_idx = vocab_size;
    for (int i = 0; i < vocab_size; ++i) {
      cumsum += indexed_probs[i].first;
      if (cumsum > config.top_p) {
        cutoff_idx = i + 1;
        break;
      }
    }

    std::set<int> kept_indices;
    for (int i = 0; i < cutoff_idx; ++i) {
      kept_indices.insert(indexed_probs[i].second);
    }

    sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
      if (kept_indices.find(i) == kept_indices.end()) {
        probs[i] = 0.0f;
      }
      sum += probs[i];
    }
    for (float& p : probs) {
      p /= (sum + 1e-10f);
    }
  }

  // Sample
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  if (config.temperature == 0.0f) {
    return std::max_element(probs.begin(), probs.end()) - probs.begin();
  }

  float r = dis(gen);
  float cumsum = 0.0f;
  for (int i = 0; i < vocab_size; ++i) {
    cumsum += probs[i];
    if (r < cumsum) {
      return i;
    }
  }

  return vocab_size - 1;
}

// ============================================================================
// Generation
// ============================================================================

std::vector<int> LLMModel::generate(const std::vector<int>& input_ids,
                                     const SamplingConfig& config,
                                     TokenCallback token_callback) {
  std::vector<int> generated = input_ids;

  KVCache cache;
  cache.init(config_, config_.max_position_embeddings);

  // Prefill
  Tensor logits = forward_with_cache(input_ids, cache);

  // Generate
  for (int i = 0; i < config.max_new_tokens; ++i) {
    int next_token = sample_token(logits, config, generated);
    generated.push_back(next_token);

    // Call the token callback if provided
    if (token_callback) {
      if (!token_callback(next_token)) {
        break;  // Callback returned false, stop generation
      }
    }

    // Check for EOS
    if (is_eos_token(next_token)) {
      break;
    }

    // Check for custom stop tokens
    for (int stop_id : config.stop_token_ids) {
      if (next_token == stop_id) {
        return generated;
      }
    }

    // Forward single token
    std::vector<int> single_token = {next_token};
    logits = forward_with_cache(single_token, cache);
  }

  return generated;
}

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<LLMModel> create_model(ModelLoader& loader, const LLMConfig& config) {
  return std::make_unique<LLMModel>(loader, config);
}

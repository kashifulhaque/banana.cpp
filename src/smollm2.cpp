#include "smollm2.h"
#include "ops.h"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <set>

// ============================================================================
// KV Cache Implementation
// ============================================================================

void KVCache::init(const SmolLM2Config& config, int max_seq_len) {
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
// SmolLM2 Model Implementation
// ============================================================================

SmolLM2Model::SmolLM2Model(ModelLoader& loader) : loader_(loader), config_() {
    std::cout << "Initializing SmolLM2 model..." << std::endl;
    std::cout << "  hidden_size: " << config_.hidden_size << std::endl;
    std::cout << "  num_layers: " << config_.num_hidden_layers << std::endl;
    std::cout << "  num_attention_heads: " << config_.num_attention_heads << std::endl;
    std::cout << "  num_kv_heads: " << config_.num_key_value_heads << std::endl;
    std::cout << "  head_dim: " << config_.head_dim() << std::endl;
    std::cout << "  vocab_size: " << config_.vocab_size << std::endl;
    
    init_rope_cache(config_.max_position_embeddings);
}

SmolLM2Model::SmolLM2Model(ModelLoader& loader, const SmolLM2Config& config) 
    : loader_(loader), config_(config) {
    std::cout << "Initializing SmolLM2 model with custom config..." << std::endl;
    init_rope_cache(config_.max_position_embeddings);
}

const Tensor* SmolLM2Model::get_weight(const std::string& name, bool warn) {
    const Tensor* weight = loader_.get(name);
    if (!weight && warn) {
        std::cerr << "Warning: Weight '" << name << "' not found!" << std::endl;
    }
    return weight;
}

// ============================================================================
// RoPE (Rotary Position Embedding) Implementation
// ============================================================================

void SmolLM2Model::init_rope_cache(int max_seq_len) {
    int head_dim = config_.head_dim();
    float theta = config_.rope_theta;
    
    // Precompute inverse frequencies
    // inv_freq = 1.0 / (theta ^ (2i / head_dim)) for i in [0, head_dim/2)
    std::vector<float> inv_freq(head_dim / 2);
    for (int i = 0; i < head_dim / 2; ++i) {
        inv_freq[i] = 1.0f / std::pow(theta, static_cast<float>(2 * i) / head_dim);
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

void SmolLM2Model::apply_rope(Tensor& q, Tensor& k, int start_pos) {
    // q shape: [seq_len, num_heads, head_dim]
    // k shape: [seq_len, num_kv_heads, head_dim]
    
    int seq_len = q.shape[0];
    int num_heads_q = q.shape[1];
    int num_heads_k = k.shape[1];
    int head_dim = q.shape[2];
    int half_dim = head_dim / 2;
    
    for (int pos = 0; pos < seq_len; ++pos) {
        int abs_pos = start_pos + pos;
        
        // Apply RoPE to query heads
        for (int h = 0; h < num_heads_q; ++h) {
            for (int i = 0; i < half_dim; ++i) {
                float cos_val = cos_cached_[abs_pos * half_dim + i];
                float sin_val = sin_cached_[abs_pos * half_dim + i];
                
                int idx1 = (pos * num_heads_q + h) * head_dim + i;
                int idx2 = (pos * num_heads_q + h) * head_dim + i + half_dim;
                
                float q1 = q.data[idx1];
                float q2 = q.data[idx2];
                
                q.data[idx1] = q1 * cos_val - q2 * sin_val;
                q.data[idx2] = q1 * sin_val + q2 * cos_val;
            }
        }
        
        // Apply RoPE to key heads
        for (int h = 0; h < num_heads_k; ++h) {
            for (int i = 0; i < half_dim; ++i) {
                float cos_val = cos_cached_[abs_pos * half_dim + i];
                float sin_val = sin_cached_[abs_pos * half_dim + i];
                
                int idx1 = (pos * num_heads_k + h) * head_dim + i;
                int idx2 = (pos * num_heads_k + h) * head_dim + i + half_dim;
                
                float k1 = k.data[idx1];
                float k2 = k.data[idx2];
                
                k.data[idx1] = k1 * cos_val - k2 * sin_val;
                k.data[idx2] = k1 * sin_val + k2 * cos_val;
            }
        }
    }
}

// ============================================================================
// RMSNorm Implementation
// ============================================================================

Tensor SmolLM2Model::rms_norm(const Tensor& x, const Tensor& weight) {
    // RMSNorm: x * weight / sqrt(mean(x^2) + eps)
    int features = x.shape.back();
    int outer_dim = x.data.size() / features;
    
    Tensor result = x;
    float eps = config_.rms_norm_eps;
    
    for (int i = 0; i < outer_dim; ++i) {
        // Calculate RMS (root mean square)
        float sum_sq = 0.0f;
        for (int j = 0; j < features; ++j) {
            float val = x.data[i * features + j];
            sum_sq += val * val;
        }
        float rms = std::sqrt(sum_sq / features + eps);
        float scale = 1.0f / rms;
        
        // Normalize and scale by weight
        for (int j = 0; j < features; ++j) {
            result.data[i * features + j] = x.data[i * features + j] * scale * weight.data[j];
        }
    }
    
    return result;
}

// ============================================================================
// Linear Projection
// ============================================================================

Tensor SmolLM2Model::linear(const Tensor& x, const Tensor& weight) {
    // x: [..., in_features]
    // weight: [out_features, in_features]
    // output: [..., out_features]
    
    int in_features = weight.shape[1];
    int out_features = weight.shape[0];
    int batch = x.data.size() / in_features;
    
    Tensor output = ops::matmul_ex(x, weight, false, true);
    if (batch == 1) {
        output.shape = {out_features};
    }
    
    // Reshape output appropriately
    if (x.shape.size() == 2 && x.shape[0] > 1) {
        output.shape = {x.shape[0], out_features};
    }
    
    return output;
}

// ============================================================================
// Embedding Layer
// ============================================================================

Tensor SmolLM2Model::embedding(const std::vector<int>& input_ids) {
    const Tensor* embed_tokens = get_weight("model.embed_tokens.weight");
    
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
// Grouped Query Attention (GQA) with RoPE
// ============================================================================

Tensor SmolLM2Model::attention(const Tensor& x, int layer_idx, int start_pos) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".self_attn.";
    
    const Tensor* q_proj = get_weight(prefix + "q_proj.weight");
    const Tensor* k_proj = get_weight(prefix + "k_proj.weight");
    const Tensor* v_proj = get_weight(prefix + "v_proj.weight");
    const Tensor* o_proj = get_weight(prefix + "o_proj.weight");
    
    if (!q_proj || !k_proj || !v_proj || !o_proj) {
        return x;
    }
    
    int seq_len = x.shape[0];
    int hidden_size = config_.hidden_size;
    int num_heads = config_.num_attention_heads;
    int num_kv_heads = config_.num_key_value_heads;
    int head_dim = config_.head_dim();
    int num_groups = num_heads / num_kv_heads;  // queries per KV head
    
    // Project Q, K, V
    Tensor q = linear(x, *q_proj);  // [seq_len, num_heads * head_dim]
    Tensor k = linear(x, *k_proj);  // [seq_len, num_kv_heads * head_dim]
    Tensor v = linear(x, *v_proj);  // [seq_len, num_kv_heads * head_dim]
    
    // Reshape for RoPE: [seq_len, num_heads, head_dim]
    q.shape = {seq_len, num_heads, head_dim};
    k.shape = {seq_len, num_kv_heads, head_dim};
    v.shape = {seq_len, num_kv_heads, head_dim};
    
    // Apply RoPE
    apply_rope(q, k, start_pos);
    
    // Compute attention scores with GQA
    // Each KV head serves 'num_groups' query heads
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    Tensor attn_output({seq_len, hidden_size});
    std::fill(attn_output.data.begin(), attn_output.data.end(), 0.0f);
    
    // For each query head
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int h = 0; h < num_heads; ++h) {
        int kv_head = h / num_groups;  // which KV head this query head uses
        
        // Compute attention scores for this head
        std::vector<float> scores(seq_len * seq_len);
        
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    int q_idx = (i * num_heads + h) * head_dim + d;
                    int k_idx = (j * num_kv_heads + kv_head) * head_dim + d;
                    score += q.data[q_idx] * k.data[k_idx];
                }
                
                // Apply causal mask
                if (j > i + start_pos) {
                    score = -1e10f;
                }
                
                scores[i * seq_len + j] = score * scale;
            }
        }
        
        // Softmax over scores
        for (int i = 0; i < seq_len; ++i) {
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
        }
        
        // Apply attention to values
        for (int i = 0; i < seq_len; ++i) {
            for (int d = 0; d < head_dim; ++d) {
                float sum = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    int v_idx = (j * num_kv_heads + kv_head) * head_dim + d;
                    sum += scores[i * seq_len + j] * v.data[v_idx];
                }
                attn_output.data[i * hidden_size + h * head_dim + d] = sum;
            }
        }
    }
    
    // Output projection
    Tensor output = linear(attn_output, *o_proj);
    output.shape = {seq_len, hidden_size};
    
    return output;
}

Tensor SmolLM2Model::attention_with_cache(const Tensor& x, int layer_idx, KVCache& cache) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".self_attn.";
    
    const Tensor* q_proj = get_weight(prefix + "q_proj.weight");
    const Tensor* k_proj = get_weight(prefix + "k_proj.weight");
    const Tensor* v_proj = get_weight(prefix + "v_proj.weight");
    const Tensor* o_proj = get_weight(prefix + "o_proj.weight");
    
    if (!q_proj || !k_proj || !v_proj || !o_proj) {
        return x;
    }
    
    int seq_len = x.shape[0];
    int hidden_size = config_.hidden_size;
    int num_heads = config_.num_attention_heads;
    int num_kv_heads = config_.num_key_value_heads;
    int head_dim = config_.head_dim();
    int num_groups = num_heads / num_kv_heads;
    int start_pos = cache.current_length;
    
    // Project Q, K, V for new tokens only
    Tensor q = linear(x, *q_proj);
    Tensor k = linear(x, *k_proj);
    Tensor v = linear(x, *v_proj);
    
    // Reshape
    q.shape = {seq_len, num_heads, head_dim};
    k.shape = {seq_len, num_kv_heads, head_dim};
    v.shape = {seq_len, num_kv_heads, head_dim};
    
    // Apply RoPE
    apply_rope(q, k, start_pos);
    
    // Store K, V in cache
    Tensor& k_cache = cache.key_cache[layer_idx];
    Tensor& v_cache = cache.value_cache[layer_idx];
    
    for (int pos = 0; pos < seq_len; ++pos) {
        int cache_pos = start_pos + pos;
        for (int h = 0; h < num_kv_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                int src_idx = (pos * num_kv_heads + h) * head_dim + d;
                int dst_idx = (cache_pos * num_kv_heads + h) * head_dim + d;
                k_cache.data[dst_idx] = k.data[src_idx];
                v_cache.data[dst_idx] = v.data[src_idx];
            }
        }
    }
    
    int total_len = start_pos + seq_len;
    
    // Compute attention with cached K, V
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    Tensor attn_output({seq_len, hidden_size});
    std::fill(attn_output.data.begin(), attn_output.data.end(), 0.0f);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int h = 0; h < num_heads; ++h) {
        int kv_head = h / num_groups;
        
        for (int i = 0; i < seq_len; ++i) {
            int abs_i = start_pos + i;
            
            // Compute attention scores
            std::vector<float> scores(total_len);
            
            for (int j = 0; j < total_len; ++j) {
                if (j > abs_i) {
                    scores[j] = -1e10f;
                } else {
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        int q_idx = (i * num_heads + h) * head_dim + d;
                        int k_idx = (j * num_kv_heads + kv_head) * head_dim + d;
                        score += q.data[q_idx] * k_cache.data[k_idx];
                    }
                    scores[j] = score * scale;
                }
            }
            
            // Softmax
            float max_score = *std::max_element(scores.begin(), scores.begin() + abs_i + 1);
            float sum = 0.0f;
            for (int j = 0; j <= abs_i; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                sum += scores[j];
            }
            for (int j = 0; j <= abs_i; ++j) {
                scores[j] /= (sum + 1e-10f);
            }
            
            // Apply to values
            for (int d = 0; d < head_dim; ++d) {
                float weighted_sum = 0.0f;
                for (int j = 0; j <= abs_i; ++j) {
                    int v_idx = (j * num_kv_heads + kv_head) * head_dim + d;
                    weighted_sum += scores[j] * v_cache.data[v_idx];
                }
                attn_output.data[i * hidden_size + h * head_dim + d] = weighted_sum;
            }
        }
    }
    
    // Output projection
    Tensor output = linear(attn_output, *o_proj);
    output.shape = {seq_len, hidden_size};
    
    return output;
}

// ============================================================================
// SwiGLU MLP
// ============================================================================

Tensor SmolLM2Model::mlp(const Tensor& x, int layer_idx) {
    // SwiGLU: output = down_proj(silu(gate_proj(x)) * up_proj(x))
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".mlp.";
    
    const Tensor* gate_proj = get_weight(prefix + "gate_proj.weight");
    const Tensor* up_proj = get_weight(prefix + "up_proj.weight");
    const Tensor* down_proj = get_weight(prefix + "down_proj.weight");
    
    if (!gate_proj || !up_proj || !down_proj) {
        return x;
    }
    
    int seq_len = x.shape[0];
    int intermediate_size = config_.intermediate_size;
    
    // gate_proj and up_proj
    Tensor gate = linear(x, *gate_proj);  // [seq_len, intermediate_size]
    Tensor up = linear(x, *up_proj);      // [seq_len, intermediate_size]
    
    // SiLU (Swish) activation: x * sigmoid(x)
    Tensor hidden({seq_len, intermediate_size});
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < intermediate_size; ++j) {
            float g = gate.data[i * intermediate_size + j];
            float u = up.data[i * intermediate_size + j];
            float silu = g / (1.0f + std::exp(-g));  // SiLU = x * sigmoid(x)
            hidden.data[i * intermediate_size + j] = silu * u;
        }
    }
    
    // down_proj
    Tensor output = linear(hidden, *down_proj);
    output.shape = {seq_len, config_.hidden_size};
    
    return output;
}

// ============================================================================
// Transformer Block
// ============================================================================

Tensor SmolLM2Model::transformer_block(const Tensor& x, int layer_idx, int start_pos) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    
    const Tensor* input_layernorm = get_weight(prefix + "input_layernorm.weight");
    const Tensor* post_attention_layernorm = get_weight(prefix + "post_attention_layernorm.weight");
    
    if (!input_layernorm || !post_attention_layernorm) {
        return x;
    }
    
    // Pre-norm attention
    Tensor normed = rms_norm(x, *input_layernorm);
    Tensor attn_output = attention(normed, layer_idx, start_pos);
    
    // Residual connection
    Tensor h({x.shape[0], x.shape[1]});
    for (size_t i = 0; i < x.data.size(); ++i) {
        h.data[i] = x.data[i] + attn_output.data[i];
    }
    
    // Pre-norm MLP
    Tensor normed2 = rms_norm(h, *post_attention_layernorm);
    Tensor mlp_output = mlp(normed2, layer_idx);
    
    // Residual connection
    Tensor output({x.shape[0], x.shape[1]});
    for (size_t i = 0; i < h.data.size(); ++i) {
        output.data[i] = h.data[i] + mlp_output.data[i];
    }
    
    return output;
}

Tensor SmolLM2Model::transformer_block_with_cache(const Tensor& x, int layer_idx, KVCache& cache) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    
    const Tensor* input_layernorm = get_weight(prefix + "input_layernorm.weight");
    const Tensor* post_attention_layernorm = get_weight(prefix + "post_attention_layernorm.weight");
    
    if (!input_layernorm || !post_attention_layernorm) {
        return x;
    }
    
    // Pre-norm attention
    Tensor normed = rms_norm(x, *input_layernorm);
    Tensor attn_output = attention_with_cache(normed, layer_idx, cache);
    
    // Residual connection
    Tensor h({x.shape[0], x.shape[1]});
    for (size_t i = 0; i < x.data.size(); ++i) {
        h.data[i] = x.data[i] + attn_output.data[i];
    }
    
    // Pre-norm MLP
    Tensor normed2 = rms_norm(h, *post_attention_layernorm);
    Tensor mlp_output = mlp(normed2, layer_idx);
    
    // Residual connection
    Tensor output({x.shape[0], x.shape[1]});
    for (size_t i = 0; i < h.data.size(); ++i) {
        output.data[i] = h.data[i] + mlp_output.data[i];
    }
    
    return output;
}

// ============================================================================
// Forward Pass
// ============================================================================

Tensor SmolLM2Model::forward(const std::vector<int>& input_ids) {
    // Embedding
    Tensor hidden_states = embedding(input_ids);
    
    // Pass through all transformer blocks
    for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
        hidden_states = transformer_block(hidden_states, layer, 0);
    }
    
    // Final layer norm
    const Tensor* final_norm = get_weight("model.norm.weight");
    if (final_norm) {
        hidden_states = rms_norm(hidden_states, *final_norm);
    }
    
    // LM head (tied with embedding weights)
    // Note: lm_head.weight may not exist if tie_word_embeddings is true
    const Tensor* lm_head = get_weight("lm_head.weight", false);
    if (!lm_head && config_.tie_word_embeddings) {
        lm_head = get_weight("model.embed_tokens.weight");
    }
    
    if (!lm_head) {
        std::cerr << "Failed to load LM head weights!" << std::endl;
        return Tensor();
    }
    
    // Output logits for the last token
    int seq_len = input_ids.size();
    int hidden_size = config_.hidden_size;
    int vocab_size = config_.vocab_size;
    
    Tensor last_hidden({1, hidden_size});
    for (int i = 0; i < hidden_size; ++i) {
        last_hidden.data[i] = hidden_states.data[(seq_len - 1) * hidden_size + i];
    }
    
    // Compute logits: last_hidden @ lm_head^T
    Tensor logits({vocab_size});
    for (int v = 0; v < vocab_size; ++v) {
        float sum = 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
            sum += last_hidden.data[h] * lm_head->data[v * hidden_size + h];
        }
        logits.data[v] = sum;
    }
    
    return logits;
}

Tensor SmolLM2Model::forward_with_cache(const std::vector<int>& input_ids, KVCache& cache) {
    // Embedding
    Tensor hidden_states = embedding(input_ids);
    
    // Pass through all transformer blocks
    for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
        hidden_states = transformer_block_with_cache(hidden_states, layer, cache);
    }
    
    // Update cache length
    cache.current_length += input_ids.size();
    
    // Final layer norm
    const Tensor* final_norm = get_weight("model.norm.weight");
    if (final_norm) {
        hidden_states = rms_norm(hidden_states, *final_norm);
    }
    
    // LM head (tied with embedding weights)
    // Note: lm_head.weight may not exist if tie_word_embeddings is true
    const Tensor* lm_head = get_weight("lm_head.weight", false);
    if (!lm_head && config_.tie_word_embeddings) {
        lm_head = get_weight("model.embed_tokens.weight");
    }
    
    if (!lm_head) {
        std::cerr << "Failed to load LM head weights!" << std::endl;
        return Tensor();
    }
    
    // Output logits for the last token
    int seq_len = input_ids.size();
    int hidden_size = config_.hidden_size;
    int vocab_size = config_.vocab_size;
    
    Tensor last_hidden({1, hidden_size});
    for (int i = 0; i < hidden_size; ++i) {
        last_hidden.data[i] = hidden_states.data[(seq_len - 1) * hidden_size + i];
    }
    
    Tensor logits({vocab_size});
    for (int v = 0; v < vocab_size; ++v) {
        float sum = 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
            sum += last_hidden.data[h] * lm_head->data[v * hidden_size + h];
        }
        logits.data[v] = sum;
    }
    
    return logits;
}

// ============================================================================
// Sampling
// ============================================================================

int SmolLM2Model::sample_token(const Tensor& logits, const SmolLM2SamplingConfig& config,
                                const std::vector<int>& generated_tokens) {
    int vocab_size = logits.shape[0];
    std::vector<float> probs = logits.data;
    
    // Apply repetition penalty
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
    
    // Apply temperature
    if (config.temperature > 0.0f) {
        for (float& p : probs) {
            p /= config.temperature;
        }
    }
    
    // Find max for stability
    float max_logit = *std::max_element(probs.begin(), probs.end());
    
    // Top-k filtering
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
    
    // Top-p (nucleus) filtering
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
    
    // Sample from distribution
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    if (config.temperature == 0.0f) {
        // Greedy
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
// Text Generation
// ============================================================================

std::vector<int> SmolLM2Model::generate(const std::vector<int>& input_ids,
                                         const SmolLM2SamplingConfig& config) {
    std::vector<int> generated = input_ids;
    
    // Initialize KV cache
    KVCache cache;
    cache.init(config_, config_.max_position_embeddings);
    
    // Prefill: process all input tokens at once
    Tensor logits = forward_with_cache(input_ids, cache);
    
    // Generate new tokens one by one
    for (int i = 0; i < config.max_new_tokens; ++i) {
        // Sample next token
        int next_token = sample_token(logits, config, generated);
        generated.push_back(next_token);
        
        // Check for EOS token (token id 2 for SmolLM2)
        if (next_token == 2) {
            break;
        }
        
        // Forward pass with single token
        std::vector<int> single_token = {next_token};
        logits = forward_with_cache(single_token, cache);
        
        // Print progress
        if ((i + 1) % 10 == 0) {
            std::cout << "Generated " << (i + 1) << " tokens..." << std::endl;
        }
    }
    
    return generated;
}

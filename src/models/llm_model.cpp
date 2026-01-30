#include "models/llm_model.h"
#include "core/ops.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <set>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

namespace models {

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

  // Initialize RoPE
  rope_ = std::make_unique<layers::RoPE>(
    config_.head_dim(), 
    config_.rope_theta,
    config_.rope_scaling_factor
  );
  rope_->init(config_.max_position_embeddings);

  // Initialize attention layer
  layers::AttentionConfig attn_config;
  attn_config.hidden_size = config_.hidden_size;
  attn_config.num_attention_heads = config_.num_attention_heads;
  attn_config.num_key_value_heads = config_.num_key_value_heads;
  attn_config.head_dim = config_.head_dim_override;
  attn_config.use_qkv_bias = config_.use_qkv_bias;
  attn_config.use_o_bias = config_.use_o_bias;
  attn_config.use_qk_norm = config_.use_qk_norm;
  attn_config.rms_norm_eps = config_.rms_norm_eps;
  attention_ = std::make_unique<layers::GroupedQueryAttention>(attn_config);

  // Initialize MLP
  layers::MLPConfig mlp_config;
  mlp_config.hidden_size = config_.hidden_size;
  mlp_config.intermediate_size = config_.intermediate_size;
  mlp_config.activation = (config_.hidden_act == ActivationType::SILU) 
                           ? layers::Activation::SILU 
                           : layers::Activation::GELU;
  mlp_config.use_bias = config_.use_mlp_bias;
  mlp_ = std::make_unique<layers::SwiGLUMLP>(mlp_config);

  // Initialize normalization layers
  rms_norm_ = std::make_unique<layers::RMSNorm>(config_.rms_norm_eps);
  layer_norm_ = std::make_unique<layers::LayerNorm>(config_.layer_norm_eps);
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
// Transformer Block
// ============================================================================

Tensor LLMModel::transformer_block(const Tensor& x, int layer_idx, int start_pos) {
  const Tensor* input_ln = get_weight(layer_weight_name(layer_idx, config_.input_layernorm));
  const Tensor* post_attn_ln = get_weight(layer_weight_name(layer_idx, config_.post_attention_layernorm));

  if (!input_ln || !post_attn_ln) {
    return x;
  }

  // Get attention weights
  const Tensor* q_proj = get_weight(layer_weight_name(layer_idx, config_.q_proj));
  const Tensor* k_proj = get_weight(layer_weight_name(layer_idx, config_.k_proj));
  const Tensor* v_proj = get_weight(layer_weight_name(layer_idx, config_.v_proj));
  const Tensor* o_proj = get_weight(layer_weight_name(layer_idx, config_.o_proj));

  if (!q_proj || !k_proj || !v_proj || !o_proj) {
    return x;
  }

  // Optional biases
  const Tensor* q_bias = nullptr;
  const Tensor* k_bias = nullptr;
  const Tensor* v_bias = nullptr;
  
  if (config_.use_qkv_bias) {
    std::string prefix = config_.layer_prefix + std::to_string(layer_idx) + ".self_attn.";
    q_bias = get_weight(prefix + "q_proj.bias", false);
    k_bias = get_weight(prefix + "k_proj.bias", false);
    v_bias = get_weight(prefix + "v_proj.bias", false);
  }

  // Q/K normalization weights (Qwen3)
  const Tensor* q_norm = nullptr;
  const Tensor* k_norm = nullptr;
  if (config_.use_qk_norm) {
    std::string prefix = config_.layer_prefix + std::to_string(layer_idx) + ".self_attn.";
    q_norm = get_weight(prefix + "q_norm.weight", false);
    k_norm = get_weight(prefix + "k_norm.weight", false);
  }

  // Pre-norm attention
  Tensor normed = (config_.norm_type == NormType::RMS_NORM) 
                  ? rms_norm_->forward(x, *input_ln) 
                  : layer_norm_->forward(x, *input_ln);

  Tensor attn_output = attention_->forward(
    normed, *q_proj, *k_proj, *v_proj, *o_proj,
    *rope_, start_pos,
    q_bias, k_bias, v_bias, q_norm, k_norm
  );

  // Residual
  Tensor h({x.shape[0], x.shape[1]});
  for (size_t i = 0; i < x.data.size(); ++i) {
    h.data[i] = x.data[i] + attn_output.data[i];
  }

  // Get MLP weights
  const Tensor* gate_proj = get_weight(layer_weight_name(layer_idx, config_.gate_proj));
  const Tensor* up_proj = get_weight(layer_weight_name(layer_idx, config_.up_proj));
  const Tensor* down_proj = get_weight(layer_weight_name(layer_idx, config_.down_proj));

  if (!gate_proj || !up_proj || !down_proj) {
    return h;
  }

  // Pre-norm MLP
  Tensor normed2 = (config_.norm_type == NormType::RMS_NORM) 
                   ? rms_norm_->forward(h, *post_attn_ln) 
                   : layer_norm_->forward(h, *post_attn_ln);

  Tensor mlp_output = mlp_->forward(normed2, *gate_proj, *up_proj, *down_proj);

  // Residual
  Tensor output({x.shape[0], x.shape[1]});
  for (size_t i = 0; i < h.data.size(); ++i) {
    output.data[i] = h.data[i] + mlp_output.data[i];
  }

  return output;
}

Tensor LLMModel::transformer_block_with_cache(const Tensor& x, int layer_idx, 
                                               layers::KVCache& cache) {
  const Tensor* input_ln = get_weight(layer_weight_name(layer_idx, config_.input_layernorm));
  const Tensor* post_attn_ln = get_weight(layer_weight_name(layer_idx, config_.post_attention_layernorm));

  if (!input_ln || !post_attn_ln) {
    return x;
  }

  // Get attention weights
  const Tensor* q_proj = get_weight(layer_weight_name(layer_idx, config_.q_proj));
  const Tensor* k_proj = get_weight(layer_weight_name(layer_idx, config_.k_proj));
  const Tensor* v_proj = get_weight(layer_weight_name(layer_idx, config_.v_proj));
  const Tensor* o_proj = get_weight(layer_weight_name(layer_idx, config_.o_proj));

  if (!q_proj || !k_proj || !v_proj || !o_proj) {
    return x;
  }

  // Optional biases and norm weights
  const Tensor* q_bias = nullptr;
  const Tensor* k_bias = nullptr;
  const Tensor* v_bias = nullptr;
  const Tensor* q_norm = nullptr;
  const Tensor* k_norm = nullptr;
  
  if (config_.use_qkv_bias) {
    std::string prefix = config_.layer_prefix + std::to_string(layer_idx) + ".self_attn.";
    q_bias = get_weight(prefix + "q_proj.bias", false);
    k_bias = get_weight(prefix + "k_proj.bias", false);
    v_bias = get_weight(prefix + "v_proj.bias", false);
  }

  if (config_.use_qk_norm) {
    std::string prefix = config_.layer_prefix + std::to_string(layer_idx) + ".self_attn.";
    q_norm = get_weight(prefix + "q_norm.weight", false);
    k_norm = get_weight(prefix + "k_norm.weight", false);
  }

  Tensor normed = (config_.norm_type == NormType::RMS_NORM) 
                  ? rms_norm_->forward(x, *input_ln) 
                  : layer_norm_->forward(x, *input_ln);

  Tensor attn_output = attention_->forward_with_cache(
    normed, *q_proj, *k_proj, *v_proj, *o_proj,
    *rope_, cache, layer_idx,
    q_bias, k_bias, v_bias, q_norm, k_norm
  );

  // Residual connection: h = x + attn_output
  Tensor h({x.shape[0], x.shape[1]});
#if defined(__APPLE__)
  vDSP_vadd(x.data.data(), 1, attn_output.data.data(), 1, h.data.data(), 1, x.data.size());
#else
  for (size_t i = 0; i < x.data.size(); ++i) {
    h.data[i] = x.data[i] + attn_output.data[i];
  }
#endif

  // Get MLP weights
  const Tensor* gate_proj = get_weight(layer_weight_name(layer_idx, config_.gate_proj));
  const Tensor* up_proj = get_weight(layer_weight_name(layer_idx, config_.up_proj));
  const Tensor* down_proj = get_weight(layer_weight_name(layer_idx, config_.down_proj));

  if (!gate_proj || !up_proj || !down_proj) {
    return h;
  }

  Tensor normed2 = (config_.norm_type == NormType::RMS_NORM) 
                   ? rms_norm_->forward(h, *post_attn_ln) 
                   : layer_norm_->forward(h, *post_attn_ln);

  Tensor mlp_output = mlp_->forward(normed2, *gate_proj, *up_proj, *down_proj);

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
                    ? rms_norm_->forward(hidden_states, *final_norm) 
                    : layer_norm_->forward(hidden_states, *final_norm);
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

  const float* last_hidden_ptr = hidden_states.data.data() + (seq_len - 1) * hidden_size;

  Tensor logits({vocab_size});

#if defined(__APPLE__)
  cblas_sgemv(
    CblasRowMajor, CblasNoTrans,
    vocab_size, hidden_size,
    1.0f,
    lm_head->data.data(), hidden_size,
    last_hidden_ptr, 1,
    0.0f,
    logits.data.data(), 1
  );
#else
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

Tensor LLMModel::forward_with_cache(const std::vector<int>& input_ids, 
                                     layers::KVCache& cache) {
  Tensor hidden_states = embedding(input_ids);

  for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
    hidden_states = transformer_block_with_cache(hidden_states, layer, cache);
  }

  cache.current_length += input_ids.size();

  const Tensor* final_norm = get_weight(config_.norm_pattern);
  if (final_norm) {
    hidden_states = (config_.norm_type == NormType::RMS_NORM) 
                    ? rms_norm_->forward(hidden_states, *final_norm) 
                    : layer_norm_->forward(hidden_states, *final_norm);
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

  const float* last_hidden_ptr = hidden_states.data.data() + (seq_len - 1) * hidden_size;

  Tensor logits({vocab_size});

#if defined(__APPLE__)
  cblas_sgemv(
    CblasRowMajor, CblasNoTrans,
    vocab_size, hidden_size,
    1.0f,
    lm_head->data.data(), hidden_size,
    last_hidden_ptr, 1,
    0.0f,
    logits.data.data(), 1
  );
#else
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

  layers::KVCache cache;
  cache.init(config_.num_hidden_layers, config_.max_position_embeddings,
             config_.num_key_value_heads, config_.head_dim());

  // Prefill
  Tensor logits = forward_with_cache(input_ids, cache);

  // Generate
  for (int i = 0; i < config.max_new_tokens; ++i) {
    int next_token = sample_token(logits, config, generated);
    generated.push_back(next_token);

    // Call the token callback if provided
    if (token_callback) {
      if (!token_callback(next_token)) {
        break;
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

} // namespace models

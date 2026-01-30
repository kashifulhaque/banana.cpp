#ifndef CONFIG_LLM_CONFIG_H
#define CONFIG_LLM_CONFIG_H

#include <string>
#include <vector>
#include <unordered_map>

/// Supported model architectures
enum class ModelArchitecture {
  LLAMA,      // LLaMA, SmolLM, SmolLM2, Llama 3.x
  QWEN,       // Qwen, Qwen2, Qwen3
  MISTRAL,    // Mistral (similar to LLaMA)
  PHI,        // Phi-2, Phi-3
  UNKNOWN
};

/// Activation function types
enum class ActivationType {
  SILU,       // SiLU / Swish (LLaMA, Qwen)
  GELU,       // GELU (GPT-2, some older models)
  GELU_NEW,   // GELU approximation
  RELU,       // ReLU
  SWIGLU      // SwiGLU (gate * silu(up))
};

/// Normalization types
enum class NormType {
  RMS_NORM,   // RMSNorm (LLaMA, Qwen, Mistral)
  LAYER_NORM  // LayerNorm (GPT-2, BERT)
};

/// Position embedding types
enum class PositionEmbeddingType {
  ROPE,           // Rotary Position Embeddings
  ALIBI,          // ALiBi (Attention with Linear Biases)
  ABSOLUTE,       // Absolute positional embeddings
  RELATIVE        // Relative positional embeddings
};

/// Generic LLM configuration that works across model families
struct LLMConfig {
  // Model identification
  std::string model_type;               // "llama", "qwen2", "mistral", etc.
  std::string model_name;               // Human-readable name
  ModelArchitecture architecture = ModelArchitecture::LLAMA;
  
  // Core dimensions
  int vocab_size = 32000;               // Vocabulary size
  int hidden_size = 2048;               // Embedding dimension (d_model)
  int intermediate_size = 5632;         // MLP intermediate dimension
  int num_hidden_layers = 22;           // Number of transformer blocks
  int num_attention_heads = 32;         // Number of query heads
  int num_key_value_heads = 32;         // Number of KV heads (GQA when < num_attention_heads)
  int max_position_embeddings = 2048;   // Maximum context length
  int head_dim_override = 0;            // Explicit head dimension (0 = compute from hidden_size/num_heads)
  
  // Normalization
  NormType norm_type = NormType::RMS_NORM;
  float rms_norm_eps = 1e-5f;           // RMSNorm epsilon
  float layer_norm_eps = 1e-5f;         // LayerNorm epsilon (if used)
  
  // Position embeddings
  PositionEmbeddingType pos_embedding_type = PositionEmbeddingType::ROPE;
  float rope_theta = 10000.0f;          // RoPE theta base
  float rope_scaling_factor = 1.0f;     // RoPE scaling (for extended context)
  std::string rope_scaling_type;        // "linear", "dynamic", "yarn", etc.
  
  // Activation
  ActivationType hidden_act = ActivationType::SILU;
  
  // Attention
  bool use_qkv_bias = false;            // Whether Q/K/V projections have bias
  bool use_o_bias = false;              // Whether output projection has bias
  bool use_qk_norm = false;             // Whether to apply RMSNorm to Q/K (Qwen3)
  float attention_dropout = 0.0f;       // Attention dropout (usually 0 for inference)
  
  // MLP
  bool use_mlp_bias = false;            // Whether MLP layers have bias
  
  // Embeddings
  bool tie_word_embeddings = false;     // Share input/output embeddings
  
  // Special tokens
  int bos_token_id = 1;
  int eos_token_id = 2;
  int pad_token_id = 0;
  std::vector<int> eos_token_ids;       // Multiple EOS tokens (some models)
  
  // Derived values
  int head_dim() const { 
    return head_dim_override > 0 ? head_dim_override : hidden_size / num_attention_heads; 
  }
  int num_queries_per_kv() const { 
    return num_attention_heads / num_key_value_heads; 
  }
  bool uses_gqa() const { return num_key_value_heads < num_attention_heads; }
  
  // Weight name patterns (for different model formats)
  std::string embed_tokens_pattern = "model.embed_tokens.weight";
  std::string lm_head_pattern = "lm_head.weight";
  std::string norm_pattern = "model.norm.weight";
  std::string layer_prefix = "model.layers.";
  
  // Layer weight patterns (relative to layer prefix)
  std::string input_layernorm = "input_layernorm.weight";
  std::string post_attention_layernorm = "post_attention_layernorm.weight";
  std::string q_proj = "self_attn.q_proj.weight";
  std::string k_proj = "self_attn.k_proj.weight";
  std::string v_proj = "self_attn.v_proj.weight";
  std::string o_proj = "self_attn.o_proj.weight";
  std::string gate_proj = "mlp.gate_proj.weight";
  std::string up_proj = "mlp.up_proj.weight";
  std::string down_proj = "mlp.down_proj.weight";
};

/// Sampling configuration for text generation
struct SamplingConfig {
  float temperature = 0.7f;           // 0.0 = greedy, 1.0 = neutral
  int top_k = 50;                     // 0 = disabled
  float top_p = 0.9f;                 // Nucleus sampling
  float repetition_penalty = 1.1f;    // 1.0 = no penalty
  float frequency_penalty = 0.0f;     // Penalize frequent tokens
  float presence_penalty = 0.0f;      // Penalize already-present tokens
  int max_new_tokens = 256;           // Maximum tokens to generate
  int min_new_tokens = 0;             // Minimum tokens before EOS allowed
  std::vector<int> stop_token_ids;    // Stop on these tokens
};

/// Predefined configurations for known models
namespace ModelPresets {

// ============================================================================
// SmolLM2 Family
// ============================================================================

inline LLMConfig smollm2_135m() {
  LLMConfig cfg;
  cfg.model_type = "llama";
  cfg.model_name = "SmolLM2-135M";
  cfg.architecture = ModelArchitecture::LLAMA;
  cfg.vocab_size = 49152;
  cfg.hidden_size = 576;
  cfg.intermediate_size = 1536;
  cfg.num_hidden_layers = 30;
  cfg.num_attention_heads = 9;
  cfg.num_key_value_heads = 3;
  cfg.max_position_embeddings = 8192;
  cfg.rms_norm_eps = 1e-5f;
  cfg.rope_theta = 100000.0f;
  cfg.hidden_act = ActivationType::SILU;
  cfg.tie_word_embeddings = true;
  cfg.bos_token_id = 1;
  cfg.eos_token_id = 2;
  return cfg;
}

inline LLMConfig smollm2_360m() {
  LLMConfig cfg;
  cfg.model_type = "llama";
  cfg.model_name = "SmolLM2-360M";
  cfg.architecture = ModelArchitecture::LLAMA;
  cfg.vocab_size = 49152;
  cfg.hidden_size = 960;
  cfg.intermediate_size = 2560;
  cfg.num_hidden_layers = 32;
  cfg.num_attention_heads = 15;
  cfg.num_key_value_heads = 5;
  cfg.max_position_embeddings = 8192;
  cfg.rms_norm_eps = 1e-5f;
  cfg.rope_theta = 100000.0f;
  cfg.hidden_act = ActivationType::SILU;
  cfg.tie_word_embeddings = true;
  cfg.bos_token_id = 1;
  cfg.eos_token_id = 2;
  return cfg;
}

inline LLMConfig smollm2_1_7b() {
  LLMConfig cfg;
  cfg.model_type = "llama";
  cfg.model_name = "SmolLM2-1.7B";
  cfg.architecture = ModelArchitecture::LLAMA;
  cfg.vocab_size = 49152;
  cfg.hidden_size = 2048;
  cfg.intermediate_size = 8192;
  cfg.num_hidden_layers = 24;
  cfg.num_attention_heads = 32;
  cfg.num_key_value_heads = 32;
  cfg.max_position_embeddings = 8192;
  cfg.rms_norm_eps = 1e-5f;
  cfg.rope_theta = 100000.0f;
  cfg.hidden_act = ActivationType::SILU;
  cfg.tie_word_embeddings = true;
  cfg.bos_token_id = 1;
  cfg.eos_token_id = 2;
  return cfg;
}

// ============================================================================
// Llama 3.2 Family
// ============================================================================

inline LLMConfig llama3_2_1b() {
  LLMConfig cfg;
  cfg.model_type = "llama";
  cfg.model_name = "Llama-3.2-1B";
  cfg.architecture = ModelArchitecture::LLAMA;
  cfg.vocab_size = 128256;
  cfg.hidden_size = 2048;
  cfg.intermediate_size = 8192;
  cfg.num_hidden_layers = 16;
  cfg.num_attention_heads = 32;
  cfg.num_key_value_heads = 8;
  cfg.max_position_embeddings = 131072;
  cfg.rms_norm_eps = 1e-5f;
  cfg.rope_theta = 500000.0f;
  cfg.hidden_act = ActivationType::SILU;
  cfg.tie_word_embeddings = true;
  cfg.bos_token_id = 128000;
  cfg.eos_token_id = 128001;
  cfg.eos_token_ids = {128001, 128008, 128009};
  return cfg;
}

inline LLMConfig llama3_2_3b() {
  LLMConfig cfg;
  cfg.model_type = "llama";
  cfg.model_name = "Llama-3.2-3B";
  cfg.architecture = ModelArchitecture::LLAMA;
  cfg.vocab_size = 128256;
  cfg.hidden_size = 3072;
  cfg.intermediate_size = 8192;
  cfg.num_hidden_layers = 28;
  cfg.num_attention_heads = 24;
  cfg.num_key_value_heads = 8;
  cfg.max_position_embeddings = 131072;
  cfg.rms_norm_eps = 1e-5f;
  cfg.rope_theta = 500000.0f;
  cfg.hidden_act = ActivationType::SILU;
  cfg.tie_word_embeddings = true;
  cfg.bos_token_id = 128000;
  cfg.eos_token_id = 128001;
  cfg.eos_token_ids = {128001, 128008, 128009};
  return cfg;
}

// ============================================================================
// Qwen Family
// ============================================================================

inline LLMConfig qwen3_0_6b() {
  LLMConfig cfg;
  cfg.model_type = "qwen2";
  cfg.model_name = "Qwen3-0.6B";
  cfg.architecture = ModelArchitecture::QWEN;
  cfg.vocab_size = 151936;
  cfg.hidden_size = 1024;
  cfg.intermediate_size = 3072;
  cfg.num_hidden_layers = 28;
  cfg.num_attention_heads = 16;
  cfg.num_key_value_heads = 8;
  cfg.head_dim_override = 128;   // Qwen3 uses 128 head_dim (not hidden_size/num_heads)
  cfg.max_position_embeddings = 32768;
  cfg.rms_norm_eps = 1e-6f;
  cfg.rope_theta = 1000000.0f;
  cfg.hidden_act = ActivationType::SILU;
  cfg.use_qkv_bias = true;  // Qwen uses bias in attention
  cfg.use_qk_norm = true;   // Qwen3 uses Q/K normalization
  cfg.tie_word_embeddings = true;
  cfg.bos_token_id = 151643;
  cfg.eos_token_id = 151645;
  return cfg;
}

inline LLMConfig qwen2_5_0_5b() {
  LLMConfig cfg;
  cfg.model_type = "qwen2";
  cfg.model_name = "Qwen2.5-0.5B";
  cfg.architecture = ModelArchitecture::QWEN;
  cfg.vocab_size = 151936;
  cfg.hidden_size = 896;
  cfg.intermediate_size = 4864;
  cfg.num_hidden_layers = 24;
  cfg.num_attention_heads = 14;
  cfg.num_key_value_heads = 2;
  cfg.max_position_embeddings = 32768;
  cfg.rms_norm_eps = 1e-6f;
  cfg.rope_theta = 1000000.0f;
  cfg.hidden_act = ActivationType::SILU;
  cfg.use_qkv_bias = true;
  cfg.tie_word_embeddings = true;
  cfg.bos_token_id = 151643;
  cfg.eos_token_id = 151645;
  return cfg;
}

} // namespace ModelPresets

#endif // CONFIG_LLM_CONFIG_H

#ifndef SMOLLM2_H
#define SMOLLM2_H

#include "tensor.h"
#include "model_loader.h"
#include <string>
#include <vector>
#include <cmath>

// SmolLM2 configuration (based on HuggingFace config.json)
struct SmolLM2Config {
    int vocab_size = 49152;          // vocabulary size
    int hidden_size = 960;           // embedding dimension (d_model)
    int intermediate_size = 2560;    // MLP intermediate dimension
    int num_hidden_layers = 32;      // number of transformer blocks
    int num_attention_heads = 15;    // number of query heads
    int num_key_value_heads = 5;     // number of KV heads (GQA: 15/5 = 3 queries per KV)
    int max_position_embeddings = 8192;  // max context length
    float rms_norm_eps = 1e-5f;      // RMSNorm epsilon
    float rope_theta = 100000.0f;    // RoPE theta base
    bool tie_word_embeddings = true; // share input/output embeddings
    
    // Derived values
    int head_dim() const { return hidden_size / num_attention_heads; }
    int num_queries_per_kv() const { return num_attention_heads / num_key_value_heads; }
};

// Sampling configuration
struct SmolLM2SamplingConfig {
    float temperature = 0.7f;         // 0.0 = greedy, 1.0 = neutral, >1.0 = more random
    int top_k = 50;                   // 0 = disabled, >0 = only sample from top k tokens
    float top_p = 0.9f;               // 0.0-1.0, nucleus sampling threshold
    float repetition_penalty = 1.1f;  // 1.0 = no penalty, >1.0 = penalize repeats
    int max_new_tokens = 256;         // maximum tokens to generate
};

// KV Cache for efficient autoregressive generation
struct KVCache {
    std::vector<Tensor> key_cache;    // [num_layers] of [max_seq, num_kv_heads, head_dim]
    std::vector<Tensor> value_cache;  // [num_layers] of [max_seq, num_kv_heads, head_dim]
    int current_length = 0;
    
    void init(const SmolLM2Config& config, int max_seq_len);
    void clear();
};

class SmolLM2Model {
public:
    SmolLM2Model(ModelLoader& loader);
    SmolLM2Model(ModelLoader& loader, const SmolLM2Config& config);
    
    // Forward pass: given input token IDs, return logits for next token
    Tensor forward(const std::vector<int>& input_ids);
    
    // Forward pass with KV cache for efficient generation
    Tensor forward_with_cache(const std::vector<int>& input_ids, KVCache& cache);
    
    // Generate text given a prompt
    std::vector<int> generate(const std::vector<int>& input_ids, 
                               const SmolLM2SamplingConfig& config = SmolLM2SamplingConfig());
    
    const SmolLM2Config& get_config() const { return config_; }
    
private:
    ModelLoader& loader_;
    SmolLM2Config config_;
    
    // Precomputed RoPE frequencies
    std::vector<float> cos_cached_;
    std::vector<float> sin_cached_;
    int max_cached_positions_ = 0;
    
    // Helper functions
    void init_rope_cache(int max_seq_len);
    
    Tensor embedding(const std::vector<int>& input_ids);
    Tensor transformer_block(const Tensor& x, int layer_idx, int start_pos);
    Tensor transformer_block_with_cache(const Tensor& x, int layer_idx, KVCache& cache);
    
    // Attention with Grouped Query Attention (GQA) and RoPE
    Tensor attention(const Tensor& x, int layer_idx, int start_pos);
    Tensor attention_with_cache(const Tensor& x, int layer_idx, KVCache& cache);
    
    // SwiGLU MLP
    Tensor mlp(const Tensor& x, int layer_idx);
    
    // RMSNorm
    Tensor rms_norm(const Tensor& x, const Tensor& weight);
    
    // Apply RoPE to query and key tensors
    void apply_rope(Tensor& q, Tensor& k, int start_pos);
    
    // Linear projection
    Tensor linear(const Tensor& x, const Tensor& weight);
    Tensor linear_with_bias(const Tensor& x, const Tensor& weight, const Tensor& bias);
    
    // Get weight by name
    const Tensor* get_weight(const std::string& name);
    
    // Sampling helpers
    int sample_token(const Tensor& logits, const SmolLM2SamplingConfig& config,
                     const std::vector<int>& generated_tokens);
};

#endif // SMOLLM2_H

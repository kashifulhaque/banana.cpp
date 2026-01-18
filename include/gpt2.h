#ifndef GPT2_H
#define GPT2_H

#include "tensor.h"
#include "model_loader.h"
#include <string>
#include <vector>

// GPT-2 configuration
struct GPT2Config {
    int n_vocab = 50257;       // vocabulary size
    int n_ctx = 1024;          // context length
    int n_embd = 768;          // embedding dimension
    int n_head = 12;           // number of attention heads
    int n_layer = 12;          // number of transformer blocks
};

// Sampling configuration
struct SamplingConfig {
    float temperature = 1.0f;      // 0.0 = greedy, 1.0 = neutral, >1.0 = more random
    int top_k = 50;                // 0 = disabled, >0 = only sample from top k tokens
    float top_p = 0.95f;           // 0.0-1.0, nucleus sampling threshold
    float repetition_penalty = 1.1f; // 1.0 = no penalty, >1.0 = penalize repeats
};

class GPT2Model {
public:
    GPT2Model(ModelLoader& loader);
    
    // Forward pass: given input token IDs, return logits for next token
    Tensor forward(const std::vector<int>& input_ids);
    
    // Generate text given a prompt
    std::vector<int> generate(const std::vector<int>& input_ids, int max_new_tokens, const SamplingConfig& config = SamplingConfig());
    
private:
    ModelLoader& loader_;
    GPT2Config config_;
    
    // Helper functions
    Tensor embedding(const std::vector<int>& input_ids);
    Tensor transformer_block(const Tensor& x, int layer_idx);
    Tensor attention(const Tensor& x, int layer_idx);
    Tensor mlp(const Tensor& x, int layer_idx);
    
    // Get weight by name
    const Tensor* get_weight(const std::string& name);
};

#endif // GPT2_H

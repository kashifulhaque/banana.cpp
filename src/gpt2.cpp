#include "gpt2.h"
#include "ops.h"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <algorithm>

GPT2Model::GPT2Model(ModelLoader& loader) : loader_(loader) {
    std::cout << "Initializing GPT-2 model..." << std::endl;
}

const Tensor* GPT2Model::get_weight(const std::string& name) {
    const Tensor* weight = loader_.get(name);
    if (!weight) {
        std::cerr << "Warning: Weight '" << name << "' not found!" << std::endl;
    }
    return weight;
}

Tensor GPT2Model::embedding(const std::vector<int>& input_ids) {
    /// get embeddings: wte (token embeddings) and wpe (position embeddings)
    const Tensor* wte = get_weight("transformer.wte.weight");
    const Tensor* wpe = get_weight("transformer.wpe.weight");
    
    if (!wte || !wpe) {
        std::cerr << "Failed to load embeddings!" << std::endl;
        return Tensor();
    }
    
    int seq_len = input_ids.size();
    int n_embd = config_.n_embd;
    
    /// create output tensor [seq_len, n_embd]
    Tensor output({seq_len, n_embd});
    
    /// add token and position embeddings
    for (int pos = 0; pos < seq_len; ++pos) {
        int token_id = input_ids[pos];
        
        for (int i = 0; i < n_embd; ++i) {
            float token_emb = wte->data[token_id * n_embd + i];
            float pos_emb = wpe->data[pos * n_embd + i];
            output.data[pos * n_embd + i] = token_emb + pos_emb;
        }
    }
    
    return output;
}

Tensor GPT2Model::attention(const Tensor& x, int layer_idx) {
    /// self-attention
    std::string prefix = "transformer.h." + std::to_string(layer_idx) + ".attn.";
    
    const Tensor* c_attn_weight = get_weight(prefix + "c_attn.weight");
    const Tensor* c_attn_bias = get_weight(prefix + "c_attn.bias");
    const Tensor* c_proj_weight = get_weight(prefix + "c_proj.weight");
    const Tensor* c_proj_bias = get_weight(prefix + "c_proj.bias");
    
    if (!c_attn_weight || !c_attn_bias || !c_proj_weight || !c_proj_bias) {
        return x;
    }
    
    int seq_len = x.shape[0];
    int n_embd = x.shape[1];
    int n_head = config_.n_head;
    int head_dim = n_embd / n_head;
    
    /// linear projection to get Q, K, V
    /// c_attn projects to 3 * n_embd (for Q, K, V concatenated)
    Tensor qkv = ops::matmul_ex(x, *c_attn_weight, false, false);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < n_embd * 3; ++j) {
            qkv.data[i * (n_embd * 3) + j] += c_attn_bias->data[j];
        }
    }
    
    /// Split into Q, K, V
    Tensor q({seq_len, n_embd});
    Tensor k({seq_len, n_embd});
    Tensor v({seq_len, n_embd});
    
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < n_embd; ++j) {
            q.data[i * n_embd + j] = qkv.data[i * (n_embd * 3) + j];
            k.data[i * n_embd + j] = qkv.data[i * (n_embd * 3) + n_embd + j];
            v.data[i * n_embd + j] = qkv.data[i * (n_embd * 3) + 2 * n_embd + j];
        }
    }
    
    /// compute attention: Q @ K^T / sqrt(head_dim)
    Tensor scores({seq_len, seq_len});
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    scores = ops::matmul_ex(q, k, false, true);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            scores.data[i * seq_len + j] *= scale;
        }
    }
    
    /// apply causal mask (no looking ahead)
    for (int i = 0; i < seq_len; ++i) {
        for (int j = i + 1; j < seq_len; ++j) {
            scores.data[i * seq_len + j] = -1e10f;  // Large negative value
        }
    }
    
    /// apply softmax
    Tensor attn_weights = ops::softmax(scores, -1);
    
    /// apply attention to values: attn_weights @ V
    Tensor attn_output = ops::matmul_ex(attn_weights, v, false, false);
    
    /// output projection
    Tensor output = ops::matmul_ex(attn_output, *c_proj_weight, false, false);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < n_embd; ++j) {
            output.data[i * n_embd + j] += c_proj_bias->data[j];
        }
    }
    
    return output;
}

Tensor GPT2Model::mlp(const Tensor& x, int layer_idx) {
    /// mlp: linear -> gelu -> linear
    std::string prefix = "transformer.h." + std::to_string(layer_idx) + ".mlp.";
    
    const Tensor* c_fc_weight = get_weight(prefix + "c_fc.weight");
    const Tensor* c_fc_bias = get_weight(prefix + "c_fc.bias");
    const Tensor* c_proj_weight = get_weight(prefix + "c_proj.weight");
    const Tensor* c_proj_bias = get_weight(prefix + "c_proj.bias");
    
    if (!c_fc_weight || !c_fc_bias || !c_proj_weight || !c_proj_bias) {
        return x;
    }
    
    int seq_len = x.shape[0];
    int n_embd = x.shape[1];
    int n_inner = n_embd * 4;  // GPT-2 uses 4x expansion
    
    /// first linear layer
    Tensor hidden = ops::matmul_ex(x, *c_fc_weight, false, false);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < n_inner; ++j) {
            hidden.data[i * n_inner + j] += c_fc_bias->data[j];
        }
    }
    
    /// gelu activation
    hidden = ops::gelu(hidden);
    
    /// second linear layer
    Tensor output = ops::matmul_ex(hidden, *c_proj_weight, false, false);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < n_embd; ++j) {
            output.data[i * n_embd + j] += c_proj_bias->data[j];
        }
    }
    
    return output;
}

Tensor GPT2Model::transformer_block(const Tensor& x, int layer_idx) {
    std::string prefix = "transformer.h." + std::to_string(layer_idx) + ".";
    
    const Tensor* ln1_weight = get_weight(prefix + "ln_1.weight");
    const Tensor* ln1_bias = get_weight(prefix + "ln_1.bias");
    const Tensor* ln2_weight = get_weight(prefix + "ln_2.weight");
    const Tensor* ln2_bias = get_weight(prefix + "ln_2.bias");
    
    if (!ln1_weight || !ln1_bias || !ln2_weight || !ln2_bias) {
        return x;
    }
    
    /// layer norm + attention + residual
    Tensor normed1 = ops::layer_norm(x, *ln1_weight, *ln1_bias);
    Tensor attn_out = attention(normed1, layer_idx);
    
    /// residual connection
    Tensor x2 = ops::add(x, attn_out);
    
    /// layer norm + MLP + residual
    Tensor normed2 = ops::layer_norm(x2, *ln2_weight, *ln2_bias);
    Tensor mlp_out = mlp(normed2, layer_idx);
    
    /// residual connection
    Tensor output = ops::add(x2, mlp_out);
    
    return output;
}

Tensor GPT2Model::forward(const std::vector<int>& input_ids) {
    /// get embeddings
    Tensor x = embedding(input_ids);
    
    std::cout << "Processing through " << config_.n_layer << " transformer layers..." << std::endl;
    
    /// pass through transformer blocks
    for (int i = 0; i < config_.n_layer; ++i) {
        x = transformer_block(x, i);
        if ((i + 1) % 3 == 0) {
            std::cout << "  Completed layer " << (i + 1) << "/" << config_.n_layer << std::endl;
        }
    }
    
    /// final layer norm
    const Tensor* ln_f_weight = get_weight("transformer.ln_f.weight");
    const Tensor* ln_f_bias = get_weight("transformer.ln_f.bias");
    
    if (ln_f_weight && ln_f_bias) {
        x = ops::layer_norm(x, *ln_f_weight, *ln_f_bias);
    }
    
    /// language model head (project to vocabulary)
    const Tensor* lm_head = get_weight("lm_head.weight");
    if (!lm_head) {
        /// in GPT-2, lm_head often shares weights with wte
        lm_head = get_weight("transformer.wte.weight");
    }
    
    if (!lm_head) {
        std::cerr << "Failed to load lm_head!" << std::endl;
        return Tensor();
    }
    
    int seq_len = x.shape[0];
    int n_embd = x.shape[1];
    int n_vocab = config_.n_vocab;
    
    /// only compute logits for the last token
    Tensor logits({1, n_vocab});
    
    for (int j = 0; j < n_vocab; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < n_embd; ++k) {
            sum += x.data[(seq_len - 1) * n_embd + k] * lm_head->data[j * n_embd + k];
        }
        logits.data[j] = sum;
    }
    
    return logits;
}

std::vector<int> GPT2Model::generate(const std::vector<int>& input_ids, int max_new_tokens, const SamplingConfig& config) {
    std::vector<int> output_ids = input_ids;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int i = 0; i < max_new_tokens; ++i) {
        /// forward pass
        Tensor logits = forward(output_ids);
        
        /// apply repetition penalty (penalize tokens already in the sequence)
        if (config.repetition_penalty != 1.0f) {
            for (int token : output_ids) {
                if (token < logits.size()) {
                    // If logit is positive, divide by penalty; if negative, multiply
                    if (logits.data[token] > 0) {
                        logits.data[token] /= config.repetition_penalty;
                    } else {
                        logits.data[token] *= config.repetition_penalty;
                    }
                }
            }
        }
        
        /// apply temperature
        if (config.temperature > 0.0f && config.temperature != 1.0f) {
            for (int j = 0; j < logits.size(); ++j) {
                logits.data[j] /= config.temperature;
            }
        }
        
        /// convert logits to probabilities using softmax
        std::vector<float> probs(logits.size());
        float max_logit = *std::max_element(logits.data.begin(), logits.data.end());
        float sum = 0.0f;
        for (int j = 0; j < logits.size(); ++j) {
            probs[j] = std::exp(logits.data[j] - max_logit);
            sum += probs[j];
        }
        for (int j = 0; j < logits.size(); ++j) {
            probs[j] /= sum;
        }
        
        int next_token = 0;
        
        if (config.temperature == 0.0f) {
            /// greedy sampling
            next_token = std::max_element(probs.begin(), probs.end()) - probs.begin();
        } else {
            /// create indices sorted by probability
            std::vector<std::pair<float, int>> prob_idx;
            for (int j = 0; j < probs.size(); ++j) {
                prob_idx.push_back({probs[j], j});
            }
            std::sort(prob_idx.begin(), prob_idx.end(), std::greater<std::pair<float, int>>());
            
            // apply top-k filtering
            int k = config.top_k;
            if (k > 0 && k < (int)prob_idx.size()) {
                prob_idx.resize(k);
            }
            
            /// apply top-p (nucleus) filtering
            if (config.top_p < 1.0f) {
                float cumsum = 0.0f;
                size_t cutoff = prob_idx.size();
                for (size_t j = 0; j < prob_idx.size(); ++j) {
                    cumsum += prob_idx[j].first;
                    if (cumsum > config.top_p) {
                        cutoff = j + 1;
                        break;
                    }
                }
                prob_idx.resize(cutoff);
            }
            
            /// renormalize probabilities
            float prob_sum = 0.0f;
            for (const auto& p : prob_idx) {
                prob_sum += p.first;
            }
            
            /// sample from the filtered distribution
            std::uniform_real_distribution<float> dist(0.0f, prob_sum);
            float random_val = dist(gen);
            float cumsum = 0.0f;
            
            for (const auto& p : prob_idx) {
                cumsum += p.first;
                if (cumsum >= random_val) {
                    next_token = p.second;
                    break;
                }
            }
        }
        
        output_ids.push_back(next_token);
        
        std::cout << "Generated token " << (i + 1) << "/" << max_new_tokens 
                  << " (id: " << next_token << ")" << std::endl;
        
        /// stop if we hit end of text token (50256 for GPT-2)
        if (next_token == 50256) {
            break;
        }
    }
    
    return output_ids;
}

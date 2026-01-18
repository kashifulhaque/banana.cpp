#ifndef SMOLLM2_TOKENIZER_H
#define SMOLLM2_TOKENIZER_H

#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <set>

// SmolLM2 uses a GPT-2 style BPE tokenizer with special tokens
class SmolLM2Tokenizer {
public:
    SmolLM2Tokenizer();
    ~SmolLM2Tokenizer();
    
    // Load tokenizer from vocab.json and merges.txt
    bool load(const std::string& tokenizer_dir);
    
    // Download and load tokenizer files from HuggingFace
    bool download_and_load(const std::string& save_dir = "weights/smollm2");
    
    // Encode text to token IDs
    std::vector<int> encode(const std::string& text);
    
    // Decode token IDs to text
    std::string decode(const std::vector<int>& tokens);
    
    // Apply chat template for instruct model
    std::string apply_chat_template(const std::vector<std::pair<std::string, std::string>>& messages,
                                     bool add_generation_prompt = true);
    
    // Special token IDs
    int get_eos_token_id() const { return eos_token_id_; }
    int get_bos_token_id() const { return bos_token_id_; }
    int get_pad_token_id() const { return pad_token_id_; }
    int get_im_start_id() const { return im_start_id_; }
    int get_im_end_id() const { return im_end_id_; }
    
    int vocab_size() const { return vocab_size_; }
    
private:
    // Vocabulary mappings
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;
    
    // BPE merge ranks
    std::map<std::pair<std::string, std::string>, int> bpe_ranks_;
    
    // Special token IDs
    int eos_token_id_ = 2;      // <|im_end|>
    int bos_token_id_ = 1;      // <|im_start|>
    int pad_token_id_ = 2;      // <|im_end|>
    int im_start_id_ = 1;       // <|im_start|>
    int im_end_id_ = 2;         // <|im_end|>
    int endoftext_id_ = 0;      // <|endoftext|>
    
    int vocab_size_ = 49152;
    
    // Byte-to-unicode mapping (GPT-2 style)
    std::unordered_map<int, std::string> byte_encoder_;
    std::unordered_map<std::string, int> byte_decoder_;
    
    void init_byte_encoder();
    
    // BPE helpers
    std::vector<std::string> bpe(const std::string& token);
    std::vector<std::string> split_to_words(const std::string& text);
    
    // Helper to get pairs of consecutive elements
    std::set<std::pair<std::string, std::string>> get_pairs(const std::vector<std::string>& word);
};

#endif // SMOLLM2_TOKENIZER_H

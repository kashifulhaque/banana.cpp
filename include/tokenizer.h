#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>

/// Chat template types for different model families
enum class ChatTemplateType {
  CHATML,       // <|im_start|>role\ncontent<|im_end|> (SmolLM, Qwen)
  LLAMA3,       // <|begin_of_text|><|start_header_id|>role<|end_header_id|>content<|eot_id|>
  LLAMA2,       // [INST] <<SYS>>system<</SYS>> user [/INST] assistant
  ALPACA,       // ### Instruction: ... ### Response:
  VICUNA,       // USER: ... ASSISTANT:
  ZEPHYR,       // <|user|>content</s><|assistant|>
  MISTRAL,      // [INST] user [/INST] assistant
  NONE          // No chat template, raw text
};

/// Configuration for tokenizer behavior
struct TokenizerConfig {
  std::string model_name;
  ChatTemplateType chat_template = ChatTemplateType::CHATML;
  
  // Special tokens
  std::string bos_token = "<|im_start|>";
  std::string eos_token = "<|im_end|>";
  std::string pad_token = "<|im_end|>";
  std::string unk_token = "<unk>";
  
  // Default system prompt (optional)
  std::string default_system_prompt;
  
  // Whether to add BOS token at the beginning
  bool add_bos_token = false;
  bool add_eos_token = false;
};

/// Generic BPE Tokenizer supporting multiple model formats
class Tokenizer {
public:
  Tokenizer();
  virtual ~Tokenizer();

  /// Load tokenizer from a directory containing vocab.json and merges.txt
  /// or tokenizer.json (HuggingFace format)
  bool load(const std::string& tokenizer_dir);

  /// Load with explicit config
  bool load(const std::string& tokenizer_dir, const TokenizerConfig& config);

  /// Load from tokenizer.json (unified HuggingFace format)
  bool load_from_tokenizer_json(const std::string& json_path);

  /// Download and load tokenizer from HuggingFace
  bool download_and_load(const std::string& model_id, 
                         const std::string& save_dir = "weights");

  /// Encode text to token IDs
  std::vector<int> encode(const std::string& text);

  /// Decode token IDs to text
  std::string decode(const std::vector<int>& tokens);

  /// Apply chat template for conversation
  std::string apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_generation_prompt = true);

  /// Get special token IDs
  int bos_token_id() const { return bos_token_id_; }
  int eos_token_id() const { return eos_token_id_; }
  int pad_token_id() const { return pad_token_id_; }
  int unk_token_id() const { return unk_token_id_; }
  
  /// Get all EOS token IDs (some models have multiple)
  const std::vector<int>& eos_token_ids() const { return eos_token_ids_; }

  /// Check if token is an EOS token
  bool is_eos_token(int token_id) const;

  int vocab_size() const { return vocab_size_; }
  
  /// Access config
  const TokenizerConfig& config() const { return config_; }
  void set_config(const TokenizerConfig& config) { config_ = config; update_special_tokens(); }

protected:
  TokenizerConfig config_;
  
  // Vocab mappings
  std::unordered_map<std::string, int> token_to_id_;
  std::unordered_map<int, std::string> id_to_token_;

  // BPE merge ranks
  std::map<std::pair<std::string, std::string>, int> bpe_ranks_;

  // Special token IDs
  int bos_token_id_ = -1;
  int eos_token_id_ = -1;
  int pad_token_id_ = -1;
  int unk_token_id_ = -1;
  std::vector<int> eos_token_ids_;

  int vocab_size_ = 0;

  // Byte-to-unicode mapping (GPT-2 style)
  std::unordered_map<int, std::string> byte_encoder_;
  std::unordered_map<std::string, int> byte_decoder_;

  // Added tokens (special tokens that bypass BPE)
  std::set<std::string> added_tokens_;

  void init_byte_encoder();
  void update_special_tokens();

  // BPE helpers
  std::vector<std::string> bpe(const std::string& token);
  std::vector<std::string> split_to_words(const std::string& text);
  std::set<std::pair<std::string, std::string>> get_pairs(const std::vector<std::string>& word);

  // Chat template implementations
  std::string apply_chatml(const std::vector<std::pair<std::string, std::string>>& messages, bool add_gen_prompt);
  std::string apply_llama3(const std::vector<std::pair<std::string, std::string>>& messages, bool add_gen_prompt);
  std::string apply_llama2(const std::vector<std::pair<std::string, std::string>>& messages, bool add_gen_prompt);
  std::string apply_mistral(const std::vector<std::pair<std::string, std::string>>& messages, bool add_gen_prompt);
};

/// Tokenizer presets for known models
namespace TokenizerPresets {

inline TokenizerConfig smollm() {
  TokenizerConfig cfg;
  cfg.model_name = "SmolLM";
  cfg.chat_template = ChatTemplateType::CHATML;
  cfg.bos_token = "<|im_start|>";
  cfg.eos_token = "<|im_end|>";
  cfg.default_system_prompt = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
  return cfg;
}

inline TokenizerConfig llama3() {
  TokenizerConfig cfg;
  cfg.model_name = "Llama-3";
  cfg.chat_template = ChatTemplateType::LLAMA3;
  cfg.bos_token = "<|begin_of_text|>";
  cfg.eos_token = "<|eot_id|>";
  cfg.default_system_prompt = "You are a helpful assistant.";
  return cfg;
}

inline TokenizerConfig qwen() {
  TokenizerConfig cfg;
  cfg.model_name = "Qwen";
  cfg.chat_template = ChatTemplateType::CHATML;
  cfg.bos_token = "<|im_start|>";
  cfg.eos_token = "<|im_end|>";
  cfg.default_system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
  return cfg;
}

inline TokenizerConfig mistral() {
  TokenizerConfig cfg;
  cfg.model_name = "Mistral";
  cfg.chat_template = ChatTemplateType::MISTRAL;
  cfg.bos_token = "<s>";
  cfg.eos_token = "</s>";
  return cfg;
}

} // namespace TokenizerPresets

#endif // TOKENIZER_H

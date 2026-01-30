#ifndef TOKENIZERS_TOKENIZER_H
#define TOKENIZERS_TOKENIZER_H

#include "config/tokenizer_config.h"
#include "tokenizers/bpe_tokenizer.h"
#include "tokenizers/chat_template.h"
#include <memory>
#include <string>
#include <vector>

namespace tokenizers {

/// Unified tokenizer interface combining BPE and chat template handling
class Tokenizer : public BPETokenizer {
public:
  Tokenizer();
  ~Tokenizer();

  /// Load tokenizer from a directory containing vocab.json and merges.txt
  /// or tokenizer.json (HuggingFace format)
  bool load(const std::string& tokenizer_dir);

  /// Load with explicit config
  bool load(const std::string& tokenizer_dir, const TokenizerConfig& config);

  /// Download and load tokenizer from HuggingFace
  bool download_and_load(const std::string& model_id, 
                         const std::string& save_dir = "weights");

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

  /// Access config
  const TokenizerConfig& config() const { return config_; }
  void set_config(const TokenizerConfig& config);

private:
  TokenizerConfig config_;
  std::unique_ptr<ChatTemplate> chat_template_;

  // Special token IDs
  int bos_token_id_ = -1;
  int eos_token_id_ = -1;
  int pad_token_id_ = -1;
  int unk_token_id_ = -1;
  std::vector<int> eos_token_ids_;

  /// Update special token IDs from vocabulary
  void update_special_tokens();
  
  /// Initialize chat template from config
  void init_chat_template();
};

} // namespace tokenizers

#endif // TOKENIZERS_TOKENIZER_H

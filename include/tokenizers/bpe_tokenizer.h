#ifndef TOKENIZERS_BPE_TOKENIZER_H
#define TOKENIZERS_BPE_TOKENIZER_H

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace tokenizers {

/// Byte Pair Encoding (BPE) tokenizer
/// Handles the core BPE algorithm used by GPT-2, LLaMA, SmolLM, etc.
class BPETokenizer {
public:
  BPETokenizer();
  virtual ~BPETokenizer();

  /// Load vocabulary from vocab.json file
  /// @param vocab_path Path to vocab.json
  /// @return true on success
  bool load_vocab(const std::string& vocab_path);

  /// Load BPE merges from merges.txt file
  /// @param merges_path Path to merges.txt
  /// @return true on success
  bool load_merges(const std::string& merges_path);

  /// Load from tokenizer.json (unified HuggingFace format)
  /// @param json_path Path to tokenizer.json
  /// @return true on success
  bool load_from_tokenizer_json(const std::string& json_path);

  /// Encode text to token IDs
  /// @param text Input text
  /// @return Vector of token IDs
  std::vector<int> encode(const std::string& text) const;

  /// Decode token IDs to text
  /// @param tokens Vector of token IDs
  /// @return Decoded text
  std::string decode(const std::vector<int>& tokens) const;

  /// Get token ID for a given token string
  int get_token_id(const std::string& token) const;

  /// Get token string for a given ID
  std::string get_token(int id) const;

  /// Check if a token is a special/added token
  bool is_special_token(const std::string& token) const;

  /// Add a special token
  void add_special_token(const std::string& token, int id);

  int vocab_size() const { return vocab_size_; }

protected:
  // Vocabulary mappings
  std::unordered_map<std::string, int> token_to_id_;
  std::unordered_map<int, std::string> id_to_token_;

  // BPE merge ranks
  std::map<std::pair<std::string, std::string>, int> bpe_ranks_;

  // Added tokens (special tokens that bypass BPE)
  std::set<std::string> added_tokens_;

  int vocab_size_ = 0;

  // Byte-to-unicode mapping (GPT-2 style)
  std::unordered_map<int, std::string> byte_encoder_;
  std::unordered_map<std::string, int> byte_decoder_;

  /// Initialize byte encoder/decoder
  void init_byte_encoder();

  /// Apply BPE to a single token
  std::vector<std::string> bpe(const std::string& token) const;

  /// Split text into words for BPE processing
  std::vector<std::string> split_to_words(const std::string& text) const;

  /// Get pairs of consecutive elements
  std::set<std::pair<std::string, std::string>> get_pairs(const std::vector<std::string>& word) const;

  /// Unescape JSON unicode sequences
  static std::string unescape_json(const std::string& s);
};

} // namespace tokenizers

#endif // TOKENIZERS_BPE_TOKENIZER_H

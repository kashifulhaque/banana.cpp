#include "tokenizers/tokenizer.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>

// Forward declaration for get_hf_token (from weight_downloader)
std::string get_hf_token();

namespace tokenizers {

Tokenizer::Tokenizer() : BPETokenizer() {
  init_chat_template();
}

Tokenizer::~Tokenizer() {}

void Tokenizer::init_chat_template() {
  chat_template_ = create_chat_template(config_.chat_template, 
                                         config_.default_system_prompt);
}

void Tokenizer::set_config(const TokenizerConfig& config) {
  config_ = config;
  init_chat_template();
  update_special_tokens();
}

void Tokenizer::update_special_tokens() {
  // Update IDs from vocabulary
  bos_token_id_ = get_token_id(config_.bos_token);
  eos_token_id_ = get_token_id(config_.eos_token);
  pad_token_id_ = get_token_id(config_.pad_token);
  unk_token_id_ = get_token_id(config_.unk_token);

  // Build list of EOS tokens
  eos_token_ids_.clear();
  if (eos_token_id_ >= 0) {
    eos_token_ids_.push_back(eos_token_id_);
  }

  // Add common EOS variants
  std::vector<std::string> eos_variants = {
    "<|im_end|>", "<|eot_id|>", "<|end|>", "</s>", "<|endoftext|>",
    "<|end_of_turn|>", "<|assistant|>"
  };
  for (const auto& variant : eos_variants) {
    int id = get_token_id(variant);
    if (id >= 0 && std::find(eos_token_ids_.begin(), eos_token_ids_.end(), id) == eos_token_ids_.end()) {
      eos_token_ids_.push_back(id);
    }
  }
}

bool Tokenizer::is_eos_token(int token_id) const {
  return std::find(eos_token_ids_.begin(), eos_token_ids_.end(), token_id) != eos_token_ids_.end();
}

bool Tokenizer::download_and_load(const std::string& model_id, const std::string& save_dir) {
  // Extract model name from ID for directory
  std::string model_name = model_id;
  size_t slash_pos = model_name.find('/');
  if (slash_pos != std::string::npos) {
    model_name = model_name.substr(slash_pos + 1);
  }
  
  std::string target_dir = save_dir + "/" + model_name;
  std::string mkdir_cmd = "mkdir -p " + target_dir;
  system(mkdir_cmd.c_str());

  std::string vocab_path = target_dir + "/vocab.json";
  std::string merges_path = target_dir + "/merges.txt";

  std::ifstream vocab_check(vocab_path);
  std::ifstream merges_check(merges_path);

  if (!vocab_check.good() || !merges_check.good()) {
    std::cout << "Downloading tokenizer files from " << model_id << "..." << std::endl;

    std::string hf_token = get_hf_token();
    std::string auth_header;
    if (!hf_token.empty()) {
      auth_header = " -H \"Authorization: Bearer " + hf_token + "\"";
    }

    std::string base_url = "https://huggingface.co/" + model_id + "/resolve/main/";
    
    std::string vocab_cmd = "curl -L --fail" + auth_header + " " + base_url + "vocab.json -o " + vocab_path + " 2>/dev/null";
    int ret1 = system(vocab_cmd.c_str());

    std::string merges_cmd = "curl -L --fail" + auth_header + " " + base_url + "merges.txt -o " + merges_path + " 2>/dev/null";
    int ret2 = system(merges_cmd.c_str());

    if (ret1 != 0 || ret2 != 0) {
      std::cerr << "Failed to download tokenizer files" << std::endl;
      return false;
    }
    std::cout << "Downloaded tokenizer files successfully" << std::endl;
  }

  return load(target_dir);
}

bool Tokenizer::load(const std::string& tokenizer_dir) {
  return load(tokenizer_dir, config_);
}

bool Tokenizer::load(const std::string& tokenizer_dir, const TokenizerConfig& config) {
  config_ = config;
  init_chat_template();
  
  // First, try to load from tokenizer.json (unified HuggingFace format)
  std::string tokenizer_json_path = tokenizer_dir + "/tokenizer.json";
  std::ifstream tokenizer_json_check(tokenizer_json_path);
  if (tokenizer_json_check.good()) {
    tokenizer_json_check.close();
    bool success = load_from_tokenizer_json(tokenizer_json_path);
    if (success) {
      update_special_tokens();
    }
    return success;
  }
  
  // Fall back to vocab.json + merges.txt format
  std::string vocab_path = tokenizer_dir + "/vocab.json";
  std::string merges_path = tokenizer_dir + "/merges.txt";

  if (!load_vocab(vocab_path)) {
    return false;
  }

  if (!load_merges(merges_path)) {
    return false;
  }

  update_special_tokens();
  return true;
}

std::string Tokenizer::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_generation_prompt) {
  
  if (chat_template_) {
    return chat_template_->apply(messages, add_generation_prompt);
  }
  
  // Fallback: just concatenate
  std::string result;
  for (const auto& msg : messages) {
    result += msg.second + "\n";
  }
  return result;
}

} // namespace tokenizers

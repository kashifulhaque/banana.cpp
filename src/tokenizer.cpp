#include "tokenizer.h"
#include "weight_downloader.h"  // For get_hf_token()
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>

Tokenizer::Tokenizer() {
  init_byte_encoder();
}

Tokenizer::~Tokenizer() {}

// Initialize the byte-to-unicode mapping (same as GPT-2)
void Tokenizer::init_byte_encoder() {
  std::vector<int> bs;
  for (int i = 33; i <= 126; ++i) bs.push_back(i);
  for (int i = 161; i <= 172; ++i) bs.push_back(i);
  for (int i = 174; i <= 255; ++i) bs.push_back(i);

  std::vector<int> cs = bs;
  int n = 0;

  for (int b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n);
      n++;
    }
  }

  for (size_t i = 0; i < bs.size(); ++i) {
    int byte_val = bs[i];
    int unicode_point = cs[i];

    std::string utf8_char;
    if (unicode_point < 0x80) {
      utf8_char = std::string(1, static_cast<char>(unicode_point));
    } else if (unicode_point < 0x800) {
      utf8_char = std::string(1, static_cast<char>(0xC0 | (unicode_point >> 6)));
      utf8_char += std::string(1, static_cast<char>(0x80 | (unicode_point & 0x3F)));
    } else if (unicode_point < 0x10000) {
      utf8_char = std::string(1, static_cast<char>(0xE0 | (unicode_point >> 12)));
      utf8_char += std::string(1, static_cast<char>(0x80 | ((unicode_point >> 6) & 0x3F)));
      utf8_char += std::string(1, static_cast<char>(0x80 | (unicode_point & 0x3F)));
    }

    byte_encoder_[byte_val] = utf8_char;
    byte_decoder_[utf8_char] = byte_val;
  }
}

void Tokenizer::update_special_tokens() {
  // Update IDs from vocabulary
  auto find_token = [this](const std::string& token) -> int {
    auto it = token_to_id_.find(token);
    return (it != token_to_id_.end()) ? it->second : -1;
  };

  bos_token_id_ = find_token(config_.bos_token);
  eos_token_id_ = find_token(config_.eos_token);
  pad_token_id_ = find_token(config_.pad_token);
  unk_token_id_ = find_token(config_.unk_token);

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
    int id = find_token(variant);
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

    // Get HF token for authentication
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
  
  // First, try to load from tokenizer.json (unified HuggingFace format)
  std::string tokenizer_json_path = tokenizer_dir + "/tokenizer.json";
  std::ifstream tokenizer_json_check(tokenizer_json_path);
  if (tokenizer_json_check.good()) {
    tokenizer_json_check.close();
    return load_from_tokenizer_json(tokenizer_json_path);
  }
  
  // Fall back to vocab.json + merges.txt format
  std::string vocab_path = tokenizer_dir + "/vocab.json";
  std::string merges_path = tokenizer_dir + "/merges.txt";

  // Load vocabulary
  std::ifstream vocab_file(vocab_path);
  if (!vocab_file.is_open()) {
    std::cerr << "Failed to open vocabulary file: " << vocab_path << std::endl;
    return false;
  }

  std::string content;
  std::string line;
  while (std::getline(vocab_file, line)) {
    content += line;
  }
  vocab_file.close();

  // Helper to unescape JSON unicode sequences
  auto unescape_json = [](const std::string& s) -> std::string {
    std::string result;
    for (size_t i = 0; i < s.length(); ++i) {
      if (s[i] == '\\' && i + 5 < s.length() && s[i + 1] == 'u') {
        std::string hex = s.substr(i + 2, 4);
        int unicode_point = std::stoi(hex, nullptr, 16);

        if (unicode_point < 0x80) {
          result += static_cast<char>(unicode_point);
        } else if (unicode_point < 0x800) {
          result += static_cast<char>(0xC0 | (unicode_point >> 6));
          result += static_cast<char>(0x80 | (unicode_point & 0x3F));
        } else if (unicode_point < 0x10000) {
          result += static_cast<char>(0xE0 | (unicode_point >> 12));
          result += static_cast<char>(0x80 | ((unicode_point >> 6) & 0x3F));
          result += static_cast<char>(0x80 | (unicode_point & 0x3F));
        }
        i += 5;
      } else if (s[i] == '\\' && i + 1 < s.length()) {
        switch (s[i + 1]) {
          case 'n': result += '\n'; i++; break;
          case 't': result += '\t'; i++; break;
          case 'r': result += '\r'; i++; break;
          case '\\': result += '\\'; i++; break;
          case '"': result += '"'; i++; break;
          default: result += s[i]; break;
        }
      } else {
        result += s[i];
      }
    }
    return result;
  };

  // Parse JSON vocabulary
  size_t pos = 0;
  while ((pos = content.find("\"", pos)) != std::string::npos) {
    size_t token_start = pos + 1;
    size_t token_end = content.find("\"", token_start);

    while (token_end != std::string::npos && token_end > 0 && content[token_end - 1] == '\\') {
      token_end = content.find("\"", token_end + 1);
    }

    if (token_end == std::string::npos) break;

    std::string token = content.substr(token_start, token_end - token_start);

    size_t colon_pos = content.find(":", token_end);
    if (colon_pos == std::string::npos) break;

    size_t num_start = content.find_first_of("-0123456789", colon_pos);
    if (num_start == std::string::npos) break;

    size_t num_end = content.find_first_not_of("0123456789", num_start + 1);
    if (num_end == std::string::npos) num_end = content.length();

    std::string num_str = content.substr(num_start, num_end - num_start);
    int id = std::stoi(num_str);

    std::string unescaped = unescape_json(token);

    token_to_id_[unescaped] = id;
    id_to_token_[id] = unescaped;

    // Track special tokens
    if (unescaped.size() > 2 && unescaped[0] == '<' && unescaped.back() == '>') {
      added_tokens_.insert(unescaped);
    }

    pos = num_end;
  }

  std::cout << "Loaded vocabulary with " << token_to_id_.size() << " tokens" << std::endl;

  // Load BPE merges
  std::ifstream merges_file(merges_path);
  if (!merges_file.is_open()) {
    std::cerr << "Failed to open merges file: " << merges_path << std::endl;
    return false;
  }

  int rank = 0;
  while (std::getline(merges_file, line)) {
    if (line.empty() || line[0] == '#') continue;

    size_t space_pos = line.find(' ');
    if (space_pos != std::string::npos) {
      std::string first = unescape_json(line.substr(0, space_pos));
      std::string second = unescape_json(line.substr(space_pos + 1));
      bpe_ranks_[{first, second}] = rank++;
    }
  }
  merges_file.close();

  std::cout << "Loaded " << bpe_ranks_.size() << " BPE merges" << std::endl;

  vocab_size_ = token_to_id_.size();
  update_special_tokens();
  
  return true;
}

bool Tokenizer::load_from_tokenizer_json(const std::string& json_path) {
  std::cout << "Loading tokenizer from: " << json_path << std::endl;
  
  std::ifstream file(json_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open tokenizer.json: " << json_path << std::endl;
    return false;
  }

  // Read entire file
  std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
  file.close();

  // Helper to unescape JSON unicode sequences
  auto unescape_json = [](const std::string& s) -> std::string {
    std::string result;
    for (size_t i = 0; i < s.length(); ++i) {
      if (s[i] == '\\' && i + 5 < s.length() && s[i + 1] == 'u') {
        std::string hex = s.substr(i + 2, 4);
        int unicode_point = std::stoi(hex, nullptr, 16);
        if (unicode_point < 0x80) {
          result += static_cast<char>(unicode_point);
        } else if (unicode_point < 0x800) {
          result += static_cast<char>(0xC0 | (unicode_point >> 6));
          result += static_cast<char>(0x80 | (unicode_point & 0x3F));
        } else if (unicode_point < 0x10000) {
          result += static_cast<char>(0xE0 | (unicode_point >> 12));
          result += static_cast<char>(0x80 | ((unicode_point >> 6) & 0x3F));
          result += static_cast<char>(0x80 | (unicode_point & 0x3F));
        }
        i += 5;
      } else if (s[i] == '\\' && i + 1 < s.length()) {
        switch (s[i + 1]) {
          case 'n': result += '\n'; i++; break;
          case 't': result += '\t'; i++; break;
          case 'r': result += '\r'; i++; break;
          case '\\': result += '\\'; i++; break;
          case '"': result += '"'; i++; break;
          default: result += s[i]; break;
        }
      } else {
        result += s[i];
      }
    }
    return result;
  };

  // Find "model" section and then "vocab" within it
  size_t model_pos = content.find("\"model\"");
  if (model_pos == std::string::npos) {
    std::cerr << "No 'model' section found in tokenizer.json" << std::endl;
    return false;
  }

  // Find vocab within model section
  size_t vocab_pos = content.find("\"vocab\"", model_pos);
  if (vocab_pos == std::string::npos) {
    std::cerr << "No 'vocab' found in model section" << std::endl;
    return false;
  }

  // Find the opening brace of vocab object
  size_t vocab_start = content.find("{", vocab_pos);
  if (vocab_start == std::string::npos) {
    std::cerr << "Failed to find vocab object start" << std::endl;
    return false;
  }

  // Find matching closing brace (properly handling strings)
  int brace_count = 1;
  size_t vocab_end = vocab_start + 1;
  bool in_string = false;
  while (vocab_end < content.size() && brace_count > 0) {
    char c = content[vocab_end];
    if (in_string) {
      if (c == '\\' && vocab_end + 1 < content.size()) {
        vocab_end += 2;  // Skip escaped character
        continue;
      } else if (c == '"') {
        in_string = false;
      }
    } else {
      if (c == '"') {
        in_string = true;
      } else if (c == '{') {
        brace_count++;
      } else if (c == '}') {
        brace_count--;
      }
    }
    vocab_end++;
  }

  std::string vocab_content = content.substr(vocab_start, vocab_end - vocab_start);

  // Parse vocab: {"token": id, ...} - properly handle escaped characters in tokens
  size_t pos = 0;
  while (pos < vocab_content.size()) {
    // Find opening quote
    pos = vocab_content.find("\"", pos);
    if (pos == std::string::npos) break;
    
    size_t token_start = pos + 1;
    
    // Find closing quote, handling escapes
    size_t token_end = token_start;
    while (token_end < vocab_content.size()) {
      if (vocab_content[token_end] == '\\' && token_end + 1 < vocab_content.size()) {
        token_end += 2;  // Skip escaped character
        continue;
      }
      if (vocab_content[token_end] == '"') {
        break;
      }
      token_end++;
    }
    
    if (token_end >= vocab_content.size()) break;

    std::string token = vocab_content.substr(token_start, token_end - token_start);
    pos = token_end + 1;

    size_t colon_pos = vocab_content.find(":", pos);
    if (colon_pos == std::string::npos) break;

    size_t num_start = vocab_content.find_first_of("-0123456789", colon_pos);
    if (num_start == std::string::npos) break;

    size_t num_end = vocab_content.find_first_not_of("0123456789", num_start + 1);
    if (num_end == std::string::npos) num_end = vocab_content.length();

    std::string num_str = vocab_content.substr(num_start, num_end - num_start);
    int id = std::stoi(num_str);
    
    pos = num_end;

    std::string unescaped = unescape_json(token);
    token_to_id_[unescaped] = id;
    id_to_token_[id] = unescaped;

    pos = num_end;
  }

  std::cout << "Loaded vocabulary with " << token_to_id_.size() << " tokens" << std::endl;

  // Parse added_tokens section for special tokens
  size_t added_tokens_pos = content.find("\"added_tokens\"");
  if (added_tokens_pos != std::string::npos) {
    size_t arr_start = content.find("[", added_tokens_pos);
    if (arr_start != std::string::npos) {
      // Find matching ]
      int bracket_count = 1;
      size_t arr_end = arr_start + 1;
      while (arr_end < content.size() && bracket_count > 0) {
        if (content[arr_end] == '[') bracket_count++;
        else if (content[arr_end] == ']') bracket_count--;
        arr_end++;
      }
      
      std::string added_tokens_content = content.substr(arr_start, arr_end - arr_start);
      
      // Parse each added token object
      size_t obj_pos = 0;
      while ((obj_pos = added_tokens_content.find("{", obj_pos)) != std::string::npos) {
        size_t obj_end = added_tokens_content.find("}", obj_pos);
        if (obj_end == std::string::npos) break;
        
        std::string obj = added_tokens_content.substr(obj_pos, obj_end - obj_pos + 1);
        
        // Extract id
        size_t id_pos = obj.find("\"id\"");
        int token_id = -1;
        if (id_pos != std::string::npos) {
          size_t id_num_start = obj.find_first_of("0123456789", id_pos);
          if (id_num_start != std::string::npos) {
            size_t id_num_end = obj.find_first_not_of("0123456789", id_num_start);
            token_id = std::stoi(obj.substr(id_num_start, id_num_end - id_num_start));
          }
        }
        
        // Extract content
        size_t content_pos = obj.find("\"content\"");
        std::string token_content;
        if (content_pos != std::string::npos) {
          size_t str_start = obj.find("\"", content_pos + 9);
          if (str_start != std::string::npos) {
            str_start++;
            size_t str_end = obj.find("\"", str_start);
            while (str_end != std::string::npos && str_end > 0 && obj[str_end - 1] == '\\') {
              str_end = obj.find("\"", str_end + 1);
            }
            if (str_end != std::string::npos) {
              token_content = unescape_json(obj.substr(str_start, str_end - str_start));
            }
          }
        }
        
        if (token_id >= 0 && !token_content.empty()) {
          token_to_id_[token_content] = token_id;
          id_to_token_[token_id] = token_content;
          added_tokens_.insert(token_content);
        }
        
        obj_pos = obj_end + 1;
      }
    }
    std::cout << "Loaded " << added_tokens_.size() << " special tokens" << std::endl;
  }

  // Find and parse merges
  size_t merges_pos = content.find("\"merges\"", model_pos);
  if (merges_pos != std::string::npos) {
    size_t arr_start = content.find("[", merges_pos);
    if (arr_start != std::string::npos) {
      // Find matching ] properly (handling strings inside)
      int bracket_count = 1;
      size_t arr_end = arr_start + 1;
      bool in_str = false;
      while (arr_end < content.size() && bracket_count > 0) {
        char c = content[arr_end];
        if (in_str) {
          if (c == '\\' && arr_end + 1 < content.size()) {
            arr_end += 2;
            continue;
          } else if (c == '"') {
            in_str = false;
          }
        } else {
          if (c == '"') {
            in_str = true;
          } else if (c == '[') {
            bracket_count++;
          } else if (c == ']') {
            bracket_count--;
          }
        }
        arr_end++;
      }
      
      std::string merges_content = content.substr(arr_start, arr_end - arr_start);
      
      int rank = 0;
      size_t str_pos = 0;
      
      while (str_pos < merges_content.size()) {
        str_pos = merges_content.find("\"", str_pos);
        if (str_pos == std::string::npos) break;
        
        size_t str_start = str_pos + 1;
        size_t str_end = str_start;
        
        // Find closing quote
        while (str_end < merges_content.size()) {
          if (merges_content[str_end] == '\\' && str_end + 1 < merges_content.size()) {
            str_end += 2;
            continue;
          }
          if (merges_content[str_end] == '"') {
            break;
          }
          str_end++;
        }
        
        if (str_end >= merges_content.size()) break;
        
        std::string merge = merges_content.substr(str_start, str_end - str_start);
        str_pos = str_end + 1;
        
        // Split by space
        size_t space_pos = merge.find(' ');
        if (space_pos != std::string::npos) {
          std::string first = unescape_json(merge.substr(0, space_pos));
          std::string second = unescape_json(merge.substr(space_pos + 1));
          bpe_ranks_[{first, second}] = rank++;
        }
      }
      
      std::cout << "Loaded " << bpe_ranks_.size() << " BPE merges" << std::endl;
    }
  }

  vocab_size_ = token_to_id_.size();
  update_special_tokens();
  
  std::cout << "Tokenizer loaded successfully (vocab size: " << vocab_size_ << ")" << std::endl;
  return true;
}

std::set<std::pair<std::string, std::string>> Tokenizer::get_pairs(const std::vector<std::string>& word) {
  std::set<std::pair<std::string, std::string>> pairs;
  for (size_t i = 1; i < word.size(); ++i) {
    pairs.insert({word[i - 1], word[i]});
  }
  return pairs;
}

std::vector<std::string> Tokenizer::bpe(const std::string& token) {
  std::vector<std::string> word;
  size_t i = 0;
  while (i < token.size()) {
    unsigned char c = static_cast<unsigned char>(token[i]);
    int char_len = 1;
    if ((c & 0xF8) == 0xF0) char_len = 4;
    else if ((c & 0xF0) == 0xE0) char_len = 3;
    else if ((c & 0xE0) == 0xC0) char_len = 2;

    word.push_back(token.substr(i, char_len));
    i += char_len;
  }

  if (word.size() <= 1) {
    return word;
  }

  while (true) {
    auto pairs = get_pairs(word);
    if (pairs.empty()) break;

    auto best_pair = pairs.end();
    int best_rank = INT_MAX;
    for (auto it = pairs.begin(); it != pairs.end(); ++it) {
      auto rank_it = bpe_ranks_.find(*it);
      if (rank_it != bpe_ranks_.end() && rank_it->second < best_rank) {
        best_rank = rank_it->second;
        best_pair = it;
      }
    }

    if (best_pair == pairs.end()) break;

    std::string first = best_pair->first;
    std::string second = best_pair->second;
    std::vector<std::string> new_word;

    i = 0;
    while (i < word.size()) {
      if (i < word.size() - 1 && word[i] == first && word[i + 1] == second) {
        new_word.push_back(first + second);
        i += 2;
      } else {
        new_word.push_back(word[i]);
        i++;
      }
    }

    word = new_word;
    if (word.size() == 1) break;
  }

  return word;
}

std::vector<std::string> Tokenizer::split_to_words(const std::string& text) {
  std::vector<std::string> words;
  std::regex pattern(R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+)", std::regex::icase);

  std::sregex_iterator it(text.begin(), text.end(), pattern);
  std::sregex_iterator end;

  while (it != end) {
    words.push_back(it->str());
    ++it;
  }

  return words;
}

std::vector<int> Tokenizer::encode(const std::string& text) {
  std::vector<int> tokens;

  // Collect all special tokens from added_tokens_
  std::vector<std::string> special_tokens(added_tokens_.begin(), added_tokens_.end());
  // Sort by length descending to match longer tokens first
  std::sort(special_tokens.begin(), special_tokens.end(),
            [](const std::string& a, const std::string& b) { return a.size() > b.size(); });

  size_t pos = 0;
  while (pos < text.size()) {
    bool found_special = false;

    for (const auto& special : special_tokens) {
      if (text.compare(pos, special.size(), special) == 0) {
        auto it = token_to_id_.find(special);
        if (it != token_to_id_.end()) {
          tokens.push_back(it->second);
          pos += special.size();
          found_special = true;
          break;
        }
      }
    }

    if (found_special) continue;

    // Find next special token
    size_t next_special = std::string::npos;
    for (const auto& special : special_tokens) {
      size_t found = text.find(special, pos);
      if (found != std::string::npos && found < next_special) {
        next_special = found;
      }
    }

    size_t end_pos = (next_special != std::string::npos) ? next_special : text.size();
    std::string chunk = text.substr(pos, end_pos - pos);

    if (!chunk.empty()) {
      std::vector<std::string> words = split_to_words(chunk);

      for (const auto& word : words) {
        std::string encoded_word;
        for (unsigned char c : word) {
          auto it = byte_encoder_.find(c);
          if (it != byte_encoder_.end()) {
            encoded_word += it->second;
          }
        }

        std::vector<std::string> bpe_tokens = bpe(encoded_word);

        for (const auto& bpe_token : bpe_tokens) {
          auto it = token_to_id_.find(bpe_token);
          if (it != token_to_id_.end()) {
            tokens.push_back(it->second);
          } else {
            for (char c : bpe_token) {
              std::string s(1, c);
              auto char_it = token_to_id_.find(s);
              if (char_it != token_to_id_.end()) {
                tokens.push_back(char_it->second);
              }
            }
          }
        }
      }
    }

    pos = end_pos;
  }

  return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
  std::string text;

  for (int token_id : tokens) {
    auto it = id_to_token_.find(token_id);
    if (it != id_to_token_.end()) {
      text += it->second;
    }
  }

  std::string result;
  size_t i = 0;
  while (i < text.size()) {
    unsigned char c = static_cast<unsigned char>(text[i]);
    int char_len = 1;
    if ((c & 0xF8) == 0xF0) char_len = 4;
    else if ((c & 0xF0) == 0xE0) char_len = 3;
    else if ((c & 0xE0) == 0xC0) char_len = 2;

    std::string utf8_char = text.substr(i, char_len);

    auto it = byte_decoder_.find(utf8_char);
    if (it != byte_decoder_.end()) {
      result += static_cast<char>(it->second);
    } else {
      result += utf8_char;
    }

    i += char_len;
  }

  return result;
}

// ============================================================================
// Chat Templates
// ============================================================================

std::string Tokenizer::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_generation_prompt) {
  
  switch (config_.chat_template) {
    case ChatTemplateType::CHATML:
      return apply_chatml(messages, add_generation_prompt);
    case ChatTemplateType::LLAMA3:
      return apply_llama3(messages, add_generation_prompt);
    case ChatTemplateType::LLAMA2:
      return apply_llama2(messages, add_generation_prompt);
    case ChatTemplateType::MISTRAL:
      return apply_mistral(messages, add_generation_prompt);
    default:
      // No template, just concatenate
      std::string result;
      for (const auto& msg : messages) {
        result += msg.second + "\n";
      }
      return result;
  }
}

std::string Tokenizer::apply_chatml(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_gen_prompt) {
  
  std::string result;

  // Add default system message if not present
  bool has_system = !messages.empty() && messages[0].first == "system";
  if (!has_system && !config_.default_system_prompt.empty()) {
    result = "<|im_start|>system\n" + config_.default_system_prompt + "<|im_end|>\n";
  }

  for (const auto& msg : messages) {
    result += "<|im_start|>" + msg.first + "\n" + msg.second + "<|im_end|>\n";
  }

  if (add_gen_prompt) {
    result += "<|im_start|>assistant\n";
  }

  return result;
}

std::string Tokenizer::apply_llama3(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_gen_prompt) {
  
  std::string result = "<|begin_of_text|>";

  // Add default system message if not present
  bool has_system = !messages.empty() && messages[0].first == "system";
  if (!has_system && !config_.default_system_prompt.empty()) {
    result += "<|start_header_id|>system<|end_header_id|>\n\n";
    result += config_.default_system_prompt + "<|eot_id|>";
  }

  for (const auto& msg : messages) {
    result += "<|start_header_id|>" + msg.first + "<|end_header_id|>\n\n";
    result += msg.second + "<|eot_id|>";
  }

  if (add_gen_prompt) {
    result += "<|start_header_id|>assistant<|end_header_id|>\n\n";
  }

  return result;
}

std::string Tokenizer::apply_llama2(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_gen_prompt) {
  
  std::string result = "<s>";
  std::string system_msg;

  // Extract system message
  auto it = messages.begin();
  if (it != messages.end() && it->first == "system") {
    system_msg = it->second;
    ++it;
  } else if (!config_.default_system_prompt.empty()) {
    system_msg = config_.default_system_prompt;
  }

  bool first = true;
  while (it != messages.end()) {
    if (it->first == "user") {
      result += "[INST] ";
      if (first && !system_msg.empty()) {
        result += "<<SYS>>\n" + system_msg + "\n<</SYS>>\n\n";
        first = false;
      }
      result += it->second + " [/INST]";
    } else if (it->first == "assistant") {
      result += " " + it->second + " </s><s>";
    }
    ++it;
  }

  if (add_gen_prompt) {
    result += "[INST] ";
  }

  return result;
}

std::string Tokenizer::apply_mistral(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_gen_prompt) {
  
  std::string result = "<s>";

  for (const auto& msg : messages) {
    if (msg.first == "user") {
      result += "[INST] " + msg.second + " [/INST]";
    } else if (msg.first == "assistant") {
      result += msg.second + "</s> ";
    }
  }

  if (add_gen_prompt) {
    // Mistral expects next turn to start with [INST]
  }

  return result;
}

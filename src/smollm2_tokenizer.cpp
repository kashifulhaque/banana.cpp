#include "smollm2_tokenizer.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <regex>
#include <set>
#include <climits>

SmolLM2Tokenizer::SmolLM2Tokenizer() {
    init_byte_encoder();
}

SmolLM2Tokenizer::~SmolLM2Tokenizer() {
}

// Initialize the byte-to-unicode mapping (same as GPT-2)
void SmolLM2Tokenizer::init_byte_encoder() {
    // Build byte encoder mapping
    std::vector<int> bs;
    for (int i = 33; i <= 126; ++i) bs.push_back(i);   // ! to ~
    for (int i = 161; i <= 172; ++i) bs.push_back(i);  // Latin-1 supplement
    for (int i = 174; i <= 255; ++i) bs.push_back(i);
    
    std::vector<int> cs = bs;
    int n = 0;
    
    // Fill in gaps with unused Unicode points
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }
    
    // Create the mappings
    for (size_t i = 0; i < bs.size(); ++i) {
        int byte_val = bs[i];
        int unicode_point = cs[i];
        
        // Convert unicode point to UTF-8 string
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

bool SmolLM2Tokenizer::download_and_load(const std::string& save_dir) {
    // Create directory
    std::string mkdir_cmd = "mkdir -p " + save_dir;
    system(mkdir_cmd.c_str());
    
    std::string vocab_path = save_dir + "/vocab.json";
    std::string merges_path = save_dir + "/merges.txt";
    
    // Check if files exist
    std::ifstream vocab_check(vocab_path);
    std::ifstream merges_check(merges_path);
    
    if (!vocab_check.good() || !merges_check.good()) {
        std::cout << "Downloading SmolLM2 tokenizer files..." << std::endl;
        
        // Download vocab.json
        std::string vocab_cmd = "curl -L https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/vocab.json -o " + vocab_path + " 2>/dev/null";
        int ret1 = system(vocab_cmd.c_str());
        
        // Download merges.txt
        std::string merges_cmd = "curl -L https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/merges.txt -o " + merges_path + " 2>/dev/null";
        int ret2 = system(merges_cmd.c_str());
        
        if (ret1 != 0 || ret2 != 0) {
            std::cerr << "Failed to download tokenizer files" << std::endl;
            return false;
        }
        std::cout << "Downloaded tokenizer files successfully" << std::endl;
    }
    
    return load(save_dir);
}

bool SmolLM2Tokenizer::load(const std::string& tokenizer_dir) {
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
            if (s[i] == '\\' && i + 5 < s.length() && s[i+1] == 'u') {
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
        
        // Handle escaped quotes
        while (token_end != std::string::npos && token_end > 0 && content[token_end - 1] == '\\') {
            token_end = content.find("\"", token_end + 1);
        }
        
        if (token_end == std::string::npos) break;
        
        std::string token = content.substr(token_start, token_end - token_start);
        
        // Find the colon and number
        size_t colon_pos = content.find(":", token_end);
        if (colon_pos == std::string::npos) break;
        
        size_t num_start = content.find_first_of("-0123456789", colon_pos);
        if (num_start == std::string::npos) break;
        
        size_t num_end = content.find_first_not_of("0123456789", num_start + 1);
        if (num_end == std::string::npos) num_end = content.length();
        
        std::string num_str = content.substr(num_start, num_end - num_start);
        int id = std::stoi(num_str);
        
        // Unescape the token
        std::string unescaped = unescape_json(token);
        
        token_to_id_[unescaped] = id;
        id_to_token_[id] = unescaped;
        
        pos = num_end;
    }
    
    std::cout << "Loaded vocabulary with " << token_to_id_.size() << " tokens" << std::endl;
    
    // Update special token IDs from vocabulary
    if (token_to_id_.count("<|endoftext|>")) endoftext_id_ = token_to_id_["<|endoftext|>"];
    if (token_to_id_.count("<|im_start|>")) {
        im_start_id_ = token_to_id_["<|im_start|>"];
        bos_token_id_ = im_start_id_;
    }
    if (token_to_id_.count("<|im_end|>")) {
        im_end_id_ = token_to_id_["<|im_end|>"];
        eos_token_id_ = im_end_id_;
        pad_token_id_ = im_end_id_;
    }
    
    // Load BPE merges
    std::ifstream merges_file(merges_path);
    if (!merges_file.is_open()) {
        std::cerr << "Failed to open merges file: " << merges_path << std::endl;
        return false;
    }
    
    int rank = 0;
    while (std::getline(merges_file, line)) {
        // Skip header line
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
    return true;
}

std::set<std::pair<std::string, std::string>> SmolLM2Tokenizer::get_pairs(const std::vector<std::string>& word) {
    std::set<std::pair<std::string, std::string>> pairs;
    for (size_t i = 1; i < word.size(); ++i) {
        pairs.insert({word[i-1], word[i]});
    }
    return pairs;
}

std::vector<std::string> SmolLM2Tokenizer::bpe(const std::string& token) {
    // Convert token to vector of characters
    std::vector<std::string> word;
    size_t i = 0;
    while (i < token.size()) {
        // Handle UTF-8 characters
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
        
        // Find the pair with lowest rank
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
        
        // Merge the best pair
        std::string first = best_pair->first;
        std::string second = best_pair->second;
        std::vector<std::string> new_word;
        
        i = 0;
        while (i < word.size()) {
            if (i < word.size() - 1 && word[i] == first && word[i+1] == second) {
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

std::vector<std::string> SmolLM2Tokenizer::split_to_words(const std::string& text) {
    // GPT-2 pattern for splitting text
    std::vector<std::string> words;
    
    // Pattern: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+
    // Simplified version for ASCII:
    std::regex pattern(R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+)", std::regex::icase);
    
    std::sregex_iterator it(text.begin(), text.end(), pattern);
    std::sregex_iterator end;
    
    while (it != end) {
        words.push_back(it->str());
        ++it;
    }
    
    return words;
}

std::vector<int> SmolLM2Tokenizer::encode(const std::string& text) {
    std::vector<int> tokens;
    
    // Check for special tokens first
    std::string remaining = text;
    
    // Handle special tokens
    std::vector<std::string> special_tokens = {
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
        "<repo_name>", "<reponame>", "<file_sep>", "<filename>",
        "<gh_stars>", "<issue_start>", "<issue_comment>", "<issue_closed>",
        "<jupyter_start>", "<jupyter_text>", "<jupyter_code>",
        "<jupyter_output>", "<jupyter_script>", "<empty_output>"
    };
    
    size_t pos = 0;
    while (pos < remaining.size()) {
        bool found_special = false;
        
        // Check for special tokens at current position
        for (const auto& special : special_tokens) {
            if (remaining.compare(pos, special.size(), special) == 0) {
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
        
        // Find the next special token
        size_t next_special = std::string::npos;
        for (const auto& special : special_tokens) {
            size_t found = remaining.find(special, pos);
            if (found != std::string::npos && found < next_special) {
                next_special = found;
            }
        }
        
        // Process regular text up to next special token
        size_t end_pos = (next_special != std::string::npos) ? next_special : remaining.size();
        std::string chunk = remaining.substr(pos, end_pos - pos);
        
        if (!chunk.empty()) {
            // Split into words
            std::vector<std::string> words = split_to_words(chunk);
            
            for (const auto& word : words) {
                // Convert to byte-encoded form
                std::string encoded_word;
                for (unsigned char c : word) {
                    auto it = byte_encoder_.find(c);
                    if (it != byte_encoder_.end()) {
                        encoded_word += it->second;
                    }
                }
                
                // Apply BPE
                std::vector<std::string> bpe_tokens = bpe(encoded_word);
                
                // Convert to IDs
                for (const auto& bpe_token : bpe_tokens) {
                    auto it = token_to_id_.find(bpe_token);
                    if (it != token_to_id_.end()) {
                        tokens.push_back(it->second);
                    } else {
                        // Unknown token - try character by character
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

std::string SmolLM2Tokenizer::decode(const std::vector<int>& tokens) {
    std::string text;
    
    for (int token_id : tokens) {
        auto it = id_to_token_.find(token_id);
        if (it != id_to_token_.end()) {
            text += it->second;
        }
    }
    
    // Convert from byte-encoded form back to regular text
    std::string result;
    size_t i = 0;
    while (i < text.size()) {
        // Handle UTF-8 characters
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

std::string SmolLM2Tokenizer::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_generation_prompt) {
    
    std::string result;
    
    // Add default system message if first message is not system
    bool has_system = !messages.empty() && messages[0].first == "system";
    if (!has_system) {
        result = "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n";
    }
    
    // Add each message
    for (const auto& msg : messages) {
        const std::string& role = msg.first;
        const std::string& content = msg.second;
        
        result += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
    }
    
    // Add generation prompt
    if (add_generation_prompt) {
        result += "<|im_start|>assistant\n";
    }
    
    return result;
}

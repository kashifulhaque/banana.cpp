#include "tokenizer.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <regex>
#include <set>
#include <climits>

Tokenizer::Tokenizer() {
}

Tokenizer::~Tokenizer() {
}

// GPT-2 uses byte-to-unicode mapping to handle any byte
static std::unordered_map<int, std::string> bytes_to_unicode() {
    std::unordered_map<int, std::string> mapping;
    
    // Direct mappings for visible ASCII and some ranges
    std::vector<int> bs;
    for (int i = 33; i <= 126; ++i) bs.push_back(i);  // ! to ~
    for (int i = 161; i <= 172; ++i) bs.push_back(i);
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
    
    // Create the mapping
    for (size_t i = 0; i < bs.size(); ++i) {
        char buf[5] = {0};
        int unicode_point = cs[i];
        
        // Convert unicode point to UTF-8
        if (unicode_point < 0x80) {
            buf[0] = unicode_point;
        } else if (unicode_point < 0x800) {
            buf[0] = 0xC0 | (unicode_point >> 6);
            buf[1] = 0x80 | (unicode_point & 0x3F);
        } else if (unicode_point < 0x10000) {
            buf[0] = 0xE0 | (unicode_point >> 12);
            buf[1] = 0x80 | ((unicode_point >> 6) & 0x3F);
            buf[2] = 0x80 | (unicode_point & 0x3F);
        } else {
            buf[0] = 0xF0 | (unicode_point >> 18);
            buf[1] = 0x80 | ((unicode_point >> 12) & 0x3F);
            buf[2] = 0x80 | ((unicode_point >> 6) & 0x3F);
            buf[3] = 0x80 | (unicode_point & 0x3F);
        }
        
        mapping[bs[i]] = std::string(buf);
    }
    
    return mapping;
}

static std::unordered_map<std::string, int> unicode_to_bytes() {
    auto b2u = bytes_to_unicode();
    std::unordered_map<std::string, int> mapping;
    for (const auto& pair : b2u) {
        mapping[pair.second] = pair.first;
    }
    return mapping;
}

bool Tokenizer::download_and_load() {
    // Check if vocab files exist
    std::ifstream vocab_check("weights/vocab.json");
    std::ifstream merges_check("weights/merges.txt");
    
    if (!vocab_check.good() || !merges_check.good()) {
        std::cout << "Downloading GPT-2 tokenizer files..." << std::endl;
        
        // Download vocab.json
        int ret1 = system("curl -L https://huggingface.co/gpt2/resolve/main/vocab.json -o weights/vocab.json 2>/dev/null");
        
        // Download merges.txt
        int ret2 = system("curl -L https://huggingface.co/gpt2/resolve/main/merges.txt -o weights/merges.txt 2>/dev/null");
        
        if (ret1 != 0 || ret2 != 0) {
            std::cerr << "Failed to download tokenizer files" << std::endl;
            return false;
        }
        std::cout << "Downloaded tokenizer files successfully" << std::endl;
    }
    
    return load_vocab("weights/vocab.json");
}

bool Tokenizer::load_vocab(const std::string& vocab_path) {
    std::ifstream vocab_file(vocab_path);
    if (!vocab_file.is_open()) {
        std::cerr << "Failed to open vocabulary file: " << vocab_path << std::endl;
        return false;
    }
    
    // Simple JSON parser for vocab (format: {"token": id, ...})
    std::string line, content;
    while (std::getline(vocab_file, line)) {
        content += line;
    }
    vocab_file.close();
    
    // Helper to unescape JSON unicode sequences like \u0120
    auto unescape_json = [](const std::string& s) -> std::string {
        std::string result;
        for (size_t i = 0; i < s.length(); ++i) {
            if (s[i] == '\\' && i + 5 < s.length() && s[i+1] == 'u') {
                // Parse \uXXXX
                std::string hex = s.substr(i + 2, 4);
                int unicode_point = std::stoi(hex, nullptr, 16);
                
                // Convert unicode point to UTF-8
                if (unicode_point < 0x80) {
                    result += (char)unicode_point;
                } else if (unicode_point < 0x800) {
                    result += (char)(0xC0 | (unicode_point >> 6));
                    result += (char)(0x80 | (unicode_point & 0x3F));
                } else if (unicode_point < 0x10000) {
                    result += (char)(0xE0 | (unicode_point >> 12));
                    result += (char)(0x80 | ((unicode_point >> 6) & 0x3F));
                    result += (char)(0x80 | (unicode_point & 0x3F));
                }
                i += 5;  // Skip \uXXXX
            } else if (s[i] == '\\' && i + 1 < s.length()) {
                // Handle other escapes
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
    
    // Parse JSON manually (simplified)
    size_t pos = 0;
    while ((pos = content.find("\"", pos)) != std::string::npos) {
        size_t start = pos + 1;
        size_t end = content.find("\"", start);
        if (end == std::string::npos) break;
        
        std::string token = unescape_json(content.substr(start, end - start));
        
        // Find the colon and number
        size_t colon = content.find(":", end);
        if (colon == std::string::npos) break;
        
        size_t num_start = colon + 1;
        while (num_start < content.length() && std::isspace(content[num_start])) num_start++;
        
        size_t num_end = num_start;
        while (num_end < content.length() && std::isdigit(content[num_end])) num_end++;
        
        if (num_end > num_start) {
            int id = std::stoi(content.substr(num_start, num_end - num_start));
            token_to_id[token] = id;
            id_to_token[id] = token;
        }
        
        // Move past the ID we just parsed
        pos = num_end;
    }
    
    // Load merges
    std::ifstream merges_file("weights/merges.txt");
    if (merges_file.is_open()) {
        std::string merge_line;
        std::getline(merges_file, merge_line); // Skip header
        
        int rank = 0;
        while (std::getline(merges_file, merge_line)) {
            if (merge_line.empty()) continue;
            
            size_t space = merge_line.find(' ');
            if (space != std::string::npos) {
                std::string first = merge_line.substr(0, space);
                std::string second = merge_line.substr(space + 1);
                bpe_ranks[{first, second}] = rank++;
            }
        }
        merges_file.close();
    }
    
    std::cout << "Loaded " << token_to_id.size() << " tokens and " 
              << bpe_ranks.size() << " merges" << std::endl;
    
    return true;
}

std::vector<std::string> Tokenizer::split_to_words(const std::string& text) {
    // GPT-2's regex pattern for splitting
    // Pattern: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    // Simplified version that works with basic regex
    std::vector<std::string> words;
    
    std::regex pattern(
        "'s|'t|'re|'ve|'m|'ll|'d"
        "| ?[a-zA-Z]+"
        "| ?[0-9]+"
        "| ?[^\\s\\w]+"
        "|\\s+(?!\\S)"
        "|\\s+"
    );
    
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    auto words_end = std::sregex_iterator();
    
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        words.push_back(i->str());
    }
    
    return words;
}

std::string Tokenizer::bytes_to_unicode(unsigned char c) {
    static auto mapping = ::bytes_to_unicode();
    return mapping[c];
}

std::vector<std::string> Tokenizer::bpe(const std::string& token) {
    if (token.length() == 0) return {};
    
    // Convert token to individual character strings
    std::vector<std::string> word;
    
    // Handle UTF-8 properly - each character might be multiple bytes
    for (size_t i = 0; i < token.length(); ) {
        int char_len = 1;
        unsigned char c = token[i];
        
        // Determine UTF-8 character length
        if ((c & 0x80) == 0) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;
        
        word.push_back(token.substr(i, char_len));
        i += char_len;
    }
    
    if (word.size() <= 1) return word;
    
    // Apply BPE merges iteratively
    while (true) {
        int min_rank = INT_MAX;
        int merge_idx = -1;
        
        // Find the pair with lowest rank (earliest in merge order)
        for (size_t i = 0; i < word.size() - 1; ++i) {
            auto pair = std::make_pair(word[i], word[i + 1]);
            auto it = bpe_ranks.find(pair);
            if (it != bpe_ranks.end() && it->second < min_rank) {
                min_rank = it->second;
                merge_idx = i;
            }
        }
        
        if (merge_idx == -1) break;  // No more merges possible
        
        // Merge the pair
        std::vector<std::string> new_word;
        for (size_t i = 0; i < word.size(); ++i) {
            if ((int)i == merge_idx) {
                new_word.push_back(word[i] + word[i + 1]);
                i++;  // Skip next element as it's been merged
            } else {
                new_word.push_back(word[i]);
            }
        }
        word = new_word;
        
        if (word.size() == 1) break;
    }
    
    return word;
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> tokens;
    
    if (token_to_id.empty()) {
        std::cerr << "Tokenizer not loaded!" << std::endl;
        return tokens;
    }
    
    // Get byte-to-unicode mapping
    static auto byte_encoder = ::bytes_to_unicode();
    
    // Split text into words using GPT-2's pattern
    auto words = split_to_words(text);
    
    for (const auto& word : words) {
        // Convert word to byte-level representation
        std::string byte_encoded;
        for (unsigned char c : word) {
            byte_encoded += byte_encoder[c];
        }
        
        // Apply BPE
        auto bpe_tokens = bpe(byte_encoded);
        
        // Convert BPE tokens to IDs
        for (const auto& bpe_token : bpe_tokens) {
            auto it = token_to_id.find(bpe_token);
            if (it != token_to_id.end()) {
                tokens.push_back(it->second);
            }
        }
    }
    
    return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
    // Get unicode-to-byte mapping
    static auto byte_decoder = ::unicode_to_bytes();
    
    std::string text;
    for (int token_id : tokens) {
        auto it = id_to_token.find(token_id);
        if (it != id_to_token.end()) {
            text += it->second;
        }
    }
    
    // Decode from byte-level representation back to UTF-8
    std::string decoded;
    for (size_t i = 0; i < text.length(); ) {
        int char_len = 1;
        unsigned char c = text[i];
        
        // Determine UTF-8 character length
        if ((c & 0x80) == 0) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;
        
        std::string utf8_char = text.substr(i, char_len);
        auto byte_it = byte_decoder.find(utf8_char);
        if (byte_it != byte_decoder.end()) {
            decoded += (char)byte_it->second;
        } else {
            // Passthrough if not in mapping
            decoded += utf8_char;
        }
        
        i += char_len;
    }
    
    return decoded;
}

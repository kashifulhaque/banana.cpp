#include <vector>
#include <string>
#include <unordered_map>

class Tokenizer {
public:
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<int, std::string> id_to_token;

    void load(const std::string& vocab_path) {
        // TODO: load vocab json/file here
        // for testing, manually insert something 'like token_to_id["hello"] = 50256;'
    }

    std::vector<int> encode(const std::string& text) {
        // TODO: implement bpe split logic
        // return dummy for compilation
        return {15496, 11}; // "Hello", ","
    }

    std::string decode(const std::vector<int>& tokens) {
        std::string text = "";
        for(int t : tokens) {
            if(id_to_token.count(t)) {
                text += id_to_token[t];
            }
        }

        return text;
    }
};

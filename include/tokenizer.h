#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

class Tokenizer {
public:
  Tokenizer();
  ~Tokenizer();

  bool load_vocab(const std::string &vocab_path);
  bool download_and_load();

  std::vector<int> encode(const std::string &text);
  std::string decode(const std::vector<int> &tokens);

private:
  std::unordered_map<std::string, int> token_to_id;
  std::unordered_map<int, std::string> id_to_token;
  std::map<std::pair<std::string, std::string>, int> bpe_ranks;

  std::vector<std::string> split_to_words(const std::string &text);
  std::string bytes_to_unicode(unsigned char c);
  std::vector<std::string> bpe(const std::string &token);
};

#endif // TOKENIZER_H

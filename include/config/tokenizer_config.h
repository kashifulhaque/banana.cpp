#ifndef CONFIG_TOKENIZER_CONFIG_H
#define CONFIG_TOKENIZER_CONFIG_H

#include <string>

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

inline TokenizerConfig llama2() {
  TokenizerConfig cfg;
  cfg.model_name = "Llama-2";
  cfg.chat_template = ChatTemplateType::LLAMA2;
  cfg.bos_token = "<s>";
  cfg.eos_token = "</s>";
  cfg.default_system_prompt = "You are a helpful assistant.";
  return cfg;
}

} // namespace TokenizerPresets

#endif // CONFIG_TOKENIZER_CONFIG_H

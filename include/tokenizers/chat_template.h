#ifndef TOKENIZERS_CHAT_TEMPLATE_H
#define TOKENIZERS_CHAT_TEMPLATE_H

#include "config/tokenizer_config.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tokenizers {

/// Message type for chat conversations
using Message = std::pair<std::string, std::string>;  // (role, content)
using Messages = std::vector<Message>;

/// Abstract base class for chat templates
class ChatTemplate {
public:
  virtual ~ChatTemplate() = default;
  
  /// Apply chat template to messages
  /// @param messages Vector of (role, content) pairs
  /// @param add_generation_prompt Whether to add assistant prompt at the end
  /// @return Formatted string ready for tokenization
  virtual std::string apply(
    const Messages& messages,
    bool add_generation_prompt = true
  ) const = 0;
  
  /// Get/set default system prompt
  const std::string& default_system_prompt() const { return default_system_prompt_; }
  void set_default_system_prompt(const std::string& prompt) { default_system_prompt_ = prompt; }

protected:
  std::string default_system_prompt_;
};

/// ChatML template (SmolLM, Qwen)
/// Format: <|im_start|>role\ncontent<|im_end|>
class ChatMLTemplate : public ChatTemplate {
public:
  ChatMLTemplate() = default;
  explicit ChatMLTemplate(const std::string& default_system_prompt) {
    default_system_prompt_ = default_system_prompt;
  }
  
  std::string apply(
    const Messages& messages,
    bool add_generation_prompt = true
  ) const override;
};

/// Llama 3 template
/// Format: <|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
class Llama3Template : public ChatTemplate {
public:
  Llama3Template() = default;
  explicit Llama3Template(const std::string& default_system_prompt) {
    default_system_prompt_ = default_system_prompt;
  }
  
  std::string apply(
    const Messages& messages,
    bool add_generation_prompt = true
  ) const override;
};

/// Llama 2 template
/// Format: <s>[INST] <<SYS>>\nsystem\n<</SYS>>\n\nuser [/INST] assistant </s><s>
class Llama2Template : public ChatTemplate {
public:
  Llama2Template() = default;
  explicit Llama2Template(const std::string& default_system_prompt) {
    default_system_prompt_ = default_system_prompt;
  }
  
  std::string apply(
    const Messages& messages,
    bool add_generation_prompt = true
  ) const override;
};

/// Mistral template
/// Format: <s>[INST] user [/INST] assistant</s>
class MistralTemplate : public ChatTemplate {
public:
  MistralTemplate() = default;
  
  std::string apply(
    const Messages& messages,
    bool add_generation_prompt = true
  ) const override;
};

/// No template - just concatenate messages
class NoTemplate : public ChatTemplate {
public:
  std::string apply(
    const Messages& messages,
    bool add_generation_prompt = true
  ) const override;
};

/// Factory function to create chat template from type
std::unique_ptr<ChatTemplate> create_chat_template(
  ChatTemplateType type,
  const std::string& default_system_prompt = ""
);

} // namespace tokenizers

#endif // TOKENIZERS_CHAT_TEMPLATE_H

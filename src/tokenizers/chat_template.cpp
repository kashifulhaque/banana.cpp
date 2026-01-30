#include "tokenizers/chat_template.h"

namespace tokenizers {

// ============================================================================
// ChatML Template
// ============================================================================

std::string ChatMLTemplate::apply(const Messages& messages, 
                                   bool add_generation_prompt) const {
  std::string result;

  // Add default system message if not present
  bool has_system = !messages.empty() && messages[0].first == "system";
  if (!has_system && !default_system_prompt_.empty()) {
    result = "<|im_start|>system\n" + default_system_prompt_ + "<|im_end|>\n";
  }

  for (const auto& msg : messages) {
    result += "<|im_start|>" + msg.first + "\n" + msg.second + "<|im_end|>\n";
  }

  if (add_generation_prompt) {
    result += "<|im_start|>assistant\n";
  }

  return result;
}

// ============================================================================
// Llama 3 Template
// ============================================================================

std::string Llama3Template::apply(const Messages& messages, 
                                   bool add_generation_prompt) const {
  std::string result = "<|begin_of_text|>";

  // Add default system message if not present
  bool has_system = !messages.empty() && messages[0].first == "system";
  if (!has_system && !default_system_prompt_.empty()) {
    result += "<|start_header_id|>system<|end_header_id|>\n\n";
    result += default_system_prompt_ + "<|eot_id|>";
  }

  for (const auto& msg : messages) {
    result += "<|start_header_id|>" + msg.first + "<|end_header_id|>\n\n";
    result += msg.second + "<|eot_id|>";
  }

  if (add_generation_prompt) {
    result += "<|start_header_id|>assistant<|end_header_id|>\n\n";
  }

  return result;
}

// ============================================================================
// Llama 2 Template
// ============================================================================

std::string Llama2Template::apply(const Messages& messages, 
                                   bool add_generation_prompt) const {
  std::string result = "<s>";
  std::string system_msg;

  // Extract system message
  auto it = messages.begin();
  if (it != messages.end() && it->first == "system") {
    system_msg = it->second;
    ++it;
  } else if (!default_system_prompt_.empty()) {
    system_msg = default_system_prompt_;
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

  if (add_generation_prompt) {
    result += "[INST] ";
  }

  return result;
}

// ============================================================================
// Mistral Template
// ============================================================================

std::string MistralTemplate::apply(const Messages& messages, 
                                    bool /*add_generation_prompt*/) const {
  std::string result = "<s>";

  for (const auto& msg : messages) {
    if (msg.first == "user") {
      result += "[INST] " + msg.second + " [/INST]";
    } else if (msg.first == "assistant") {
      result += msg.second + "</s> ";
    }
  }

  return result;
}

// ============================================================================
// No Template
// ============================================================================

std::string NoTemplate::apply(const Messages& messages, 
                               bool /*add_generation_prompt*/) const {
  std::string result;
  for (const auto& msg : messages) {
    result += msg.second + "\n";
  }
  return result;
}

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<ChatTemplate> create_chat_template(ChatTemplateType type,
                                                    const std::string& default_system_prompt) {
  switch (type) {
    case ChatTemplateType::CHATML:
      return std::make_unique<ChatMLTemplate>(default_system_prompt);
    case ChatTemplateType::LLAMA3:
      return std::make_unique<Llama3Template>(default_system_prompt);
    case ChatTemplateType::LLAMA2:
      return std::make_unique<Llama2Template>(default_system_prompt);
    case ChatTemplateType::MISTRAL:
      return std::make_unique<MistralTemplate>();
    case ChatTemplateType::NONE:
    default:
      return std::make_unique<NoTemplate>();
  }
}

} // namespace tokenizers

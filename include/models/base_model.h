#ifndef MODELS_BASE_MODEL_H
#define MODELS_BASE_MODEL_H

#include "config/llm_config.h"
#include "core/tensor.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace models {

/// Callback type for streaming token generation
/// Called with each new token ID as it's generated
/// Return false to stop generation early
using TokenCallback = std::function<bool(int token_id)>;

/// Abstract base class for all language models
class BaseModel {
public:
  virtual ~BaseModel() = default;

  /// Forward pass: given input token IDs, return logits for next token
  virtual Tensor forward(const std::vector<int>& input_ids) = 0;

  /// Generate text given a prompt
  /// @param input_ids Tokenized input prompt
  /// @param config Sampling configuration
  /// @param token_callback Optional callback for streaming output
  /// @return Full sequence including input and generated tokens
  virtual std::vector<int> generate(
    const std::vector<int>& input_ids,
    const SamplingConfig& config = SamplingConfig(),
    TokenCallback token_callback = nullptr
  ) = 0;

  /// Access configuration
  virtual const LLMConfig& get_config() const = 0;

  /// Check if a token is an EOS token
  virtual bool is_eos_token(int token_id) const = 0;
};

} // namespace models

#endif // MODELS_BASE_MODEL_H

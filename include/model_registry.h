#ifndef MODEL_REGISTRY_H
#define MODEL_REGISTRY_H

#include "llm_config.h"
#include "tokenizer.h"
#include <string>
#include <optional>
#include <vector>

/// Result of model detection
struct ModelInfo {
  LLMConfig config;
  TokenizerConfig tokenizer_config;
  std::string model_id;         // HuggingFace model ID
  std::string weights_path;     // Path to weights
  std::string tokenizer_path;   // Path to tokenizer files
};

/// Registry for supported models with auto-detection
class ModelRegistry {
public:
  /// Detect model type from config.json in a directory
  static std::optional<ModelInfo> detect_from_directory(const std::string& model_dir);
  
  /// Detect model from HuggingFace model ID
  static std::optional<ModelInfo> detect_from_model_id(const std::string& model_id);
  
  /// Get config for a known model by name
  static std::optional<ModelInfo> get_model_info(const std::string& model_name);
  
  /// List all registered models
  static std::vector<std::string> list_models();
  
  /// Parse config.json and return LLMConfig
  static std::optional<LLMConfig> parse_config_json(const std::string& config_path);
  
  /// Infer tokenizer config from model config
  static TokenizerConfig infer_tokenizer_config(const LLMConfig& config);

private:
  /// Map model type string to architecture
  static ModelArchitecture string_to_architecture(const std::string& model_type);
  
  /// Map architecture string to activation type
  static ActivationType string_to_activation(const std::string& activation);
};

/// Known HuggingFace model IDs
namespace KnownModels {
  // SmolLM family
  constexpr const char* SMOLLM2_135M = "HuggingFaceTB/SmolLM2-135M-Instruct";
  constexpr const char* SMOLLM2_360M = "HuggingFaceTB/SmolLM2-360M-Instruct";
  constexpr const char* SMOLLM2_1_7B = "HuggingFaceTB/SmolLM2-1.7B-Instruct";
  
  // Llama family
  constexpr const char* LLAMA3_2_1B = "meta-llama/Llama-3.2-1B-Instruct";
  constexpr const char* LLAMA3_2_3B = "meta-llama/Llama-3.2-3B-Instruct";
  
  // Qwen family
  constexpr const char* QWEN2_5_0_5B = "Qwen/Qwen2.5-0.5B-Instruct";
  constexpr const char* QWEN2_5_1_5B = "Qwen/Qwen2.5-1.5B-Instruct";
  constexpr const char* QWEN3_0_6B = "Qwen/Qwen3-0.6B";
}

#endif // MODEL_REGISTRY_H

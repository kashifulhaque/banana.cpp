#include "registry/model_registry.h"
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <algorithm>

// Simple JSON value extraction (avoiding external dependencies)
namespace {

std::string extract_string(const std::string& json, const std::string& key) {
  std::string pattern = "\"" + key + "\"\\s*:\\s*\"([^\"]*)\"";
  std::regex re(pattern);
  std::smatch match;
  if (std::regex_search(json, match, re)) {
    return match[1].str();
  }
  return "";
}

int extract_int(const std::string& json, const std::string& key, int default_val = 0) {
  std::string pattern = "\"" + key + "\"\\s*:\\s*(-?[0-9]+)";
  std::regex re(pattern);
  std::smatch match;
  if (std::regex_search(json, match, re)) {
    try {
      return std::stoi(match[1].str());
    } catch (...) {}
  }
  return default_val;
}

float extract_float(const std::string& json, const std::string& key, float default_val = 0.0f) {
  std::string pattern = "\"" + key + "\"\\s*:\\s*(-?[0-9]+\\.?[0-9]*(?:[eE][-+]?[0-9]+)?)";
  std::regex re(pattern);
  std::smatch match;
  if (std::regex_search(json, match, re)) {
    try {
      return std::stof(match[1].str());
    } catch (...) {}
  }
  return default_val;
}

bool extract_bool(const std::string& json, const std::string& key, bool default_val = false) {
  std::string pattern = "\"" + key + "\"\\s*:\\s*(true|false)";
  std::regex re(pattern);
  std::smatch match;
  if (std::regex_search(json, match, re)) {
    return match[1].str() == "true";
  }
  return default_val;
}

std::vector<int> extract_int_array(const std::string& json, const std::string& key) {
  std::vector<int> result;
  std::string pattern = "\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]";
  std::regex re(pattern);
  std::smatch match;
  if (std::regex_search(json, match, re)) {
    std::string arr = match[1].str();
    std::regex num_re("-?[0-9]+");
    std::sregex_iterator it(arr.begin(), arr.end(), num_re);
    std::sregex_iterator end;
    while (it != end) {
      try {
        result.push_back(std::stoi(it->str()));
      } catch (...) {}
      ++it;
    }
  }
  return result;
}

std::string read_file_contents(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) return "";
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

} // anonymous namespace

ModelArchitecture ModelRegistry::string_to_architecture(const std::string& model_type) {
  std::string lower = model_type;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  
  if (lower.find("llama") != std::string::npos) return ModelArchitecture::LLAMA;
  if (lower.find("qwen") != std::string::npos) return ModelArchitecture::QWEN;
  if (lower.find("mistral") != std::string::npos) return ModelArchitecture::MISTRAL;
  if (lower.find("phi") != std::string::npos) return ModelArchitecture::PHI;
  
  // SmolLM uses LLaMA architecture
  if (lower.find("smol") != std::string::npos) return ModelArchitecture::LLAMA;
  
  return ModelArchitecture::LLAMA; // Default to LLaMA-style
}

ActivationType ModelRegistry::string_to_activation(const std::string& activation) {
  std::string lower = activation;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  
  if (lower == "silu" || lower == "swish") return ActivationType::SILU;
  if (lower == "gelu") return ActivationType::GELU;
  if (lower == "gelu_new" || lower == "gelu_fast") return ActivationType::GELU_NEW;
  if (lower == "relu") return ActivationType::RELU;
  
  return ActivationType::SILU;
}

std::optional<LLMConfig> ModelRegistry::parse_config_json(const std::string& config_path) {
  std::string json = read_file_contents(config_path);
  if (json.empty()) {
    std::cerr << "Failed to read config file: " << config_path << std::endl;
    return std::nullopt;
  }
  
  LLMConfig config;
  
  // Model type
  config.model_type = extract_string(json, "model_type");
  if (config.model_type.empty()) {
    // Try architectures array
    std::string arch_pattern = "\"architectures\"\\s*:\\s*\\[\\s*\"([^\"]*)\"";
    std::regex arch_re(arch_pattern);
    std::smatch match;
    if (std::regex_search(json, match, arch_re)) {
      config.model_type = match[1].str();
    }
  }
  
  config.architecture = string_to_architecture(config.model_type);
  
  // Core dimensions
  config.vocab_size = extract_int(json, "vocab_size", 32000);
  config.hidden_size = extract_int(json, "hidden_size", 2048);
  config.intermediate_size = extract_int(json, "intermediate_size", 8192);
  config.num_hidden_layers = extract_int(json, "num_hidden_layers", 32);
  config.num_attention_heads = extract_int(json, "num_attention_heads", 32);
  config.num_key_value_heads = extract_int(json, "num_key_value_heads", config.num_attention_heads);
  config.max_position_embeddings = extract_int(json, "max_position_embeddings", 2048);
  
  // Normalization
  config.rms_norm_eps = extract_float(json, "rms_norm_eps", 1e-5f);
  config.layer_norm_eps = extract_float(json, "layer_norm_eps", 1e-5f);
  
  // RoPE
  config.rope_theta = extract_float(json, "rope_theta", 10000.0f);
  
  // Check for rope_scaling
  if (json.find("rope_scaling") != std::string::npos) {
    config.rope_scaling_type = extract_string(json, "type");
    config.rope_scaling_factor = extract_float(json, "factor", 1.0f);
  }
  
  // Activation
  std::string hidden_act = extract_string(json, "hidden_act");
  if (!hidden_act.empty()) {
    config.hidden_act = string_to_activation(hidden_act);
  }
  
  // Bias settings (important for Qwen)
  config.use_qkv_bias = extract_bool(json, "attention_bias", false);
  if (!config.use_qkv_bias) {
    config.use_qkv_bias = extract_bool(json, "add_qkv_bias", false);
  }
  
  // Embeddings
  config.tie_word_embeddings = extract_bool(json, "tie_word_embeddings", false);
  
  // Special tokens
  config.bos_token_id = extract_int(json, "bos_token_id", 1);
  config.eos_token_id = extract_int(json, "eos_token_id", 2);
  config.pad_token_id = extract_int(json, "pad_token_id", 0);
  
  // Multiple EOS tokens
  config.eos_token_ids = extract_int_array(json, "eos_token_id");
  if (config.eos_token_ids.empty()) {
    config.eos_token_ids.push_back(config.eos_token_id);
  }
  
  // Model name from _name_or_path if available
  config.model_name = extract_string(json, "_name_or_path");
  if (config.model_name.empty()) {
    config.model_name = config.model_type;
  }
  
  return config;
}

TokenizerConfig ModelRegistry::infer_tokenizer_config(const LLMConfig& config) {
  TokenizerConfig tok_config;
  tok_config.model_name = config.model_name;
  
  switch (config.architecture) {
    case ModelArchitecture::LLAMA:
      // Check if it's SmolLM (uses ChatML) or Llama 3 (uses Llama3 template)
      if (config.model_name.find("SmolLM") != std::string::npos ||
          config.model_name.find("smollm") != std::string::npos) {
        tok_config = TokenizerPresets::smollm();
      } else if (config.model_name.find("Llama-3") != std::string::npos ||
                 config.model_name.find("llama-3") != std::string::npos ||
                 config.bos_token_id > 100000) {
        tok_config = TokenizerPresets::llama3();
      } else {
        tok_config = TokenizerPresets::llama2();
      }
      break;
      
    case ModelArchitecture::QWEN:
      tok_config = TokenizerPresets::qwen();
      break;
      
    case ModelArchitecture::MISTRAL:
      tok_config = TokenizerPresets::mistral();
      break;
      
    default:
      tok_config.chat_template = ChatTemplateType::CHATML;
      break;
  }
  
  return tok_config;
}

std::optional<ModelInfo> ModelRegistry::detect_from_directory(const std::string& model_dir) {
  std::string config_path = model_dir + "/config.json";
  
  auto config_opt = parse_config_json(config_path);
  if (!config_opt) {
    return std::nullopt;
  }
  
  ModelInfo info;
  info.config = *config_opt;
  info.tokenizer_config = infer_tokenizer_config(info.config);
  info.weights_path = model_dir;
  info.tokenizer_path = model_dir;
  
  return info;
}

std::optional<ModelInfo> ModelRegistry::detect_from_model_id(const std::string& model_id) {
  // First check if it's a known model
  auto known = get_model_info(model_id);
  if (known) {
    return known;
  }
  
  std::cerr << "Unknown model: " << model_id << std::endl;
  std::cerr << "Please provide a local path with config.json" << std::endl;
  return std::nullopt;
}

std::optional<ModelInfo> ModelRegistry::get_model_info(const std::string& model_name) {
  // Normalize name for comparison
  std::string lower = model_name;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  
  ModelInfo info;
  
  // SmolLM2 family
  if (lower.find("smollm2-135m") != std::string::npos || 
      model_name == KnownModels::SMOLLM2_135M) {
    info.config = ModelPresets::smollm2_135m();
    info.tokenizer_config = TokenizerPresets::smollm();
    info.model_id = KnownModels::SMOLLM2_135M;
    return info;
  }
  
  if (lower.find("smollm2-360m") != std::string::npos || 
      model_name == KnownModels::SMOLLM2_360M) {
    info.config = ModelPresets::smollm2_360m();
    info.tokenizer_config = TokenizerPresets::smollm();
    info.model_id = KnownModels::SMOLLM2_360M;
    return info;
  }
  
  if (lower.find("smollm2-1.7b") != std::string::npos || 
      lower.find("smollm2-1_7b") != std::string::npos ||
      model_name == KnownModels::SMOLLM2_1_7B) {
    info.config = ModelPresets::smollm2_1_7b();
    info.tokenizer_config = TokenizerPresets::smollm();
    info.model_id = KnownModels::SMOLLM2_1_7B;
    return info;
  }
  
  // Llama 3.2 family
  if (lower.find("llama-3.2-1b") != std::string::npos ||
      lower.find("llama3.2-1b") != std::string::npos ||
      model_name == KnownModels::LLAMA3_2_1B) {
    info.config = ModelPresets::llama3_2_1b();
    info.tokenizer_config = TokenizerPresets::llama3();
    info.model_id = KnownModels::LLAMA3_2_1B;
    return info;
  }
  
  if (lower.find("llama-3.2-3b") != std::string::npos ||
      lower.find("llama3.2-3b") != std::string::npos ||
      model_name == KnownModels::LLAMA3_2_3B) {
    info.config = ModelPresets::llama3_2_3b();
    info.tokenizer_config = TokenizerPresets::llama3();
    info.model_id = KnownModels::LLAMA3_2_3B;
    return info;
  }
  
  // Qwen family
  if (lower.find("qwen2.5-0.5b") != std::string::npos ||
      lower.find("qwen2_5-0_5b") != std::string::npos ||
      model_name == KnownModels::QWEN2_5_0_5B) {
    info.config = ModelPresets::qwen2_5_0_5b();
    info.tokenizer_config = TokenizerPresets::qwen();
    info.model_id = KnownModels::QWEN2_5_0_5B;
    return info;
  }
  
  if (lower.find("qwen3-0.6b") != std::string::npos ||
      lower.find("qwen3-0_6b") != std::string::npos ||
      model_name == KnownModels::QWEN3_0_6B) {
    info.config = ModelPresets::qwen3_0_6b();
    info.tokenizer_config = TokenizerPresets::qwen();
    info.model_id = KnownModels::QWEN3_0_6B;
    return info;
  }
  
  return std::nullopt;
}

std::vector<std::string> ModelRegistry::list_models() {
  return {
    "SmolLM2-135M-Instruct",
    "SmolLM2-360M-Instruct", 
    "SmolLM2-1.7B-Instruct",
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Qwen2.5-0.5B-Instruct",
    "Qwen3-0.6B"
  };
}

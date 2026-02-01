/**
 * 🍌 banana.cpp
 * 
 * A pure C++ implementation supporting multiple small language models:
 * - SmolLM2 (135M, 360M, 1.7B)
 * - Llama 3.2 (1B, 3B)
 * - Qwen 2.5, Qwen 3
 * 
 * Usage:
 *   ./llm_inference [options]
 * 
 * Options:
 *   --model <name>      Model name or path (e.g., smollm2-360m, llama-3.2-1b)
 *   --weights <path>    Path to weights directory
 *   --tokenizer <path>  Path to tokenizer directory
 *   --prompt <text>     Input prompt for generation
 *   --max-tokens <n>    Maximum new tokens to generate (default: 256)
 *   --temperature <f>   Sampling temperature (default: 0.7)
 *   --top-k <n>         Top-k sampling (default: 50)
 *   --top-p <f>         Top-p (nucleus) sampling (default: 0.9)
 *   --interactive       Interactive chat mode
 *   --list-models       List all supported models
 */

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "config/llm_config.h"
#include "models/llm_model.h"
#include "utils/model_loader.h"
#include "registry/model_registry.h"
#include "tokenizers/tokenizer.h"
#include "utils/weight_downloader.h"

void print_usage(const char* program_name) {
  std::cout << "🍌 banana.cpp" << std::endl;
  std::cout << "Supported models: SmolLM2, Llama 3.2, Qwen 2.5/3" << std::endl << std::endl;
  std::cout << "Usage: " << program_name << " [options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  --model <name>            Model name (e.g., smollm2-360m, llama-3.2-1b, qwen3-0.6b)" << std::endl;
  std::cout << "  --weights <path>          Path to weights directory" << std::endl;
  std::cout << "  --tokenizer <path>        Path to tokenizer directory" << std::endl;
  std::cout << "  --prompt <text>           Input prompt for generation" << std::endl;
  std::cout << "  --max-tokens <n>          Maximum new tokens to generate (default: 256)" << std::endl;
  std::cout << "  --temperature <f>         Sampling temperature (default: 0.7)" << std::endl;
  std::cout << "  --top-k <n>               Top-k sampling (default: 50)" << std::endl;
  std::cout << "  --top-p <f>               Top-p (nucleus) sampling (default: 0.9)" << std::endl;
  std::cout << "  --repetition-penalty <f>  Repetition penalty (default: 1.1)" << std::endl;
  std::cout << "  --interactive             Interactive chat mode" << std::endl;
  std::cout << "  --download                Download model weights from HuggingFace" << std::endl;
  std::cout << "  --precision <type>        Weight precision: fp32, fp16, bf16 (default: bf16)" << std::endl;
  std::cout << "  --list-models             List all supported models" << std::endl;
  std::cout << "  --help                    Show this help message" << std::endl;
}

void list_supported_models() {
  std::cout << "Supported Models:" << std::endl;
  
  std::cout << "SmolLM2 Family:" << std::endl;
  std::cout << "  smollm2-135m    SmolLM2-135M-Instruct (576 hidden, 30 layers)" << std::endl;
  std::cout << "  smollm2-360m    SmolLM2-360M-Instruct (960 hidden, 32 layers)" << std::endl;
  std::cout << "  smollm2-1.7b    SmolLM2-1.7B-Instruct (2048 hidden, 24 layers)" << std::endl << std::endl;
  
  std::cout << "Llama 3.2 Family:" << std::endl;
  std::cout << "  llama-3.2-1b    Llama-3.2-1B-Instruct (2048 hidden, 16 layers)" << std::endl;
  std::cout << "  llama-3.2-3b    Llama-3.2-3B-Instruct (3072 hidden, 28 layers)" << std::endl << std::endl;
  
  std::cout << "Qwen Family:" << std::endl;
  std::cout << "  qwen2.5-0.5b    Qwen2.5-0.5B-Instruct (896 hidden, 24 layers)" << std::endl;
  std::cout << "  qwen3-0.6b      Qwen3-0.6B (1024 hidden, 28 layers)" << std::endl;
}

struct Config {
  std::string model_name = "smollm2-360m";
  std::string weights_path;
  std::string tokenizer_path;
  std::string prompt;
  int max_tokens = 256;
  float temperature = 0.7f;
  int top_k = 50;
  float top_p = 0.9f;
  float repetition_penalty = 1.1f;
  bool interactive = false;
  bool download = false;
  bool list_models = false;
  WeightPrecision precision = WeightPrecision::BF16;
};

Config parse_args(int argc, char* argv[]) {
  Config config;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      exit(0);
    } else if (arg == "--list-models") {
      config.list_models = true;
    } else if (arg == "--model" && i + 1 < argc) {
      config.model_name = argv[++i];
    } else if (arg == "--weights" && i + 1 < argc) {
      config.weights_path = argv[++i];
    } else if (arg == "--tokenizer" && i + 1 < argc) {
      config.tokenizer_path = argv[++i];
    } else if (arg == "--prompt" && i + 1 < argc) {
      config.prompt = argv[++i];
    } else if (arg == "--max-tokens" && i + 1 < argc) {
      config.max_tokens = std::stoi(argv[++i]);
    } else if (arg == "--temperature" && i + 1 < argc) {
      config.temperature = std::stof(argv[++i]);
    } else if (arg == "--top-k" && i + 1 < argc) {
      config.top_k = std::stoi(argv[++i]);
    } else if (arg == "--top-p" && i + 1 < argc) {
      config.top_p = std::stof(argv[++i]);
    } else if (arg == "--repetition-penalty" && i + 1 < argc) {
      config.repetition_penalty = std::stof(argv[++i]);
    } else if (arg == "--interactive") {
      config.interactive = true;
    } else if (arg == "--download") {
      config.download = true;
    } else if (arg == "--precision" && i + 1 < argc) {
      std::string prec = argv[++i];
      if (prec == "fp32") {
        config.precision = WeightPrecision::FP32;
      } else if (prec == "fp16") {
        config.precision = WeightPrecision::FP16;
      } else if (prec == "bf16") {
        config.precision = WeightPrecision::BF16;
      } else {
        std::cerr << "Unknown precision: " << prec << " (use fp32, fp16, or bf16)" << std::endl;
        exit(1);
      }
    }
  }

  return config;
}

void run_single_prompt(
  models::LLMModel& model,
  tokenizers::Tokenizer& tokenizer,
  const std::string& prompt,
  const SamplingConfig& sampling_config
) {
  // Apply chat template
  std::vector<std::pair<std::string, std::string>> messages = {{"user", prompt}};
  std::string formatted = tokenizer.apply_chat_template(messages, true);

  std::cout << "\n[Input]\n" << formatted << std::endl;

  // Tokenize
  auto start_encode = std::chrono::high_resolution_clock::now();
  std::vector<int> input_ids = tokenizer.encode(formatted);
  auto end_encode = std::chrono::high_resolution_clock::now();

  // Streaming response
  std::cout << "[Generating...]" << std::endl;
  auto start_gen = std::chrono::high_resolution_clock::now();

  int tokens_generated = 0;
  auto token_callback = [&tokenizer, &tokens_generated](int token_id) -> bool {
    std::vector<int> single_token = {token_id};
    std::string token_text = tokenizer.decode(single_token);
    std::cout << token_text << std::flush;
    tokens_generated++;
    return true;
  };

  std::vector<int> output_ids = model.generate(input_ids, sampling_config, token_callback);

  auto end_gen = std::chrono::high_resolution_clock::now();

  auto encode_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_encode - start_encode).count();
  auto gen_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_gen - start_gen).count();
  int new_tokens = output_ids.size() - input_ids.size();
  float tokens_per_sec = (gen_time > 0) ? (new_tokens * 1000.0f / gen_time) : 0;

  std::cout << "\n\n[Stats]" << std::endl;
  std::cout << "  Input tokens: " << input_ids.size() << std::endl;
  std::cout << "  Output tokens: " << output_ids.size() << " (+" << new_tokens << " new)" << std::endl;
  std::cout << "  Encode time: " << encode_time << " ms" << std::endl;
  std::cout << "  Generate time: " << gen_time << " ms" << std::endl;
  std::cout << "  Speed: " << tokens_per_sec << " tokens/sec" << std::endl;
}

void run_interactive(
  models::LLMModel& model,
  tokenizers::Tokenizer& tokenizer,
  const SamplingConfig& sampling_config,
  const std::string& model_name
) {
  std::cout << model_name << " Interactive Chat" << std::endl;
  std::cout << "Type your message and press Enter." << std::endl;
  std::cout << "Commands:" << std::endl;
  std::cout << "  /quit  - Exit the chat" << std::endl;
  std::cout << "  /clear - Clear chat history" << std::endl;
  std::cout << "  /temp <f> - Set temperature" << std::endl;

  std::vector<std::pair<std::string, std::string>> history;
  SamplingConfig current_config = sampling_config;

  while (true) {
    std::cout << "You: ";
    std::string input;
    std::getline(std::cin, input);
    if (input.empty()) continue;

    if (input == "/quit" || input == "/exit") {
      std::cout << "Goodbye!" << std::endl;
      break;
    } else if (input == "/clear") {
      history.clear();
      std::cout << "[Chat history cleared]" << std::endl << std::endl;
      continue;
    } else if (input.substr(0, 6) == "/temp ") {
      try {
        current_config.temperature = std::stof(input.substr(6));
        std::cout << "[Temperature set to " << current_config.temperature << "]" << std::endl << std::endl;
      } catch (...) {
        std::cout << "[Invalid temperature value]" << std::endl << std::endl;
      }
      continue;
    }

    history.push_back({"user", input});
    std::string formatted = tokenizer.apply_chat_template(history, true);
    std::vector<int> input_ids = tokenizer.encode(formatted);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> output_ids = model.generate(input_ids, current_config);
    auto end = std::chrono::high_resolution_clock::now();

    // Decode only the new tokens
    std::vector<int> new_tokens(output_ids.begin() + input_ids.size(), output_ids.end());
    std::string response = tokenizer.decode(new_tokens);

    std::vector<std::string> eos_markers = { "<|im_end|>", "<|eot_id|>", "</s>", "<|end|>", "<|endoftext|>" };
    for (const auto& marker : eos_markers) {
      size_t end_pos = response.find(marker);
      if (end_pos != std::string::npos) {
        response = response.substr(0, end_pos);
        break;
      }
    }

    history.push_back({"assistant", response});

    auto gen_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    float tokens_per_sec = (gen_time > 0) ? (new_tokens.size() * 1000.0f / gen_time) : 0;

    std::cout << "\nAssistant: " << response << std::endl;
    std::cout << "[" << new_tokens.size() << " tokens, " << tokens_per_sec << " tok/s]" << std::endl << std::endl;
  }
}

int main(int argc, char* argv[]) {
  Config config = parse_args(argc, argv);

  if (config.list_models) {
    list_supported_models();
    return 0;
  }

  std::cout << "🍌 banana.cpp" << std::endl;

  std::optional<ModelInfo> model_info;
  
  if (!config.weights_path.empty()) {
    model_info = ModelRegistry::detect_from_directory(config.weights_path);
  }
  
  if (!model_info) {
    model_info = ModelRegistry::get_model_info(config.model_name);
  }
  
  if (!model_info) {
    std::cerr << "Unknown model: " << config.model_name << std::endl;
    std::cerr << "Use --list-models to see supported models." << std::endl;
    return 1;
  }

  std::cout << "Model: " << model_info->config.model_name << std::endl;
  std::cout << "Architecture: " << model_info->config.model_type << std::endl << std::endl;

  if (config.weights_path.empty()) {
    std::string model_id = model_info->model_id;
    size_t slash = model_id.find('/');
    std::string model_dir = (slash != std::string::npos) ? model_id.substr(slash + 1) : model_id;
    config.weights_path = "weights/" + model_dir;
  }
  if (config.tokenizer_path.empty()) {
    config.tokenizer_path = config.weights_path;
  }

  if (config.download) {
    std::cout << "Downloading model: " << model_info->model_id << std::endl << std::endl;
    WeightDownloader downloader(model_info->model_id);
    downloader.set_precision(config.precision);

    if (!downloader.download_and_export(config.weights_path)) {
      std::cerr << "Failed to download and export weights!" << std::endl;
      return 1;
    }

    std::cout << "\nWeights downloaded successfully to: " << config.weights_path << std::endl << std::endl;
    std::cout << "You can now run inference without --download flag." << std::endl;
    return 0;
  }

  WeightDownloader downloader(model_info->model_id);
  if (!downloader.weights_exist(config.weights_path)) {
    std::cout << "Model weights not found at: " << config.weights_path << std::endl << std::endl;
    std::cout << "Would you like to download them now? [Y/n]: ";

    std::string response;
    std::getline(std::cin, response);

    if (response.empty() || response[0] == 'Y' || response[0] == 'y') {
      downloader.set_precision(config.precision);
      if (!downloader.download_and_export(config.weights_path)) {
        std::cerr << "Failed to download weights!" << std::endl;
        return 1;
      }
      std::cout << "\n";
    } else {
      std::cerr << "Cannot proceed without model weights." << std::endl;
      std::cerr << "Run with --download flag to download weights." << std::endl;
      return 1;
    }
  }

  std::cout << "Loading tokenizer from: " << config.tokenizer_path << std::endl;
  tokenizers::Tokenizer tokenizer;
  tokenizer.set_config(model_info->tokenizer_config);
  
  if (!tokenizer.load(config.tokenizer_path)) {
    std::cerr << "Trying to download tokenizer..." << std::endl;
    if (!tokenizer.download_and_load(model_info->model_id, "weights")) {
      std::cerr << "Failed to load tokenizer!" << std::endl;
      return 1;
    }
  }
  std::cout << "Tokenizer loaded successfully (vocab size: " << tokenizer.vocab_size() << ")" << std::endl << std::endl;

  std::vector<std::string> weight_candidates = {
    config.weights_path + "/model.bin",
    config.weights_path + "/smollm2_weights.bin",
    config.weights_path + "/weights.bin",
    config.weights_path
  };
  
  std::string weights_file;
  for (const auto& candidate : weight_candidates) {
    std::ifstream test(candidate);
    if (test.good()) {
      weights_file = candidate;
      break;
    }
  }
  
  if (weights_file.empty()) {
    std::cerr << "Model weights not found. Tried:" << std::endl;
    for (const auto& candidate : weight_candidates) {
      std::cerr << "  - " << candidate << std::endl;
    }
    std::cerr << "Run with --download flag to download weights from HuggingFace." << std::endl;
    return 1;
  }
  
  std::cout << "Loading model weights from: " << weights_file << std::endl;
  ModelLoader loader(weights_file);
  if (!loader.load()) {
    std::cerr << "Failed to load model weights!" << std::endl;
    std::cerr << "Run with --download flag to download weights from HuggingFace." << std::endl;
    return 1;
  }
  std::cout << std::endl;

  auto model = models::create_model(loader, model_info->config);

  SamplingConfig sampling_config;
  sampling_config.max_new_tokens = config.max_tokens;
  sampling_config.temperature = config.temperature;
  sampling_config.top_k = config.top_k;
  sampling_config.top_p = config.top_p;
  sampling_config.repetition_penalty = config.repetition_penalty;

  if (config.interactive) {
    run_interactive(*model, tokenizer, sampling_config, model_info->config.model_name);
  } else if (!config.prompt.empty()) {
    run_single_prompt(*model, tokenizer, config.prompt, sampling_config);
  } else {
    // Default demo prompt
    std::string demo_prompt = "What is the capital of France?";
    std::cout << "No prompt specified, using demo: \"" << demo_prompt << "\"\n";
    run_single_prompt(*model, tokenizer, demo_prompt, sampling_config);
  }

  return 0;
}

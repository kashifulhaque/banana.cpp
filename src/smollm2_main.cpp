/**
 * SmolLM2 Inference Engine
 * 
 * A pure C++ implementation of the SmolLM2-360M-Instruct model.
 * 
 * Usage:
 *   ./smollm2_inference [options]
 * 
 * Options:
 *   --weights <path>    Path to weights file (default: weights/smollm2/smollm2_weights.bin)
 *   --tokenizer <path>  Path to tokenizer directory (default: weights/smollm2)
 *   --prompt <text>     Input prompt for generation
 *   --max-tokens <n>    Maximum new tokens to generate (default: 256)
 *   --temperature <f>   Sampling temperature (default: 0.7)
 *   --top-k <n>         Top-k sampling (default: 50)
 *   --top-p <f>         Top-p (nucleus) sampling (default: 0.9)
 *   --interactive       Interactive chat mode
 */

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>

#include "smollm2.h"
#include "smollm2_tokenizer.h"
#include "model_loader.h"

void print_usage(const char* program_name) {
    std::cout << "SmolLM2 Inference Engine\n";
    std::cout << "========================\n\n";
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --weights <path>     Path to weights file (default: weights/smollm2/smollm2_weights.bin)\n";
    std::cout << "  --tokenizer <path>   Path to tokenizer directory (default: weights/smollm2)\n";
    std::cout << "  --prompt <text>      Input prompt for generation\n";
    std::cout << "  --max-tokens <n>     Maximum new tokens to generate (default: 256)\n";
    std::cout << "  --temperature <f>    Sampling temperature (default: 0.7)\n";
    std::cout << "  --top-k <n>          Top-k sampling (default: 50)\n";
    std::cout << "  --top-p <f>          Top-p (nucleus) sampling (default: 0.9)\n";
    std::cout << "  --repetition-penalty <f>  Repetition penalty (default: 1.1)\n";
    std::cout << "  --interactive        Interactive chat mode\n";
    std::cout << "  --help               Show this help message\n";
}

struct Config {
    std::string weights_path = "weights/smollm2/smollm2_weights.bin";
    std::string tokenizer_path = "weights/smollm2";
    std::string prompt;
    int max_tokens = 256;
    float temperature = 0.7f;
    int top_k = 50;
    float top_p = 0.9f;
    float repetition_penalty = 1.1f;
    bool interactive = false;
};

Config parse_args(int argc, char* argv[]) {
    Config config;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
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
        }
    }
    
    return config;
}

void run_single_prompt(SmolLM2Model& model, SmolLM2Tokenizer& tokenizer, 
                       const std::string& prompt, const SmolLM2SamplingConfig& sampling_config) {
    /// apply chat template
    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", prompt}
    };
    std::string formatted = tokenizer.apply_chat_template(messages, true);
    
    std::cout << "\n[Input]\n" << formatted << "\n";
    
    /// tokenize
    auto start_encode = std::chrono::high_resolution_clock::now();
    std::vector<int> input_ids = tokenizer.encode(formatted);
    auto end_encode = std::chrono::high_resolution_clock::now();
    
    std::cout << "[Tokens] " << input_ids.size() << " input tokens\n";
    std::cout << "[Token IDs] ";
    for (size_t i = 0; i < std::min(input_ids.size(), size_t(20)); ++i) {
        std::cout << input_ids[i] << " ";
    }
    if (input_ids.size() > 20) std::cout << "...";
    std::cout << "\n\n";
    
    /// generate
    std::cout << "[Generating...]\n";
    auto start_gen = std::chrono::high_resolution_clock::now();
    
    std::vector<int> output_ids = model.generate(input_ids, sampling_config);
    
    auto end_gen = std::chrono::high_resolution_clock::now();
    
    /// decode
    std::string output = tokenizer.decode(output_ids);
    
    /// calculate timing
    auto encode_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_encode - start_encode).count();
    auto gen_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_gen - start_gen).count();
    int new_tokens = output_ids.size() - input_ids.size();
    float tokens_per_sec = (gen_time > 0) ? (new_tokens * 1000.0f / gen_time) : 0;
    
    std::cout << "\n[Output]\n" << output << "\n\n";
    std::cout << "[Stats]\n";
    std::cout << "  Input tokens: " << input_ids.size() << "\n";
    std::cout << "  Output tokens: " << output_ids.size() << " (+" << new_tokens << " new)\n";
    std::cout << "  Encode time: " << encode_time << " ms\n";
    std::cout << "  Generate time: " << gen_time << " ms\n";
    std::cout << "  Speed: " << tokens_per_sec << " tokens/sec\n";
}

void run_interactive(SmolLM2Model& model, SmolLM2Tokenizer& tokenizer,
                     const SmolLM2SamplingConfig& sampling_config) {
    std::cout << "\n====================================\n";
    std::cout << "SmolLM2 Interactive Chat\n";
    std::cout << "====================================\n";
    std::cout << "Type your message and press Enter.\n";
    std::cout << "Commands:\n";
    std::cout << "  /quit  - Exit the chat\n";
    std::cout << "  /clear - Clear chat history\n";
    std::cout << "  /temp <f> - Set temperature\n";
    std::cout << "====================================\n\n";
    
    std::vector<std::pair<std::string, std::string>> history;
    SmolLM2SamplingConfig current_config = sampling_config;
    
    while (true) {
        std::cout << "You: ";
        std::string input;
        std::getline(std::cin, input);
        
        if (input.empty()) continue;
        
        /// handle commands
        if (input == "/quit" || input == "/exit") {
            std::cout << "Goodbye!\n";
            break;
        } else if (input == "/clear") {
            history.clear();
            std::cout << "[Chat history cleared]\n\n";
            continue;
        } else if (input.substr(0, 6) == "/temp ") {
            try {
                current_config.temperature = std::stof(input.substr(6));
                std::cout << "[Temperature set to " << current_config.temperature << "]\n\n";
            } catch (...) {
                std::cout << "[Invalid temperature value]\n\n";
            }
            continue;
        }
        
        /// add user message to history
        history.push_back({"user", input});
        
        /// apply chat template
        std::string formatted = tokenizer.apply_chat_template(history, true);
        
        /// tokenize
        std::vector<int> input_ids = tokenizer.encode(formatted);
        
        /// generate
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<int> output_ids = model.generate(input_ids, current_config);
        auto end = std::chrono::high_resolution_clock::now();
        
        /// decode only the new tokens
        std::vector<int> new_tokens(output_ids.begin() + input_ids.size(), output_ids.end());
        std::string response = tokenizer.decode(new_tokens);
        
        /// clean up response (remove trailing special tokens)
        size_t end_pos = response.find("<|im_end|>");
        if (end_pos != std::string::npos) {
            response = response.substr(0, end_pos);
        }
        
        /// add assistant response to history
        history.push_back({"assistant", response});
        
        /// calculate timing
        auto gen_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        float tokens_per_sec = (gen_time > 0) ? (new_tokens.size() * 1000.0f / gen_time) : 0;
        
        std::cout << "\nSmolLM2: " << response << "\n";
        std::cout << "[" << new_tokens.size() << " tokens, " << tokens_per_sec << " tok/s]\n\n";
    }
}

int main(int argc, char* argv[]) {
    Config config = parse_args(argc, argv);
    
    std::cout << "====================================\n";
    std::cout << "SmolLM2 Inference Engine\n";
    std::cout << "====================================\n\n";
    
    /// load tokenizer
    std::cout << "Loading tokenizer from: " << config.tokenizer_path << "\n";
    SmolLM2Tokenizer tokenizer;
    if (!tokenizer.load(config.tokenizer_path)) {
        std::cerr << "Trying to download tokenizer...\n";
        if (!tokenizer.download_and_load(config.tokenizer_path)) {
            std::cerr << "Failed to load tokenizer!\n";
            return 1;
        }
    }
    std::cout << "Tokenizer loaded successfully (vocab size: " << tokenizer.vocab_size() << ")\n\n";
    
    /// load model weights
    std::cout << "Loading model weights from: " << config.weights_path << "\n";
    ModelLoader loader(config.weights_path);
    if (!loader.load()) {
        std::cerr << "Failed to load model weights!\n";
        std::cerr << "Please run: python scripts/export_smollm2_weights.py\n";
        return 1;
    }
    std::cout << "\n";
    
    /// initialize model
    SmolLM2Model model(loader);
    
    /// setup sampling config
    SmolLM2SamplingConfig sampling_config;
    sampling_config.max_new_tokens = config.max_tokens;
    sampling_config.temperature = config.temperature;
    sampling_config.top_k = config.top_k;
    sampling_config.top_p = config.top_p;
    sampling_config.repetition_penalty = config.repetition_penalty;
    
    if (config.interactive) {
        run_interactive(model, tokenizer, sampling_config);
    } else if (!config.prompt.empty()) {
        run_single_prompt(model, tokenizer, config.prompt, sampling_config);
    } else {
        /// default demo prompt
        std::string demo_prompt = "What is the capital of France?";
        std::cout << "No prompt specified, using demo: \"" << demo_prompt << "\"\n";
        run_single_prompt(model, tokenizer, demo_prompt, sampling_config);
    }
    
    return 0;
}

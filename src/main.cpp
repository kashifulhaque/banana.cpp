#include <iostream>
#include <string>
#include "model_loader.h"
#include "tokenizer.h"
#include "tensor.h"
#include "ops.h"
#include "gpt2.h"

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "   GPT-2 Inference Engine (C++)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Load model weights
    std::cout << "Step 1: Loading model weights..." << std::endl;
    ModelLoader model_loader("weights/gpt2_weights.bin");
    if (!model_loader.load()) {
        std::cerr << "Failed to load model weights. Please run:" << std::endl;
        std::cerr << "  python scripts/export_weights.py" << std::endl;
        return 1;
    }
    std::cout << std::endl;
    
    // Initialize tokenizer
    std::cout << "Step 2: Loading tokenizer..." << std::endl;
    Tokenizer tokenizer;
    if (!tokenizer.download_and_load()) {
        std::cerr << "Failed to load tokenizer" << std::endl;
        return 1;
    }
    std::cout << std::endl;
    
    // Initialize GPT-2 model
    std::cout << "Step 3: Initializing GPT-2 model..." << std::endl;
    GPT2Model model(model_loader);
    std::cout << std::endl;
    
    // Get prompt from command line or use default
    std::string prompt = "Hello, my name is";
    if (argc > 1) {
        prompt = "";
        for (int i = 1; i < argc; ++i) {
            if (i > 1) prompt += " ";
            prompt += argv[i];
        }
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Encode prompt
    std::cout << "Step 4: Encoding prompt..." << std::endl;
    std::vector<int> input_ids = tokenizer.encode(prompt);
    
    std::cout << "Encoded tokens: [";
    for (size_t i = 0; i < input_ids.size(); ++i) {
        std::cout << input_ids[i];
        if (i < input_ids.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << std::endl;
    
    // Generate text with improved sampling
    std::cout << "Step 5: Generating text..." << std::endl;
    int max_new_tokens = 50;
    
    // Configure sampling parameters for better, more diverse output
    SamplingConfig sampling;
    sampling.temperature = 0.8f;       // Balanced for coherent but diverse text
    sampling.top_k = 50;               // Consider top 50 tokens
    sampling.top_p = 0.95f;            // Nucleus sampling at 95%
    sampling.repetition_penalty = 1.05f; // Very mild repetition penalty
    
    std::cout << "Using sampling: temp=" << sampling.temperature 
              << ", top_k=" << sampling.top_k 
              << ", top_p=" << sampling.top_p 
              << ", rep_penalty=" << sampling.repetition_penalty << std::endl;
    
    std::vector<int> output_ids = model.generate(input_ids, max_new_tokens, sampling);
    std::cout << std::endl;
    
    // Decode output
    std::cout << "Step 6: Decoding output..." << std::endl;
    std::string generated_text = tokenizer.decode(output_ids);
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Generated text:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << generated_text << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}

#include <iostream>
#include <string>
#include "model_loader.h"
#include "tokenizer.h"
#include "tensor.h"
#include "ops.h"

int main(int argc, char* argv[]) {
    std::cout << "GPT-2 Inference Engine" << std::endl;
    
    // Load model weights
    ModelLoader model("weights/gpt2_weights.bin");
    if (!model.load()) {
        std::cerr << "Failed to load model weights" << std::endl;
        return 1;
    }
    
    // Initialize tokenizer
    Tokenizer tokenizer;
    
    // Inference loop
    std::string prompt = "Hello, world!";
    std::cout << "Prompt: " << prompt << std::endl;
    
    // TODO: Implement inference loop
    
    return 0;
}

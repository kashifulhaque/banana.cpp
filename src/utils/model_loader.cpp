#include "utils/model_loader.h"
#include <fstream>
#include <iostream>

ModelLoader::ModelLoader(const std::string& weights_path) 
    : weights_path_(weights_path) {
}

ModelLoader::~ModelLoader() {
}

bool ModelLoader::load() {
    std::ifstream file(weights_path_, std::ios::binary);
    if(!file.is_open()) {
        std::cerr << "Failed to open weights file: " << weights_path_ << std::endl;
        return false;
    }

    std::cout << "Loading weights from: " << weights_path_ << std::endl;

    // Check for new format with magic number
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
    
    bool new_format = (magic == 0x534D4C32); // "SML2"
    uint32_t version = 1;
    TensorPrecision precision = TensorPrecision::FP32;
    uint32_t tensor_count_header = 0;

    if (new_format) {
        file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
        
        if (version >= 2) {
            uint8_t precision_byte;
            file.read(reinterpret_cast<char*>(&precision_byte), sizeof(uint8_t));
            precision = static_cast<TensorPrecision>(precision_byte);
            file.read(reinterpret_cast<char*>(&tensor_count_header), sizeof(uint32_t));
        }
        
        std::cout << "Format: v" << version;
        switch (precision) {
            case TensorPrecision::FP32: std::cout << " (FP32)"; break;
            case TensorPrecision::FP16: std::cout << " (FP16)"; break;
            case TensorPrecision::BF16: std::cout << " (BF16)"; break;
        }
        std::cout << std::endl;
    } else {
        // Old format - seek back to beginning
        file.seekg(0);
        std::cout << "Format: legacy (FP32)" << std::endl;
    }

    int tensor_count = 0;

    while(file.peek() != EOF) {
        /// read name length
        int name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(int));
        if (file.eof()) break;

        /// read name
        std::string name(name_len, ' ');
        file.read(&name[0], name_len);

        /// read shape length and dimensions
        int shape_len;
        file.read(reinterpret_cast<char*>(&shape_len), sizeof(int));
        std::vector<int> shape(shape_len);
        file.read(reinterpret_cast<char*>(shape.data()), shape_len * sizeof(int));

        /// calculate total size
        int total_size = 1;
        for (int dim : shape) {
            total_size *= dim;
        }

        /// create tensor and read data
        Tensor t;
        t.shape = shape;
        t.storage_precision = precision;

        if (precision == TensorPrecision::FP32) {
            t.data.resize(total_size);
            file.read(reinterpret_cast<char*>(t.data.data()), total_size * sizeof(float));
        } else {
            // Read as fp16/bf16, will convert to fp32 on access
            t.data_f16.resize(total_size);
            file.read(reinterpret_cast<char*>(t.data_f16.data()), total_size * sizeof(uint16_t));
            // Immediately convert to fp32 for computation
            t.ensure_fp32();
        }

        /// store in map
        weights[name] = std::move(t);
        tensor_count++;
        
        // Print shape info
        std::cout << "  [" << tensor_count << "] " << name << " (";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }
    
    file.close();
    std::cout << "Successfully loaded " << tensor_count << " tensors" << std::endl;
    return true;
}

const Tensor* ModelLoader::get(const std::string& name) const {
    auto it = weights.find(name);
    if (it != weights.end()) {
        return &it->second;
    }
    return nullptr;
}

bool ModelLoader::has(const std::string& name) const {
    return weights.find(name) != weights.end();
}

std::vector<std::string> ModelLoader::get_weight_names() const {
    std::vector<std::string> names;
    for (const auto& pair : weights) {
        names.push_back(pair.first);
    }
    return names;
}

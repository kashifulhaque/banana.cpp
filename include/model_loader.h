#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <string>
#include <vector>
#include <unordered_map>
#include "tensor.h"

class ModelLoader {
public:
    ModelLoader(const std::string& weights_path);
    ~ModelLoader();
    
    bool load();
    
    // Get a tensor by name
    const Tensor* get(const std::string& name) const;
    
    // Check if a tensor exists
    bool has(const std::string& name) const;
    
    // Get all weight names
    std::vector<std::string> get_weight_names() const;
    
    std::unordered_map<std::string, Tensor> weights;
    
private:
    std::string weights_path_;
};

#endif // MODEL_LOADER_H

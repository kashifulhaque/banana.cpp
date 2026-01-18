#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <string>
#include <vector>

class ModelLoader {
public:
    ModelLoader(const std::string& weights_path);
    ~ModelLoader();
    
    bool load();
    
private:
    std::string weights_path_;
};

#endif // MODEL_LOADER_H

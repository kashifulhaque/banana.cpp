#include <fstream>
#include <iostream>

#include "model_loader.h"

void ModelLoader::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()) {
        std:cerr << "Failed to open weights file!" << std::endl;
        return;
    }

    while(file.peek() != EOF) {
        /// read name length
        int name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(int));

        /// read name
        std::string name(name_len, ' ');
        file.read(&name[0], name_len);

        /// read shape length and dimensions
        int shape_len;
        file.read(reinterpret_cast<char*>(&shape_len), sizeof(int));
        std::vector<int> shape(shape_len);
        file.read(reinterpret_cast<char*>(shape.data()), shape_len * sizeof(int));

        /// create tensor and read data
        Tensor t(shape);
        file.read(reinterpret_cast<char*>(t.data.dat()), t.size() * sizeof(float))

        /// store in map
        weights[name] = t;
        std::cout << "Loaded: " << name << std::endl;
    }
    file.close();
}

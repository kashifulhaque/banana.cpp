#ifndef UTILS_MODEL_LOADER_H
#define UTILS_MODEL_LOADER_H

#include "core/tensor.h"
#include <string>
#include <unordered_map>
#include <vector>

/// Model weight loader for binary format
class ModelLoader {
public:
  ModelLoader(const std::string &weights_path);
  ~ModelLoader();

  /// Load weights from file
  /// @return true on success
  bool load();

  /// Get a tensor by name
  /// @param name Weight name (e.g., "model.embed_tokens.weight")
  /// @return Pointer to tensor or nullptr if not found
  const Tensor *get(const std::string &name) const;

  /// Check if a tensor exists
  bool has(const std::string &name) const;

  /// Get all weight names
  std::vector<std::string> get_weight_names() const;

  /// Direct access to weights (for advanced use)
  std::unordered_map<std::string, Tensor> weights;

private:
  std::string weights_path_;
};

#endif // UTILS_MODEL_LOADER_H

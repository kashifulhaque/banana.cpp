#include "core/ops.h"
#include <cmath>

namespace ops {

Tensor matmul(const Tensor &a, const Tensor &b) {
  return Tensor::matmul(a, b);
}

Tensor matmul_ex(const Tensor &a, const Tensor &b, bool transpose_a, bool transpose_b) {
  return Tensor::matmul_ex(a, b, transpose_a, transpose_b);
}

Tensor add(const Tensor &a, const Tensor &b) {
  return Tensor::add(a, b);
}

Tensor gelu(const Tensor &x) {
  Tensor result = x;
  for (size_t i = 0; i < result.data.size(); ++i) {
    float v = x.data[i];
    result.data[i] = 0.5f * v * (1.0f + std::tanh(std::sqrt(2.0f / 3.14159265f) * (v + 0.044715f * v * v * v)));
  }
  return result;
}

Tensor softmax(const Tensor &x, int dim) {
  Tensor result = x;
  
  if (dim == -1) {
    int features = x.shape.back();
    int outer_dim = x.data.size() / features;
    
    for (int i = 0; i < outer_dim; ++i) {
      // Find max
      float max_val = -1e10f;
      for (int j = 0; j < features; ++j) {
        max_val = std::max(max_val, x.data[i * features + j]);
      }
      
      // Exp and sum
      float sum = 0.0f;
      for (int j = 0; j < features; ++j) {
        result.data[i * features + j] = std::exp(x.data[i * features + j] - max_val);
        sum += result.data[i * features + j];
      }
      
      // Normalize
      for (int j = 0; j < features; ++j) {
        result.data[i * features + j] /= (sum + 1e-10f);
      }
    }
  }
  
  return result;
}

Tensor layer_norm(const Tensor &x, const Tensor &gamma, const Tensor &beta) {
  int features = x.shape.back();
  int outer_dim = x.data.size() / features;
  
  Tensor result = x;
  
  for (int i = 0; i < outer_dim; ++i) {
    // Mean
    float mean = 0.0f;
    for (int j = 0; j < features; ++j) {
      mean += x.data[i * features + j];
    }
    mean /= features;
    
    // Variance
    float variance = 0.0f;
    for (int j = 0; j < features; ++j) {
      float diff = x.data[i * features + j] - mean;
      variance += diff * diff;
    }
    variance /= features;
    
    // Normalize
    float std_inv = 1.0f / std::sqrt(variance + 1e-5f);
    for (int j = 0; j < features; ++j) {
      result.data[i * features + j] = (x.data[i * features + j] - mean) * std_inv * gamma.data[j] + beta.data[j];
    }
  }
  
  return result;
}

} // namespace ops

#include "ops.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>

namespace ops {

Tensor matmul(const Tensor& a, const Tensor& b) {
    return Tensor::matmul(a, b);
}

Tensor add(const Tensor& a, const Tensor& b) {
    return Tensor::add(a, b);
}

Tensor gelu(const Tensor& x) {
    /// gelu approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Tensor result = x;
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    
    for (size_t i = 0; i < x.data.size(); ++i) {
        float val = x.data[i];
        float x3 = val * val * val;
        float inner = sqrt_2_over_pi * (val + 0.044715f * x3);
        result.data[i] = 0.5f * val * (1.0f + std::tanh(inner));
    }
    
    return result;
}

Tensor softmax(const Tensor& x, int dim) {
    Tensor result = x;
    
    /// simple 1D or 2D softmax (for last dimension)
    if (x.shape.size() == 1) {
        /// 1D softmax
        float max_val = *std::max_element(x.data.begin(), x.data.end());
        float sum = 0.0f;
        
        for (size_t i = 0; i < x.data.size(); ++i) {
            result.data[i] = std::exp(x.data[i] - max_val);
            sum += result.data[i];
        }
        
        for (size_t i = 0; i < result.data.size(); ++i) {
            result.data[i] /= sum;
        }
    } else if (x.shape.size() == 2) {
        /// 2D softmax over last dimension
        int rows = x.shape[0];
        int cols = x.shape[1];
        
        for (int i = 0; i < rows; ++i) {
            float max_val = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < cols; ++j) {
                max_val = std::max(max_val, x.data[i * cols + j]);
            }
            
            float sum = 0.0f;
            for (int j = 0; j < cols; ++j) {
                result.data[i * cols + j] = std::exp(x.data[i * cols + j] - max_val);
                sum += result.data[i * cols + j];
            }
            
            for (int j = 0; j < cols; ++j) {
                result.data[i * cols + j] /= sum;
            }
        }
    }
    
    return result;
}

Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta) {
    Tensor result = x;
    
    /// assuming x is 2D: (batch, features) or 1D: (features,)
    int features = x.shape.back();
    int outer_dim = x.data.size() / features;
    
    const float eps = 1e-5f;
    
    for (int i = 0; i < outer_dim; ++i) {
        /// calculate mean
        float mean = 0.0f;
        for (int j = 0; j < features; ++j) {
            mean += x.data[i * features + j];
        }
        mean /= features;
        
        /// calculate variance
        float variance = 0.0f;
        for (int j = 0; j < features; ++j) {
            float diff = x.data[i * features + j] - mean;
            variance += diff * diff;
        }
        variance /= features;
        
        /// normalize and scale
        float std = std::sqrt(variance + eps);
        for (int j = 0; j < features; ++j) {
            float normalized = (x.data[i * features + j] - mean) / std;
            result.data[i * features + j] = gamma.data[j] * normalized + beta.data[j];
        }
    }
    
    return result;
}

} // namespace ops

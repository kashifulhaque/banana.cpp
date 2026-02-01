#include "layers/normalization.h"
#include <cmath>
#include <vector>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

namespace layers {

// RMSNorm

Tensor RMSNorm::forward(const Tensor& x, const Tensor& weight,
                        const Tensor* /*bias*/) const {
  int features = x.shape.back();
  int outer_dim = x.data.size() / features;

  Tensor result = x;

#if defined(__APPLE__)
  // Vectorized RMSNorm using Accelerate vDSP
  for (int i = 0; i < outer_dim; ++i) {
    const float* row = x.data.data() + i * features;
    float* out_row = result.data.data() + i * features;
    
    // Compute sum of squares using vectorized dot product
    float sum_sq;
    vDSP_dotpr(row, 1, row, 1, &sum_sq, features);
    
    float rms = std::sqrt(sum_sq / features + eps_);
    float scale = 1.0f / rms;
    
    // Scale the input: out = row * scale
    vDSP_vsmul(row, 1, &scale, out_row, 1, features);
    
    // Element-wise multiply with weight: out = out * weight
    vDSP_vmul(out_row, 1, weight.data.data(), 1, out_row, 1, features);
  }
#else
  // Fallback scalar implementation
  for (int i = 0; i < outer_dim; ++i) {
    float sum_sq = 0.0f;
    for (int j = 0; j < features; ++j) {
      float val = x.data[i * features + j];
      sum_sq += val * val;
    }
    float rms = std::sqrt(sum_sq / features + eps_);
    float scale = 1.0f / rms;

    for (int j = 0; j < features; ++j) {
      result.data[i * features + j] = x.data[i * features + j] * scale * weight.data[j];
    }
  }
#endif

  return result;
}

// LayerNorm

Tensor LayerNorm::forward(const Tensor& x, const Tensor& weight, const Tensor* bias) const {
  int features = x.shape.back();
  int outer_dim = x.data.size() / features;

  Tensor result = x;

#if defined(__APPLE__)
  // Vectorized LayerNorm using Accelerate vDSP
  std::vector<float> centered(features);
  
  for (int i = 0; i < outer_dim; ++i) {
    const float* row = x.data.data() + i * features;
    float* out_row = result.data.data() + i * features;
    
    // Calculate mean using vDSP
    float mean;
    vDSP_meanv(row, 1, &mean, features);
    
    // Center the data: centered = row - mean
    float neg_mean = -mean;
    vDSP_vsadd(row, 1, &neg_mean, centered.data(), 1, features);
    
    // Calculate variance using dot product: variance = sum(centered^2) / features
    float variance;
    vDSP_dotpr(centered.data(), 1, centered.data(), 1, &variance, features);
    variance /= features;
    
    // Normalize: out = centered * std_inv
    float std_inv = 1.0f / std::sqrt(variance + eps_);
    vDSP_vsmul(centered.data(), 1, &std_inv, out_row, 1, features);
    
    // Scale by weight: out = out * weight
    vDSP_vmul(out_row, 1, weight.data.data(), 1, out_row, 1, features);
    
    // Add bias if present
    if (bias) {
      vDSP_vadd(out_row, 1, bias->data.data(), 1, out_row, 1, features);
    }
  }
#else
  // Fallback scalar implementation
  for (int i = 0; i < outer_dim; ++i) {
    // Calculate mean
    float mean = 0.0f;
    for (int j = 0; j < features; ++j) {
      mean += x.data[i * features + j];
    }
    mean /= features;

    // Calculate variance
    float variance = 0.0f;
    for (int j = 0; j < features; ++j) {
      float diff = x.data[i * features + j] - mean;
      variance += diff * diff;
    }
    variance /= features;

    // Normalize
    float std_inv = 1.0f / std::sqrt(variance + eps_);
    for (int j = 0; j < features; ++j) {
      float normalized = (x.data[i * features + j] - mean) * std_inv;
      result.data[i * features + j] = weight.data[j] * normalized;
      if (bias) {
        result.data[i * features + j] += bias->data[j];
      }
    }
  }
#endif

  return result;
}

} // namespace layers

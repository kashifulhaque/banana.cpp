#include "layers/mlp.h"
#include "core/ops.h"
#include <cmath>
#include <algorithm>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace layers {

// SwiGLU MLP

SwiGLUMLP::SwiGLUMLP(const MLPConfig& config) : config_(config) {}

Tensor SwiGLUMLP::linear(const Tensor& x, const Tensor& weight) {
  int in_features = weight.shape[1];
  int out_features = weight.shape[0];
  int batch = x.data.size() / in_features;

  Tensor output = ops::matmul_ex(x, weight, false, true);
  if (batch == 1) {
    output.shape = {out_features};
  }

  if (x.shape.size() == 2 && x.shape[0] > 1) {
    output.shape = {x.shape[0], out_features};
  }

  return output;
}

void SwiGLUMLP::apply_activation(Tensor& x) const {
  int total = x.data.size();
  
#if defined(__APPLE__)
  if (config_.activation == Activation::SILU) {
    // Compute sigmoid(x): 1 / (1 + exp(-x))
    std::vector<float> neg_x(total);
    float neg_one = -1.0f;
    vDSP_vsmul(x.data.data(), 1, &neg_one, neg_x.data(), 1, total);
    
    vvexpf(neg_x.data(), neg_x.data(), &total);
    
    float one = 1.0f;
    vDSP_vsadd(neg_x.data(), 1, &one, neg_x.data(), 1, total);
    
    // x / (1 + exp(-x))
    vDSP_vdiv(neg_x.data(), 1, x.data.data(), 1, x.data.data(), 1, total);
    return;
  }
#endif
  
  // Scalar fallback
  for (int i = 0; i < total; ++i) {
    float val = x.data[i];
    
    switch (config_.activation) {
      case Activation::SILU:
        x.data[i] = val / (1.0f + std::exp(-val));
        break;
      case Activation::GELU:
      case Activation::GELU_NEW:
        x.data[i] = 0.5f * val * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (val + 0.044715f * val * val * val)));
        break;
      case Activation::RELU:
        x.data[i] = std::max(0.0f, val);
        break;
    }
  }
}

Tensor SwiGLUMLP::forward(
    const Tensor& x,
    const Tensor& gate_weight,
    const Tensor& up_weight,
    const Tensor& down_weight) {
  
  int seq_len = x.shape[0];
  int intermediate_size = config_.intermediate_size;

  Tensor gate = linear(x, gate_weight);
  Tensor up = linear(x, up_weight);

  // Apply activation to gate
  apply_activation(gate);

  // Element-wise multiply: hidden = silu(gate) * up
  Tensor hidden({seq_len, intermediate_size});
  
#if defined(__APPLE__)
  int total = seq_len * intermediate_size;
  vDSP_vmul(gate.data.data(), 1, up.data.data(), 1, hidden.data.data(), 1, total);
#else
  for (int i = 0; i < seq_len * intermediate_size; ++i) {
    hidden.data[i] = gate.data[i] * up.data[i];
  }
#endif

  // Down projection
  Tensor output = linear(hidden, down_weight);
  output.shape = {seq_len, config_.hidden_size};

  return output;
}

// Standard MLP

StandardMLP::StandardMLP(const MLPConfig& config) : config_(config) {}

Tensor StandardMLP::linear(const Tensor& x, const Tensor& weight) {
  int in_features = weight.shape[1];
  int out_features = weight.shape[0];
  int batch = x.data.size() / in_features;

  Tensor output = ops::matmul_ex(x, weight, false, true);
  if (batch == 1) {
    output.shape = {out_features};
  }

  if (x.shape.size() == 2 && x.shape[0] > 1) {
    output.shape = {x.shape[0], out_features};
  }

  return output;
}

void StandardMLP::apply_activation(Tensor& x) const {
  int total = x.data.size();
  
  for (int i = 0; i < total; ++i) {
    float val = x.data[i];
    
    switch (config_.activation) {
      case Activation::SILU:
        x.data[i] = val / (1.0f + std::exp(-val));
        break;
      case Activation::GELU:
      case Activation::GELU_NEW:
        x.data[i] = 0.5f * val * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * 
                    (val + 0.044715f * val * val * val)));
        break;
      case Activation::RELU:
        x.data[i] = std::max(0.0f, val);
        break;
    }
  }
}

Tensor StandardMLP::forward(
    const Tensor& x,
    const Tensor& /*gate_weight*/,
    const Tensor& up_weight,
    const Tensor& down_weight) {
  
  int seq_len = x.shape[0];

  // up projection
  Tensor hidden = linear(x, up_weight);
  
  // Apply activation
  apply_activation(hidden);

  // Down projection
  Tensor output = linear(hidden, down_weight);
  output.shape = {seq_len, config_.hidden_size};

  return output;
}

} // namespace layers

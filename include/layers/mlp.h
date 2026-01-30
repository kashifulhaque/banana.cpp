#ifndef LAYERS_MLP_H
#define LAYERS_MLP_H

#include "core/tensor.h"

namespace layers {

/// Activation function types
enum class Activation {
  SILU,       // SiLU / Swish (LLaMA, Qwen)
  GELU,       // GELU (GPT-2, some older models)
  GELU_NEW,   // GELU approximation
  RELU        // ReLU
};

/// Configuration for MLP layer
struct MLPConfig {
  int hidden_size = 2048;
  int intermediate_size = 8192;
  Activation activation = Activation::SILU;
  bool use_bias = false;
};

/// Abstract base class for MLP layers
class MLPLayer {
public:
  virtual ~MLPLayer() = default;
  
  /// Forward pass through MLP
  /// @param x Input tensor [seq_len, hidden_size]
  /// @param gate_weight Gate projection weight
  /// @param up_weight Up projection weight
  /// @param down_weight Down projection weight
  /// @return Output tensor [seq_len, hidden_size]
  virtual Tensor forward(
    const Tensor& x,
    const Tensor& gate_weight,
    const Tensor& up_weight,
    const Tensor& down_weight
  ) = 0;
};

/// SwiGLU MLP layer (Gated Linear Unit with SiLU activation)
/// Used by LLaMA, Qwen, Mistral, SmolLM, and most modern LLMs
/// Computes: down(silu(gate(x)) * up(x))
class SwiGLUMLP : public MLPLayer {
public:
  explicit SwiGLUMLP(const MLPConfig& config);
  
  Tensor forward(
    const Tensor& x,
    const Tensor& gate_weight,
    const Tensor& up_weight,
    const Tensor& down_weight
  ) override;
  
  const MLPConfig& config() const { return config_; }

private:
  MLPConfig config_;
  
  // Apply activation function to tensor
  void apply_activation(Tensor& x) const;
  
  // Linear projection helper
  Tensor linear(const Tensor& x, const Tensor& weight);
};

/// Standard MLP layer (without gating)
/// Used by GPT-2, BERT, and some older models
/// Computes: down(activation(up(x)))
class StandardMLP : public MLPLayer {
public:
  explicit StandardMLP(const MLPConfig& config);
  
  Tensor forward(
    const Tensor& x,
    const Tensor& gate_weight,  // Unused, for interface compatibility
    const Tensor& up_weight,
    const Tensor& down_weight
  ) override;
  
  const MLPConfig& config() const { return config_; }

private:
  MLPConfig config_;
  
  void apply_activation(Tensor& x) const;
  Tensor linear(const Tensor& x, const Tensor& weight);
};

} // namespace layers

#endif // LAYERS_MLP_H

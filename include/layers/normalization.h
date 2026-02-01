#ifndef LAYERS_NORMALIZATION_H
#define LAYERS_NORMALIZATION_H

#include "core/tensor.h"

namespace layers {

/// Abstract base class for normalization layers
class NormLayer {
public:
  virtual ~NormLayer() = default;
  
  /// Apply normalization to input tensor
  /// @param x Input tensor of shape [seq_len, hidden_size] or [hidden_size]
  /// @param weight Normalization weight (gamma)
  /// @param bias Optional bias (beta), nullptr if not used
  /// @return Normalized tensor of same shape as input
  virtual Tensor forward(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias = nullptr
  ) const = 0;
};

/// Root Mean Square Layer Normalization
/// Used by LLaMA, Qwen, Mistral, and most modern LLMs
class RMSNorm : public NormLayer {
public:
  explicit RMSNorm(float eps = 1e-5f) : eps_(eps) {}
  
  Tensor forward(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias = nullptr
  ) const override;
  
  void set_eps(float eps) { eps_ = eps; }
  float eps() const { return eps_; }

private:
  float eps_;
};

/// Standard Layer Normalization
/// Used by GPT-2, BERT, and some older models
class LayerNorm : public NormLayer {
public:
  explicit LayerNorm(float eps = 1e-5f) : eps_(eps) {}
  
  Tensor forward(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias = nullptr
  ) const override;
  
  void set_eps(float eps) { eps_ = eps; }
  float eps() const { return eps_; }

private:
  float eps_;
};

} // namespace layers

#endif // LAYERS_NORMALIZATION_H

#ifndef CORE_OPS_H
#define CORE_OPS_H

#include "core/tensor.h"

namespace ops {

/// Matrix multiplication operations
Tensor matmul(const Tensor &a, const Tensor &b);
Tensor matmul_ex(const Tensor &a, const Tensor &b, bool transpose_a, bool transpose_b);
Tensor add(const Tensor &a, const Tensor &b);

/// Activation functions
Tensor gelu(const Tensor &x);
Tensor softmax(const Tensor &x, int dim = -1);

/// Layer normalization
Tensor layer_norm(const Tensor &x, const Tensor &gamma, const Tensor &beta);

} // namespace ops

#endif // CORE_OPS_H

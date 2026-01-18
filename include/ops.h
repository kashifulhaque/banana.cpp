#ifndef OPS_H
#define OPS_H

#include "tensor.h"

namespace ops {
/// matrix ops
Tensor matmul(const Tensor &a, const Tensor &b);
Tensor add(const Tensor &a, const Tensor &b);

/// activation functions
Tensor gelu(const Tensor &x);
Tensor softmax(const Tensor &x, int dim = -1);

/// layer norm
Tensor layer_norm(const Tensor &x, const Tensor &gamma, const Tensor &beta);
} // namespace ops

#endif // OPS_H

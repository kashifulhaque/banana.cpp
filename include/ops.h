#ifndef OPS_H
#define OPS_H

#include "tensor.h"

namespace ops {
    // Matrix operations
    Tensor matmul(const Tensor& a, const Tensor& b);
    Tensor add(const Tensor& a, const Tensor& b);
    
    // Activation functions
    Tensor gelu(const Tensor& x);
    Tensor softmax(const Tensor& x, int dim = -1);
    
    // Layer normalization
    Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta);
}

#endif // OPS_H

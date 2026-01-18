#include "ops.h"
#include <cmath>
#include <algorithm>

namespace ops {

Tensor matmul(const Tensor& a, const Tensor& b) {
    // TODO: Implement matrix multiplication
    Tensor result;
    return result;
}

Tensor add(const Tensor& a, const Tensor& b) {
    // TODO: Implement element-wise addition
    Tensor result;
    return result;
}

Tensor gelu(const Tensor& x) {
    // TODO: Implement GELU activation
    // GELU(x) = x * Φ(x), where Φ(x) is the CDF of standard normal distribution
    Tensor result;
    return result;
}

Tensor softmax(const Tensor& x, int dim) {
    // TODO: Implement softmax
    Tensor result;
    return result;
}

Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta) {
    // TODO: Implement layer normalization
    Tensor result;
    return result;
}

} // namespace ops

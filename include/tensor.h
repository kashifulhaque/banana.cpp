#ifndef TENSOR_H
#define TENSOR_H

#include <cmath>
#include <vector>
#include <numeric>
#include <cassert>
#include <iostream>

struct Tensor {
    std::vector<float> data;
    std::vector<int> shape;

    Tensor() {}

    /// constructor to init with zeros
    Tensor(std::vector<int> s) : shape(s) {
        int size = 1;
        for (int dim : shape) size *= dim;  // Fixed: should be *= not +=
        data.resize(size, 0.0f);
    }

    /// helper to access data linearly
    float& operator[](int i) {
        return data[i];
    }
    const float& operator[](int i) const {
        return data[i];
    }

    int size() const {
        return data.size();
    }

    /// basic matmul (naive implementation, as of now)
    /// A (m x k) * B (k x n) -> C (m x n)
    static Tensor matmul(const Tensor& A, const Tensor& B) {
        assert(A.shape.size() >= 2 && B.shape.size() >= 2);
        assert(A.shape.back() == B.shape[0]);   // check for 'K'

        int m = A.shape[0];
        int k = A.shape[1];
        int n = B.shape[1];

        Tensor C({m, n});

        for(int i = 0; i < m; ++i) {
            for(int j = 0; j < n; ++j) {  // Fixed: j < n (not i < n), and removed comma
                float sum = 0.0f;
                for(int p = 0; p < k; ++p) {
                    sum += A.data[i * k + p] * B.data[p * n + j];
                }
                C.data[i * n + j] = sum;
            }
        }

        return C;
    }

    /// element-wise addition
    static Tensor add(const Tensor& A, const Tensor& B) {
        assert(A.shape == B.shape);
        Tensor C = A;   // copy shape and size
        for(size_t i = 0; i < A.data.size(); ++i) {
            C.data[i] += B.data[i];
        }
        return C;
    }
};

#endif // TENSOR_H

#ifndef TENSOR_H
#define TENSOR_H

#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

struct Tensor {
  std::vector<float> data;
  std::vector<int> shape;

  Tensor() {}

  /// init with zeros
  Tensor(std::vector<int> s) : shape(s) {
    int size = 1;
    for (int dim : shape) size *= dim;
    data.resize(size, 0.0f);
  }

  /// helper to access data linearly
  float &operator[](int i) { return data[i]; }
  const float &operator[](int i) const { return data[i]; }

  int size() const { return data.size(); }

  /// basic matmul (naive implementation, as of now)
  /// A (m x k) * B (k x n) -> C (m x n)
  static Tensor matmul(const Tensor &A, const Tensor &B) {
    assert(A.shape.size() >= 2 && B.shape.size() >= 2);
    assert(A.shape.back() == B.shape[0]); // check for 'K'

    int m = A.shape[0];
    int k = A.shape[1];
    int n = B.shape[1];

    Tensor C({m, n});

#if defined(__APPLE__)
    // Use Accelerate BLAS for fast SGEMM on macOS
    cblas_sgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k,
      1.0f,
      A.data.data(), k,
      B.data.data(), n,
      0.0f,
      C.data.data(), n
    );
#else
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        float sum = 0.0f;
        for (int p = 0; p < k; ++p) sum += A.data[i * k + p] * B.data[p * n + j];
        C.data[i * n + j] = sum;
      }
    }
#endif

    return C;
  }

  /// matmul with optional transpose flags
  /// C = (A op) * (B op)
  static Tensor matmul_ex(const Tensor &A, const Tensor &B, bool transposeA, bool transposeB) {
    assert(A.shape.size() >= 2 && B.shape.size() >= 2);

    int a_rows = A.shape[0];
    int a_cols = A.shape[1];
    int b_rows = B.shape[0];
    int b_cols = B.shape[1];

    int m = transposeA ? a_cols : a_rows;
    int kA = transposeA ? a_rows : a_cols;
    int kB = transposeB ? b_cols : b_rows;
    int n = transposeB ? b_rows : b_cols;

    assert(kA == kB);

    Tensor C({m, n});

#if defined(__APPLE__)
    CBLAS_TRANSPOSE transA = transposeA ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE transB = transposeB ? CblasTrans : CblasNoTrans;
    int lda = transposeA ? m : kA;
    int ldb = transposeB ? kB : n;

    cblas_sgemm(
      CblasRowMajor,
      transA,
      transB,
      m, n, kA,
      1.0f,
      A.data.data(), lda,
      B.data.data(), ldb,
      0.0f,
      C.data.data(), n
    );
#else
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        float sum = 0.0f;
        for (int p = 0; p < kA; ++p) {
          float a_val = transposeA ? A.data[p * a_cols + i] : A.data[i * a_cols + p];
          float b_val = transposeB ? B.data[j * b_cols + p] : B.data[p * b_cols + j];
          sum += a_val * b_val;
        }
        C.data[i * n + j] = sum;
      }
    }
#endif

    return C;
  }

  /// element-wise addition
  static Tensor add(const Tensor &A, const Tensor &B) {
    assert(A.shape == B.shape);
    Tensor C = A; // copy shape and size
    for (size_t i = 0; i < A.data.size(); ++i) C.data[i] += B.data[i];
    return C;
  }
};

#endif // TENSOR_H

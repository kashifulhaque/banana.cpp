#ifndef CORE_TENSOR_H
#define CORE_TENSOR_H

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

/// Storage precision for tensors
enum class TensorPrecision {
  FP32 = 0,  // 32-bit float (4 bytes)
  FP16 = 1,  // 16-bit float (2 bytes)
  BF16 = 2   // bfloat16 (2 bytes)
};

/// Half-precision utilities for tensor operations
namespace tensor_fp16 {

inline uint16_t float_to_fp16(float value) {
  uint32_t f32;
  std::memcpy(&f32, &value, sizeof(float));
  uint32_t sign = (f32 >> 31) & 0x1;
  int32_t exp = ((f32 >> 23) & 0xFF) - 127;
  uint32_t mantissa = f32 & 0x7FFFFF;
  uint16_t f16;
  if (exp > 15) {
    f16 = (sign << 15) | 0x7C00;
  } else if (exp < -14) {
    if (exp < -24) {
      f16 = (sign << 15);
    } else {
      mantissa = (mantissa | 0x800000) >> (-exp - 14 + 13);
      f16 = (sign << 15) | (mantissa & 0x3FF);
    }
  } else {
    f16 = (sign << 15) | ((exp + 15) << 10) | (mantissa >> 13);
  }
  return f16;
}

inline float fp16_to_float(uint16_t value) {
  uint32_t sign = (value >> 15) & 0x1;
  uint32_t exp = (value >> 10) & 0x1F;
  uint32_t mantissa = value & 0x3FF;
  uint32_t f32;
  if (exp == 0) {
    if (mantissa == 0) {
      f32 = sign << 31;
    } else {
      exp = 1;
      while ((mantissa & 0x400) == 0) {
        mantissa <<= 1;
        exp--;
      }
      mantissa &= 0x3FF;
      f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }
  } else if (exp == 31) {
    f32 = (sign << 31) | 0x7F800000 | (mantissa << 13);
  } else {
    f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
  }
  float result;
  std::memcpy(&result, &f32, sizeof(float));
  return result;
}

inline uint16_t float_to_bf16(float value) {
  uint32_t f32;
  std::memcpy(&f32, &value, sizeof(float));
  uint32_t rounding = 0x00008000;
  f32 += rounding;
  return static_cast<uint16_t>(f32 >> 16);
}

inline float bf16_to_float(uint16_t value) {
  uint32_t f32 = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &f32, sizeof(float));
  return result;
}

} // namespace tensor_fp16

struct Tensor {
  std::vector<float> data;        // FP32 data (for computation)
  std::vector<uint16_t> data_f16; // FP16/BF16 storage (optional)
  std::vector<int> shape;
  TensorPrecision storage_precision = TensorPrecision::FP32;

  Tensor() {}

  /// init with zeros
  Tensor(std::vector<int> s) : shape(s) {
    int size = 1;
    for (int dim : shape) size *= dim;
    data.resize(size, 0.0f);
  }

  /// init with precision
  Tensor(std::vector<int> s, TensorPrecision precision) : shape(s), storage_precision(precision) {
    int size = 1;
    for (int dim : shape) size *= dim;
    if (precision == TensorPrecision::FP32) {
      data.resize(size, 0.0f);
    } else {
      data_f16.resize(size, 0);
    }
  }

  /// helper to access data linearly (always returns fp32)
  float &operator[](int i) { 
    ensure_fp32();
    return data[i]; 
  }
  const float &operator[](int i) const { 
    // For const access, we must have fp32 data already
    return data[i]; 
  }

  /// Get value at index (handles precision conversion)
  float get(int i) const {
    if (!data.empty()) {
      return data[i];
    } else if (!data_f16.empty()) {
      if (storage_precision == TensorPrecision::FP16) {
        return tensor_fp16::fp16_to_float(data_f16[i]);
      } else {
        return tensor_fp16::bf16_to_float(data_f16[i]);
      }
    }
    return 0.0f;
  }

  /// Set value at index (handles precision conversion)
  void set(int i, float value) {
    if (!data.empty()) {
      data[i] = value;
    } else if (!data_f16.empty()) {
      if (storage_precision == TensorPrecision::FP16) {
        data_f16[i] = tensor_fp16::float_to_fp16(value);
      } else {
        data_f16[i] = tensor_fp16::float_to_bf16(value);
      }
    }
  }

  int size() const { 
    return data.empty() ? static_cast<int>(data_f16.size()) : static_cast<int>(data.size()); 
  }

  /// Convert to FP32 for computation (lazy conversion)
  void ensure_fp32() {
    if (!data_f16.empty() && data.empty()) {
      data.resize(data_f16.size());
      if (storage_precision == TensorPrecision::FP16) {
        for (size_t i = 0; i < data_f16.size(); ++i) {
          data[i] = tensor_fp16::fp16_to_float(data_f16[i]);
        }
      } else { // BF16
        for (size_t i = 0; i < data_f16.size(); ++i) {
          data[i] = tensor_fp16::bf16_to_float(data_f16[i]);
        }
      }
      // Clear fp16 data to save memory
      data_f16.clear();
      data_f16.shrink_to_fit();
      storage_precision = TensorPrecision::FP32;
    }
  }

  /// Convert to reduced precision for storage
  void convert_to_fp16() {
    ensure_fp32();
    data_f16.resize(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      data_f16[i] = tensor_fp16::float_to_fp16(data[i]);
    }
    data.clear();
    data.shrink_to_fit();
    storage_precision = TensorPrecision::FP16;
  }

  void convert_to_bf16() {
    ensure_fp32();
    data_f16.resize(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      data_f16[i] = tensor_fp16::float_to_bf16(data[i]);
    }
    data.clear();
    data.shrink_to_fit();
    storage_precision = TensorPrecision::BF16;
  }

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

#endif // CORE_TENSOR_H

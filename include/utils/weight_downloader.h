#ifndef UTILS_WEIGHT_DOWNLOADER_H
#define UTILS_WEIGHT_DOWNLOADER_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

/// Storage precision for weights
enum class WeightPrecision {
  FP32,  // 32-bit float (4 bytes)
  FP16,  // 16-bit float (2 bytes)
  BF16   // bfloat16 (2 bytes)
};

/// Half-precision float utilities
namespace fp16_utils {

/// Convert float32 to float16 (IEEE 754)
uint16_t float_to_fp16(float value);

/// Convert float16 to float32
float fp16_to_float(uint16_t value);

/// Convert float32 to bfloat16
uint16_t float_to_bf16(float value);

/// Convert bfloat16 to float32
float bf16_to_float(uint16_t value);

} // namespace fp16_utils

/// Weight tensor metadata from safetensors
struct TensorInfo {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  int64_t data_offset;
  int64_t data_size;
};

/// Get HuggingFace token from environment, .env file, or user input
/// Returns empty string if no token is available
std::string get_hf_token();

/// Downloads and exports model weights from HuggingFace
class WeightDownloader {
public:
  WeightDownloader(const std::string &model_name = "HuggingFaceTB/SmolLM2-360M-Instruct");
  ~WeightDownloader();

  /// Set the storage precision (default: BF16)
  void set_precision(WeightPrecision precision) { precision_ = precision; }

  /// Download and export weights to binary format
  /// Returns true on success
  bool download_and_export(const std::string &output_dir = "weights/smollm2");

  /// Download only tokenizer files
  bool download_tokenizer(const std::string &output_dir = "weights/smollm2");

  /// Check if weights already exist
  bool weights_exist(const std::string &output_dir = "weights/smollm2");

private:
  std::string model_name_;
  std::string hf_token_;
  WeightPrecision precision_ = WeightPrecision::BF16;

  /// HuggingFace API helpers
  std::string get_hf_file_url(const std::string &filename);
  bool download_file(const std::string &url, const std::string &output_path);
  bool download_file_with_progress(const std::string &url, const std::string &output_path);

  /// Safetensors parsing
  bool parse_safetensors_header(const std::string &filepath,
                                 std::unordered_map<std::string, TensorInfo> &tensors,
                                 int64_t &header_size);

  /// Convert safetensors to custom binary format
  bool convert_safetensors_to_binary(const std::string &safetensors_path,
                                      const std::string &output_path);

  /// Write tensor data in target precision
  void write_tensor_data(std::ofstream &out, const float *data, size_t count);
  void write_tensor_data(std::ofstream &out, const uint16_t *data, size_t count, 
                         const std::string &src_dtype);
};

#endif // UTILS_WEIGHT_DOWNLOADER_H

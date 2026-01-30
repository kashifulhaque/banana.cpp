#include "weight_downloader.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>

// For system() and popen() calls
#include <cstdio>
#include <cstdlib>

namespace fp16_utils {

uint16_t float_to_fp16(float value) {
  uint32_t f32;
  std::memcpy(&f32, &value, sizeof(float));

  uint32_t sign = (f32 >> 31) & 0x1;
  int32_t exp = ((f32 >> 23) & 0xFF) - 127;
  uint32_t mantissa = f32 & 0x7FFFFF;

  uint16_t f16;

  if (exp > 15) {
    // Overflow to infinity
    f16 = (sign << 15) | 0x7C00;
  } else if (exp < -14) {
    // Underflow to zero or denormal
    if (exp < -24) {
      f16 = (sign << 15);
    } else {
      // Denormalized
      mantissa = (mantissa | 0x800000) >> (-exp - 14 + 13);
      f16 = (sign << 15) | (mantissa & 0x3FF);
    }
  } else {
    // Normalized
    f16 = (sign << 15) | ((exp + 15) << 10) | (mantissa >> 13);
  }

  return f16;
}

float fp16_to_float(uint16_t value) {
  uint32_t sign = (value >> 15) & 0x1;
  uint32_t exp = (value >> 10) & 0x1F;
  uint32_t mantissa = value & 0x3FF;

  uint32_t f32;

  if (exp == 0) {
    if (mantissa == 0) {
      // Zero
      f32 = sign << 31;
    } else {
      // Denormalized -> normalized float32
      exp = 1;
      while ((mantissa & 0x400) == 0) {
        mantissa <<= 1;
        exp--;
      }
      mantissa &= 0x3FF;
      f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }
  } else if (exp == 31) {
    // Inf or NaN
    f32 = (sign << 31) | 0x7F800000 | (mantissa << 13);
  } else {
    // Normalized
    f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
  }

  float result;
  std::memcpy(&result, &f32, sizeof(float));
  return result;
}

uint16_t float_to_bf16(float value) {
  uint32_t f32;
  std::memcpy(&f32, &value, sizeof(float));
  // BF16 is simply the upper 16 bits of float32 (with rounding)
  uint32_t rounding = 0x00008000; // round to nearest even
  f32 += rounding;
  return static_cast<uint16_t>(f32 >> 16);
}

float bf16_to_float(uint16_t value) {
  uint32_t f32 = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &f32, sizeof(float));
  return result;
}

} // namespace fp16_utils

WeightDownloader::WeightDownloader(const std::string &model_name)
    : model_name_(model_name) {}

WeightDownloader::~WeightDownloader() {}

std::string WeightDownloader::get_hf_file_url(const std::string &filename) {
  return "https://huggingface.co/" + model_name_ + "/resolve/main/" + filename;
}

bool WeightDownloader::weights_exist(const std::string &output_dir) {
  std::string weights_path = output_dir + "/smollm2_weights.bin";
  struct stat buffer;
  return (stat(weights_path.c_str(), &buffer) == 0);
}

bool WeightDownloader::download_file(const std::string &url,
                                      const std::string &output_path) {
  std::string cmd =
      "curl -L --fail --silent --show-error \"" + url + "\" -o \"" + output_path + "\"";
  int ret = system(cmd.c_str());
  return ret == 0;
}

bool WeightDownloader::download_file_with_progress(const std::string &url,
                                                    const std::string &output_path) {
  std::string cmd = "curl -L --fail --progress-bar \"" + url + "\" -o \"" + output_path + "\"";
  int ret = system(cmd.c_str());
  return ret == 0;
}

bool WeightDownloader::download_tokenizer(const std::string &output_dir) {
  // Create output directory
  std::string mkdir_cmd = "mkdir -p \"" + output_dir + "\"";
  system(mkdir_cmd.c_str());

  std::vector<std::string> tokenizer_files = {
      "vocab.json",
      "merges.txt", 
      "tokenizer.json",
      "tokenizer_config.json",
      "special_tokens_map.json",
      "chat_template.jinja"
  };

  std::cout << "Downloading tokenizer files...\n";

  for (const auto &filename : tokenizer_files) {
    std::string url = get_hf_file_url(filename);
    std::string output_path = output_dir + "/" + filename;

    // Check if file already exists
    struct stat buffer;
    if (stat(output_path.c_str(), &buffer) == 0) {
      std::cout << "  " << filename << " (already exists)\n";
      continue;
    }

    std::cout << "  Downloading " << filename << "...\n";
    if (!download_file(url, output_path)) {
      // Some files are optional (e.g., chat_template.jinja might not exist)
      if (filename != "chat_template.jinja") {
        std::cerr << "  Warning: Failed to download " << filename << "\n";
      }
    }
  }

  std::cout << "Tokenizer files downloaded to " << output_dir << "\n";
  return true;
}

bool WeightDownloader::parse_safetensors_header(
    const std::string &filepath,
    std::unordered_map<std::string, TensorInfo> &tensors,
    int64_t &header_size) {

  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open safetensors file: " << filepath << "\n";
    return false;
  }

  // Read header size (8 bytes, little-endian uint64)
  uint64_t header_len;
  file.read(reinterpret_cast<char *>(&header_len), 8);
  header_size = static_cast<int64_t>(header_len) + 8; // +8 for the size field itself

  // Read header JSON
  std::string header_json(header_len, '\0');
  file.read(&header_json[0], header_len);
  file.close();

  // Simple JSON parsing for safetensors header
  // Format: {"tensor_name": {"dtype": "F16", "shape": [dim1, dim2], "data_offsets": [start, end]}, ...}

  size_t pos = 0;
  while ((pos = header_json.find("\"", pos)) != std::string::npos) {
    // Find tensor name
    size_t name_start = pos + 1;
    size_t name_end = header_json.find("\"", name_start);
    if (name_end == std::string::npos) break;

    std::string tensor_name = header_json.substr(name_start, name_end - name_start);
    pos = name_end + 1;

    // Skip "__metadata__"
    if (tensor_name == "__metadata__") {
      // Skip to next tensor
      size_t brace_pos = header_json.find("{", pos);
      if (brace_pos != std::string::npos) {
        int brace_count = 1;
        pos = brace_pos + 1;
        while (pos < header_json.size() && brace_count > 0) {
          if (header_json[pos] == '{') brace_count++;
          else if (header_json[pos] == '}') brace_count--;
          pos++;
        }
      }
      continue;
    }

    // Find the opening brace for this tensor's metadata
    size_t brace_start = header_json.find("{", pos);
    if (brace_start == std::string::npos) break;

    // Find matching closing brace
    int brace_count = 1;
    size_t brace_end = brace_start + 1;
    while (brace_end < header_json.size() && brace_count > 0) {
      if (header_json[brace_end] == '{') brace_count++;
      else if (header_json[brace_end] == '}') brace_count--;
      brace_end++;
    }

    std::string tensor_meta = header_json.substr(brace_start, brace_end - brace_start);
    pos = brace_end;

    TensorInfo info;
    info.name = tensor_name;

    // Parse dtype
    size_t dtype_pos = tensor_meta.find("\"dtype\"");
    if (dtype_pos != std::string::npos) {
      size_t dtype_val_start = tensor_meta.find("\"", dtype_pos + 7);
      if (dtype_val_start != std::string::npos) {
        dtype_val_start++;
        size_t dtype_val_end = tensor_meta.find("\"", dtype_val_start);
        info.dtype = tensor_meta.substr(dtype_val_start, dtype_val_end - dtype_val_start);
      }
    }

    // Parse shape
    size_t shape_pos = tensor_meta.find("\"shape\"");
    if (shape_pos != std::string::npos) {
      size_t shape_start = tensor_meta.find("[", shape_pos);
      size_t shape_end = tensor_meta.find("]", shape_start);
      if (shape_start != std::string::npos && shape_end != std::string::npos) {
        std::string shape_str = tensor_meta.substr(shape_start + 1, shape_end - shape_start - 1);
        std::stringstream ss(shape_str);
        std::string dim_str;
        while (std::getline(ss, dim_str, ',')) {
          // Trim whitespace
          size_t start = dim_str.find_first_not_of(" \t");
          size_t end = dim_str.find_last_not_of(" \t");
          if (start != std::string::npos) {
            info.shape.push_back(std::stoll(dim_str.substr(start, end - start + 1)));
          }
        }
      }
    }

    // Parse data_offsets
    size_t offsets_pos = tensor_meta.find("\"data_offsets\"");
    if (offsets_pos != std::string::npos) {
      size_t offset_start = tensor_meta.find("[", offsets_pos);
      size_t offset_end = tensor_meta.find("]", offset_start);
      if (offset_start != std::string::npos && offset_end != std::string::npos) {
        std::string offset_str = tensor_meta.substr(offset_start + 1, offset_end - offset_start - 1);
        size_t comma_pos = offset_str.find(",");
        if (comma_pos != std::string::npos) {
          int64_t start_offset = std::stoll(offset_str.substr(0, comma_pos));
          int64_t end_offset = std::stoll(offset_str.substr(comma_pos + 1));
          info.data_offset = start_offset;
          info.data_size = end_offset - start_offset;
        }
      }
    }

    tensors[tensor_name] = info;
  }

  return !tensors.empty();
}

void WeightDownloader::write_tensor_data(std::ofstream &out, const float *data,
                                          size_t count) {
  if (precision_ == WeightPrecision::FP32) {
    out.write(reinterpret_cast<const char *>(data), count * sizeof(float));
  } else if (precision_ == WeightPrecision::FP16) {
    std::vector<uint16_t> fp16_data(count);
    for (size_t i = 0; i < count; ++i) {
      fp16_data[i] = fp16_utils::float_to_fp16(data[i]);
    }
    out.write(reinterpret_cast<const char *>(fp16_data.data()),
              count * sizeof(uint16_t));
  } else { // BF16
    std::vector<uint16_t> bf16_data(count);
    for (size_t i = 0; i < count; ++i) {
      bf16_data[i] = fp16_utils::float_to_bf16(data[i]);
    }
    out.write(reinterpret_cast<const char *>(bf16_data.data()),
              count * sizeof(uint16_t));
  }
}

void WeightDownloader::write_tensor_data(std::ofstream &out, const uint16_t *data,
                                          size_t count, const std::string &src_dtype) {
  if (precision_ == WeightPrecision::FP32) {
    // Convert to fp32
    std::vector<float> fp32_data(count);
    for (size_t i = 0; i < count; ++i) {
      if (src_dtype == "BF16") {
        fp32_data[i] = fp16_utils::bf16_to_float(data[i]);
      } else { // F16
        fp32_data[i] = fp16_utils::fp16_to_float(data[i]);
      }
    }
    out.write(reinterpret_cast<const char *>(fp32_data.data()),
              count * sizeof(float));
  } else if (precision_ == WeightPrecision::FP16) {
    if (src_dtype == "F16") {
      // Direct copy
      out.write(reinterpret_cast<const char *>(data), count * sizeof(uint16_t));
    } else { // BF16 -> FP16
      std::vector<uint16_t> fp16_data(count);
      for (size_t i = 0; i < count; ++i) {
        float val = fp16_utils::bf16_to_float(data[i]);
        fp16_data[i] = fp16_utils::float_to_fp16(val);
      }
      out.write(reinterpret_cast<const char *>(fp16_data.data()),
                count * sizeof(uint16_t));
    }
  } else { // BF16
    if (src_dtype == "BF16") {
      // Direct copy
      out.write(reinterpret_cast<const char *>(data), count * sizeof(uint16_t));
    } else { // F16 -> BF16
      std::vector<uint16_t> bf16_data(count);
      for (size_t i = 0; i < count; ++i) {
        float val = fp16_utils::fp16_to_float(data[i]);
        bf16_data[i] = fp16_utils::float_to_bf16(val);
      }
      out.write(reinterpret_cast<const char *>(bf16_data.data()),
                count * sizeof(uint16_t));
    }
  }
}

bool WeightDownloader::convert_safetensors_to_binary(
    const std::string &safetensors_path, const std::string &output_path) {

  std::unordered_map<std::string, TensorInfo> tensors;
  int64_t header_size;

  if (!parse_safetensors_header(safetensors_path, tensors, header_size)) {
    std::cerr << "Failed to parse safetensors header\n";
    return false;
  }

  std::cout << "Found " << tensors.size() << " tensors in safetensors file\n";

  // Open safetensors file for reading data
  std::ifstream in_file(safetensors_path, std::ios::binary);
  if (!in_file.is_open()) {
    std::cerr << "Failed to open safetensors file for reading\n";
    return false;
  }

  // Open output file
  std::ofstream out_file(output_path, std::ios::binary);
  if (!out_file.is_open()) {
    std::cerr << "Failed to open output file: " << output_path << "\n";
    return false;
  }

  // Write magic number and version for format identification
  const uint32_t MAGIC = 0x534D4C32; // "SML2" in little-endian
  const uint32_t VERSION = 2;        // Version 2 = supports fp16/bf16
  uint8_t precision_byte = static_cast<uint8_t>(precision_);

  out_file.write(reinterpret_cast<const char *>(&MAGIC), sizeof(uint32_t));
  out_file.write(reinterpret_cast<const char *>(&VERSION), sizeof(uint32_t));
  out_file.write(reinterpret_cast<const char *>(&precision_byte), sizeof(uint8_t));

  // Write tensor count
  uint32_t tensor_count = static_cast<uint32_t>(tensors.size());
  out_file.write(reinterpret_cast<const char *>(&tensor_count), sizeof(uint32_t));

  int count = 0;
  for (const auto &pair : tensors) {
    const TensorInfo &info = pair.second;
    count++;

    std::cout << "  [" << count << "/" << tensor_count << "] " << info.name << " ("
              << info.dtype << ") [";
    for (size_t i = 0; i < info.shape.size(); ++i) {
      std::cout << info.shape[i];
      if (i < info.shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    // Write tensor name
    int32_t name_len = static_cast<int32_t>(info.name.length());
    out_file.write(reinterpret_cast<const char *>(&name_len), sizeof(int32_t));
    out_file.write(info.name.c_str(), name_len);

    // Write shape
    int32_t shape_len = static_cast<int32_t>(info.shape.size());
    out_file.write(reinterpret_cast<const char *>(&shape_len), sizeof(int32_t));
    for (int64_t dim : info.shape) {
      int32_t dim32 = static_cast<int32_t>(dim);
      out_file.write(reinterpret_cast<const char *>(&dim32), sizeof(int32_t));
    }

    // Calculate total elements
    size_t total_elements = 1;
    for (int64_t dim : info.shape) {
      total_elements *= static_cast<size_t>(dim);
    }

    // Read tensor data from safetensors
    in_file.seekg(header_size + info.data_offset);

    if (info.dtype == "F32") {
      std::vector<float> data(total_elements);
      in_file.read(reinterpret_cast<char *>(data.data()), info.data_size);
      write_tensor_data(out_file, data.data(), total_elements);
    } else if (info.dtype == "F16" || info.dtype == "BF16") {
      std::vector<uint16_t> data(total_elements);
      in_file.read(reinterpret_cast<char *>(data.data()), info.data_size);
      write_tensor_data(out_file, data.data(), total_elements, info.dtype);
    } else {
      std::cerr << "Unsupported dtype: " << info.dtype << "\n";
      return false;
    }
  }

  in_file.close();
  out_file.close();

  std::cout << "Successfully converted " << count << " tensors\n";
  return true;
}

bool WeightDownloader::download_and_export(const std::string &output_dir) {
  // Create output directory
  std::string mkdir_cmd = "mkdir -p \"" + output_dir + "\"";
  system(mkdir_cmd.c_str());

  std::string weights_path = output_dir + "/smollm2_weights.bin";

  // Check if weights already exist
  if (weights_exist(output_dir)) {
    std::cout << "Weights already exist at " << weights_path << "\n";
    std::cout << "Delete the file to re-download.\n";
    return true;
  }

  std::cout << "========================================\n";
  std::cout << "SmolLM2 Weight Downloader\n";
  std::cout << "========================================\n";
  std::cout << "Model: " << model_name_ << "\n";
  std::cout << "Precision: ";
  switch (precision_) {
  case WeightPrecision::FP32:
    std::cout << "FP32 (4 bytes per param)\n";
    break;
  case WeightPrecision::FP16:
    std::cout << "FP16 (2 bytes per param)\n";
    break;
  case WeightPrecision::BF16:
    std::cout << "BF16 (2 bytes per param)\n";
    break;
  }
  std::cout << "Output: " << output_dir << "\n";
  std::cout << "========================================\n\n";

  // Download tokenizer files first
  if (!download_tokenizer(output_dir)) {
    std::cerr << "Warning: Failed to download some tokenizer files\n";
  }

  // Download model.safetensors
  std::string safetensors_path = output_dir + "/model.safetensors";
  std::string safetensors_url = get_hf_file_url("model.safetensors");

  struct stat buffer;
  if (stat(safetensors_path.c_str(), &buffer) != 0) {
    std::cout << "\nDownloading model weights (this may take a while)...\n";
    std::cout << "URL: " << safetensors_url << "\n\n";

    if (!download_file_with_progress(safetensors_url, safetensors_path)) {
      std::cerr << "Failed to download model weights\n";
      return false;
    }
    std::cout << "\nDownload complete!\n";
  } else {
    std::cout << "\nSafetensors file already exists, using cached version\n";
  }

  // Convert to custom binary format
  std::cout << "\nConverting to optimized binary format...\n";
  if (!convert_safetensors_to_binary(safetensors_path, weights_path)) {
    std::cerr << "Failed to convert weights\n";
    return false;
  }

  // Get file sizes
  struct stat st;
  stat(safetensors_path.c_str(), &st);
  int64_t safetensors_size = st.st_size;

  stat(weights_path.c_str(), &st);
  int64_t binary_size = st.st_size;

  std::cout << "\n========================================\n";
  std::cout << "Export Complete!\n";
  std::cout << "========================================\n";
  std::cout << "Original (safetensors): " << std::fixed << std::setprecision(2)
            << (safetensors_size / 1e9) << " GB\n";
  std::cout << "Converted (binary):     " << std::fixed << std::setprecision(2)
            << (binary_size / 1e9) << " GB\n";
  std::cout << "Weights file: " << weights_path << "\n";
  std::cout << "Tokenizer dir: " << output_dir << "\n";
  std::cout << "========================================\n";

  // Optionally delete the safetensors file to save space
  std::cout << "\nNote: You can delete " << safetensors_path << " to save disk space.\n";

  return true;
}

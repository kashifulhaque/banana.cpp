cmake_minimum_required(VERSION 3.15)
project(llm_inference)

# Using C++20 to work around stdlib linker issues on macOS 26/Xcode 17
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Ensure consistent stdlib on macOS
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS deployment version")
    
    # Workaround for __hash_memory linker issue on Xcode 17+/macOS 26+
    # When using Homebrew LLVM, we need to use its libc++ runtime
    if(CMAKE_CXX_COMPILER MATCHES "/opt/homebrew/opt/llvm" OR CMAKE_CXX_COMPILER MATCHES "/usr/local/opt/llvm")
        get_filename_component(LLVM_ROOT "${CMAKE_CXX_COMPILER}/../.." ABSOLUTE)
        message(STATUS "Using Homebrew LLVM libc++ from: ${LLVM_ROOT}")
        # Use Homebrew LLVM's libc++ headers and libraries
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nostdinc++ -isystem ${LLVM_ROOT}/include/c++/v1")
        link_directories(${LLVM_ROOT}/lib/c++)
        set(CMAKE_BUILD_RPATH "${LLVM_ROOT}/lib/c++")
        set(CMAKE_INSTALL_RPATH "${LLVM_ROOT}/lib/c++")
    endif()
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# ============================================================================
# Source Files - Organized by Module
# ============================================================================

# Core module
set(CORE_SOURCES
    src/core/ops.cpp
)

# Layers module
set(LAYERS_SOURCES
    src/layers/attention.cpp
    src/layers/mlp.cpp
    src/layers/normalization.cpp
    src/layers/position_embedding.cpp
)

# Models module
set(MODELS_SOURCES
    src/models/llm_model.cpp
)

# Tokenizers module
set(TOKENIZERS_SOURCES
    src/tokenizers/bpe_tokenizer.cpp
    src/tokenizers/chat_template.cpp
    src/tokenizers/tokenizer.cpp
)

# Registry module
set(REGISTRY_SOURCES
    src/registry/model_registry.cpp
)

# Utils module
set(UTILS_SOURCES
    src/utils/model_loader.cpp
    src/utils/weight_downloader.cpp
)

# All library sources
set(LLM_LIB_SOURCES
    ${CORE_SOURCES}
    ${LAYERS_SOURCES}
    ${MODELS_SOURCES}
    ${TOKENIZERS_SOURCES}
    ${REGISTRY_SOURCES}
    ${UTILS_SOURCES}
)

# ============================================================================
# Build Configuration
# ============================================================================

# Create static library for reuse
add_library(llm_lib STATIC ${LLM_LIB_SOURCES})

# Create main executable
add_executable(llm_inference src/main.cpp)
target_link_libraries(llm_inference PRIVATE llm_lib)

# Link Accelerate on macOS for fast BLAS
if(APPLE)
    target_compile_definitions(llm_lib PRIVATE ACCELERATE_NEW_LAPACK)
    target_link_libraries(llm_lib PRIVATE "-framework Accelerate")
    target_compile_definitions(llm_inference PRIVATE ACCELERATE_NEW_LAPACK)
    target_link_libraries(llm_inference PRIVATE "-framework Accelerate")
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(llm_lib PRIVATE -Wall -Wextra -O3 -march=native)
    target_compile_options(llm_inference PRIVATE -Wall -Wextra -O3 -march=native)
endif()

# Optional: Enable OpenMP for parallel processing
find_package(OpenMP)

# Apple Clang does not ship OpenMP runtime; try Homebrew libomp
if(APPLE AND NOT OpenMP_CXX_FOUND)
    find_path(OPENMP_INCLUDE_DIR omp.h
        PATHS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
    find_library(OPENMP_LIBRARY omp
        PATHS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
    if(OPENMP_INCLUDE_DIR AND OPENMP_LIBRARY)
        set(OpenMP_CXX_FOUND TRUE)
        set(OpenMP_omp_LIBRARY ${OPENMP_LIBRARY})
    endif()
endif()

if(OpenMP_CXX_FOUND)
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(llm_lib PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(llm_inference PUBLIC OpenMP::OpenMP_CXX)
    elseif(OpenMP_omp_LIBRARY)
        target_compile_options(llm_lib PRIVATE -Xpreprocessor -fopenmp)
        target_compile_options(llm_inference PRIVATE -Xpreprocessor -fopenmp)
        target_link_libraries(llm_lib PUBLIC ${OpenMP_omp_LIBRARY})
        target_link_libraries(llm_inference PUBLIC ${OpenMP_omp_LIBRARY})
        target_include_directories(llm_lib PRIVATE ${OPENMP_INCLUDE_DIR})
        target_include_directories(llm_inference PRIVATE ${OPENMP_INCLUDE_DIR})
    endif()
endif()

# ============================================================================
# Print Build Summary
# ============================================================================
message(STATUS "")
message(STATUS "LLM Inference Engine Build Configuration")
message(STATUS "=========================================")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
if(APPLE)
    message(STATUS "macOS Deployment Target: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
    message(STATUS "Accelerate Framework: Enabled")
endif()
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: Enabled")
else()
    message(STATUS "OpenMP: Not Found")
endif()
message(STATUS "")

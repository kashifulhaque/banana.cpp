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

# Generic LLM inference engine sources
set(LLM_SOURCES
    src/llm_model.cpp
    src/model_loader.cpp
    src/model_registry.cpp
    src/ops.cpp
    src/tokenizer.cpp
    src/weight_downloader.cpp
    src/main.cpp
)

# Create generic LLM inference executable
add_executable(llm_inference ${LLM_SOURCES})

# Link Accelerate on macOS for fast BLAS
if(APPLE)
    target_compile_definitions(llm_inference PRIVATE ACCELERATE_NEW_LAPACK)
    target_link_libraries(llm_inference PRIVATE "-framework Accelerate")
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
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
        target_link_libraries(llm_inference PUBLIC OpenMP::OpenMP_CXX)
    elseif(OpenMP_omp_LIBRARY)
        target_compile_options(llm_inference PRIVATE -Xpreprocessor -fopenmp)
        target_link_libraries(llm_inference PUBLIC ${OpenMP_omp_LIBRARY})
        target_include_directories(llm_inference PRIVATE ${OPENMP_INCLUDE_DIR})
    endif()
endif()

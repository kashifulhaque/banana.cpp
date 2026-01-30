cmake_minimum_required(VERSION 3.15)
project(llm_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# SmolLM2 sources
set(SMOLLM2_SOURCES
    src/model_loader.cpp
    src/ops.cpp
    src/smollm2_main.cpp
    src/smollm2.cpp
    src/smollm2_tokenizer.cpp
)

# Create SmolLM2 executable
add_executable(smollm2_inference ${SMOLLM2_SOURCES})

# Link Accelerate on macOS for fast BLAS
if(APPLE)
    target_compile_definitions(smollm2_inference PRIVATE ACCELERATE_NEW_LAPACK)
    target_link_libraries(smollm2_inference PRIVATE "-framework Accelerate")
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(smollm2_inference PRIVATE -Wall -Wextra -O3 -march=native)
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
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_omp_LIBRARY ${OPENMP_LIBRARY})
    endif()
endif()

if(OpenMP_CXX_FOUND)
    if(OpenMP_CXX_FLAGS)
        target_compile_options(smollm2_inference PRIVATE ${OpenMP_CXX_FLAGS})
    endif()
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(smollm2_inference PUBLIC OpenMP::OpenMP_CXX)
    else()
        target_link_libraries(smollm2_inference PUBLIC ${OpenMP_omp_LIBRARY})
        target_include_directories(smollm2_inference PRIVATE ${OPENMP_INCLUDE_DIR})
    endif()
endif()

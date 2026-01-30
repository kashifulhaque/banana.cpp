cmake_minimum_required(VERSION 3.15)
project(llm_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Ensure consistent stdlib on macOS
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS deployment version")
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

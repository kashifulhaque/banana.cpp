cmake_minimum_required(VERSION 3.15)
project(llm_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Common source files
set(COMMON_SOURCES
    src/model_loader.cpp
    src/ops.cpp
)

# GPT-2 sources
set(GPT2_SOURCES
    ${COMMON_SOURCES}
    src/main.cpp
    src/tokenizer.cpp
    src/gpt2.cpp
)

# SmolLM2 sources
set(SMOLLM2_SOURCES
    ${COMMON_SOURCES}
    src/smollm2_main.cpp
    src/smollm2.cpp
    src/smollm2_tokenizer.cpp
)

# Create GPT-2 executable
add_executable(gpt2_inference ${GPT2_SOURCES})

# Create SmolLM2 executable
add_executable(smollm2_inference ${SMOLLM2_SOURCES})

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(gpt2_inference PRIVATE -Wall -Wextra -O3 -march=native)
    target_compile_options(smollm2_inference PRIVATE -Wall -Wextra -O3 -march=native)
endif()

# Optional: Enable OpenMP for parallel processing
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gpt2_inference PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(smollm2_inference PUBLIC OpenMP::OpenMP_CXX)
endif()

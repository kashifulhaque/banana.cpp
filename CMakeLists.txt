cmake_minimum_required(VERSION 3.15)
project(gpt2_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/main.cpp
    src/model_loader.cpp
    src/ops.cpp
    src/tokenizer.cpp
)

# Create executable
add_executable(gpt2_inference ${SOURCES})

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(gpt2_inference PRIVATE -Wall -Wextra -O3)
endif()

// Benchmark: Accelerate SGEMM vs vDSP vs Metal Performance Shaders
// Compile: clang++ -std=c++17 -O3 -framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework Foundation benchmark_matmul.mm -o benchmark_matmul

#include <Accelerate/Accelerate.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

// Timing helper
class Timer {
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_time;
};

// Fill with random values
void fill_random(std::vector<float>& vec) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : vec) v = dist(gen);
}

// Benchmark Accelerate cblas_sgemm
double benchmark_cblas_sgemm(int M, int N, int K, int iterations) {
    std::vector<float> A(M * K), B(K * N), C(M * N);
    fill_random(A);
    fill_random(B);
    
    // Warmup
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C.data(), N);
    
    Timer timer;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C.data(), N);
    }
    return timer.elapsed_ms() / iterations;
}

// Benchmark vDSP_mmul
double benchmark_vdsp_mmul(int M, int N, int K, int iterations) {
    std::vector<float> A(M * K), B(K * N), C(M * N);
    fill_random(A);
    fill_random(B);
    
    // Warmup
    vDSP_mmul(A.data(), 1, B.data(), 1, C.data(), 1, M, N, K);
    
    Timer timer;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        vDSP_mmul(A.data(), 1, B.data(), 1, C.data(), 1, M, N, K);
    }
    return timer.elapsed_ms() / iterations;
}

// Benchmark Metal Performance Shaders
double benchmark_mps(int M, int N, int K, int iterations) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal device not available" << std::endl;
            return -1;
        }
        
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        
        // Create buffers
        std::vector<float> A(M * K), B(K * N), C(M * N);
        fill_random(A);
        fill_random(B);
        
        id<MTLBuffer> bufferA = [device newBufferWithBytes:A.data()
                                                    length:A.size() * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithBytes:B.data()
                                                    length:B.size() * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [device newBufferWithLength:C.size() * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
        
        // Create matrix descriptors
        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                           columns:K
                                                                          rowBytes:K * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                           columns:N
                                                                          rowBytes:N * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                           columns:N
                                                                          rowBytes:N * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
        
        MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];
        
        // Create matrix multiplication kernel
        MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                            transposeLeft:NO
                                                                           transposeRight:NO
                                                                               resultRows:M
                                                                            resultColumns:N
                                                                          interiorColumns:K
                                                                                    alpha:1.0
                                                                                     beta:0.0];
        
        // Warmup
        id<MTLCommandBuffer> warmupBuffer = [commandQueue commandBuffer];
        [matmul encodeToCommandBuffer:warmupBuffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
        [warmupBuffer commit];
        [warmupBuffer waitUntilCompleted];
        
        // Benchmark
        Timer timer;
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
            [matmul encodeToCommandBuffer:cmdBuffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
        }
        return timer.elapsed_ms() / iterations;
    }
}

// Benchmark MPS with batched commands (more realistic for inference)
double benchmark_mps_batched(int M, int N, int K, int iterations) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return -1;
        
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        
        std::vector<float> A(M * K), B(K * N), C(M * N);
        fill_random(A);
        fill_random(B);
        
        id<MTLBuffer> bufferA = [device newBufferWithBytes:A.data()
                                                    length:A.size() * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithBytes:B.data()
                                                    length:B.size() * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [device newBufferWithLength:C.size() * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
        
        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];
        
        MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];
        
        MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                            transposeLeft:NO transposeRight:NO
                                                                               resultRows:M resultColumns:N interiorColumns:K
                                                                                    alpha:1.0 beta:0.0];
        
        // Warmup
        id<MTLCommandBuffer> warmupBuffer = [commandQueue commandBuffer];
        [matmul encodeToCommandBuffer:warmupBuffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
        [warmupBuffer commit];
        [warmupBuffer waitUntilCompleted];
        
        // Batch all operations into single command buffer
        Timer timer;
        timer.start();
        id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
        for (int i = 0; i < iterations; ++i) {
            [matmul encodeToCommandBuffer:cmdBuffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
        }
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
        return timer.elapsed_ms() / iterations;
    }
}

void run_benchmark(int M, int N, int K, int iterations) {
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "Matrix size: [" << M << " x " << K << "] * [" << K << " x " << N << "] = [" << M << " x " << N << "]\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    
    double cblas_time = benchmark_cblas_sgemm(M, N, K, iterations);
    double vdsp_time = benchmark_vdsp_mmul(M, N, K, iterations);
    double mps_time = benchmark_mps(M, N, K, iterations);
    double mps_batched_time = benchmark_mps_batched(M, N, K, iterations);
    
    // Calculate GFLOPS (2*M*N*K operations for matmul)
    double flops = 2.0 * M * N * K;
    double cblas_gflops = (flops / (cblas_time * 1e-3)) / 1e9;
    double vdsp_gflops = (flops / (vdsp_time * 1e-3)) / 1e9;
    double mps_gflops = (flops / (mps_time * 1e-3)) / 1e9;
    double mps_batched_gflops = (flops / (mps_batched_time * 1e-3)) / 1e9;
    
    std::cout << "\nResults (average per iteration):\n";
    std::cout << "┌────────────────────────┬────────────────┬────────────────┬──────────┐\n";
    std::cout << "│ Method                 │ Time (ms)      │ GFLOPS         │ Speedup  │\n";
    std::cout << "├────────────────────────┼────────────────┼────────────────┼──────────┤\n";
    printf("│ %-22s │ %14.4f │ %14.2f │ %8.2fx │\n", "Accelerate (cblas)", cblas_time, cblas_gflops, 1.0);
    printf("│ %-22s │ %14.4f │ %14.2f │ %8.2fx │\n", "vDSP_mmul", vdsp_time, vdsp_gflops, cblas_time / vdsp_time);
    printf("│ %-22s │ %14.4f │ %14.2f │ %8.2fx │\n", "MPS (per-call sync)", mps_time, mps_gflops, cblas_time / mps_time);
    printf("│ %-22s │ %14.4f │ %14.2f │ %8.2fx │\n", "MPS (batched)", mps_batched_time, mps_batched_gflops, cblas_time / mps_batched_time);
    std::cout << "└────────────────────────┴────────────────┴────────────────┴──────────┘\n";
}

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     Matrix Multiplication Benchmark - Apple Silicon            ║\n";
    std::cout << "║     Accelerate vs vDSP vs Metal Performance Shaders            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    
    // Small matrices (typical for single token inference)
    run_benchmark(1, 576, 576, 1000);      // Single token, SmolLM2 hidden size
    run_benchmark(1, 1536, 576, 1000);     // FFN up projection
    
    // Medium matrices (small batch or multi-token)
    run_benchmark(32, 576, 576, 100);
    run_benchmark(128, 576, 576, 100);
    
    // Large matrices (prefill / large batch)
    run_benchmark(512, 576, 576, 50);
    run_benchmark(1024, 576, 576, 20);
    run_benchmark(2048, 576, 576, 10);
    
    // Typical LLM sizes (larger models)
    run_benchmark(1, 2048, 2048, 100);     // Single token, larger model
    run_benchmark(128, 2048, 2048, 20);    // Batch inference
    run_benchmark(512, 4096, 4096, 10);    // Large model prefill
    
    std::cout << "\n✓ Benchmark complete!\n";
    std::cout << "\nNotes:\n";
    std::cout << "- MPS (per-call sync): Each matmul waits for GPU completion (high overhead)\n";
    std::cout << "- MPS (batched): Multiple matmuls in one command buffer (realistic for inference)\n";
    std::cout << "- For small matrices, CPU (Accelerate) often wins due to GPU dispatch overhead\n";
    std::cout << "- For large matrices, MPS typically provides significant speedup\n";
    
    return 0;
}

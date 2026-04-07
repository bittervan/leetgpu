#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        const cudaError_t error__ = (call);                                                    \
        if (error__ != cudaSuccess) {                                                          \
            std::ostringstream oss__;                                                          \
            oss__ << "CUDA error: " << cudaGetErrorString(error__) << " at " << __FILE__       \
                  << ":" << __LINE__;                                                          \
            throw std::runtime_error(oss__.str());                                             \
        }                                                                                      \
    } while (false)

__global__ void reduction_kernel(const float* input, float* output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = idx * 256;
    float temp = 0;
    for (int i = 0; i < 256; i++) {
        if (offset + i < N) {
            temp += input[offset + i];
        }
    }
    if (offset < N)
        output[idx] = temp;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    static const int BLOCK_SIZE = 256;
    float* result = nullptr;
    const float* previous_result = input;

    while (1) {
        int n_threads = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaMalloc(&result, n_threads * sizeof(float));
        dim3 grid_size((n_threads + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
        dim3 block_size(BLOCK_SIZE, 1, 1);
        reduction_kernel<<<grid_size, block_size>>>(previous_result, result, N);
        N = n_threads;
        if (previous_result != input) {
            cudaFree((void*)previous_result);
        }
        previous_result = result;
        if (n_threads == 1) {
            break;
        }
    }

    cudaDeviceSynchronize();
    cudaMemcpy(output, result, sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(result);
    cudaDeviceSynchronize();
}

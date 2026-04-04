#include <cuda_runtime.h>

#include <curand_mtgp32_kernel.h>
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

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output, int input_size,
                                      int kernel_size) {
    // TODO: implement valid 1D convolution.
    // output[i] = sum(input[i + j] * kernel[j]), 0 <= j < kernel_size.
    __shared__ float kernel_data[4096];
    __shared__ float required_input[4096 + 255];
    int output_size = input_size - kernel_size + 1;

    int kernel_copy_time = (kernel_size + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < kernel_copy_time; i++) {
        int index = i * blockDim.x + threadIdx.x;
        if (index < kernel_size) 
            kernel_data[index] = kernel[index];
    }

    int private_input_size = kernel_size + blockDim.x - 1;
    int input_copy_time = (private_input_size + blockDim.x - 1) / blockDim.x;
    int offset = blockDim.x * blockIdx.x;
    for (int i = 0; i < input_copy_time; i++) {
        int index = i * blockDim.x + threadIdx.x;
        if (index < private_input_size)
            required_input[index] = input[index + offset];
    }

    __syncthreads();

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    // if (index < output_size) output[index] = 0;
    int tx = threadIdx.x;
    float p_val = 0;

    for (int i = 0; i < kernel_size; i++) {
        p_val += required_input[tx + i] * kernel_data[i];
    }
    
    if (index < output_size) {
        output[index] = p_val;
    }
    
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

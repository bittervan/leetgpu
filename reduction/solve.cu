#include <cuda_runtime.h>

__global__ void reduction_kernel(const float* input, float* output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int p_index = threadIdx.x;
    __shared__ float s_buf[256];
    int current_size = 128;

    if (index < N) {
        s_buf[p_index] = input[index];
    } else {
        s_buf[p_index] = 0;
    }
    __syncthreads();

    while (current_size) {
        if (p_index < current_size) {
            s_buf[p_index] += s_buf[p_index + current_size];
        }
        __syncthreads();
        current_size /= 2;
    }

    if (p_index == 0)
        output[blockIdx.x] = s_buf[0];
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int current_size = N;
    int malloc_size = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* mutable_input = nullptr;
    cudaError_t result = cudaMalloc(&mutable_input, N * sizeof(float));
    cudaMemcpy(mutable_input, input, N * sizeof(float), cudaMemcpyDeviceToDevice);

    float* pivot_buffer = nullptr;
    result = cudaMalloc(&pivot_buffer, malloc_size * sizeof(float));

    while (current_size > 1) {
        int blocksPerGrid = (current_size + threadsPerBlock - 1) / threadsPerBlock;
        reduction_kernel<<<blocksPerGrid, threadsPerBlock>>>(mutable_input, pivot_buffer, current_size);
        cudaMemcpy(mutable_input, pivot_buffer, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToDevice);
        current_size = blocksPerGrid;
    }

    cudaMemcpy(output, mutable_input, sizeof(float), cudaMemcpyDeviceToDevice);
}

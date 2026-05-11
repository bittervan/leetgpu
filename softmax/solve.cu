#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <vector>

__global__ void softmax_kernel(const float* input, float* output, int N, float sum) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        output[idx] = input[idx] / sum;
    } 
}

__global__ void getmax_kernel(const float* input, float* output, int N) {
    extern __shared__ float s_data[];
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int p_index = threadIdx.x;
    if (index < N) {
        s_data[p_index] = input[index];
    } else {
        s_data[p_index] = -FLT_MAX;
    }
    __syncthreads();

    int current_stride = blockDim.x / 2;

    while (current_stride) {
        if (p_index < current_stride) {
            s_data[p_index] = fmaxf(s_data[p_index], s_data[p_index + current_stride]);
        }
        __syncthreads();
        current_stride /= 2;
    }

    output[blockIdx.x] = s_data[0];
}

__global__ void elementwise_sub_and_exp_kernel(const float *input, float *output, float max, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N)
        output[index] = expf(input[index] - max);
}

void getmax(const float* input, float *pivot, float* output, int N, int threadsPerBlock) {
    int current_size = N;
    int blocksPerGrid = (current_size + threadsPerBlock - 1) / threadsPerBlock;
    const float* current_input = input;

    while (current_size > 1) {
        getmax_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(current_input, pivot, current_size);
        current_size = blocksPerGrid;
        blocksPerGrid = (current_size + threadsPerBlock - 1) / threadsPerBlock;
        current_input = pivot;
    }

    cudaMemcpy(output, current_input, sizeof(float), cudaMemcpyDeviceToHost);
}

void elementwise_sub_and_exp(const float *input, float *output, int N, int threadsPerBlock, float max) {
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_sub_and_exp_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, max, N);
}

__global__ void reduction_kernel(const float *input, float *output, int N) {
    extern __shared__ float s_data[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int p_idx = threadIdx.x;
    if (idx < N) {
        s_data[p_idx] = input[idx];
    } else {
        s_data[p_idx] = 0;
    }
    __syncthreads();

    int current_stride = blockDim.x / 2;
    while (current_stride) {
        if (p_idx < current_stride) {
            s_data[p_idx] += s_data[p_idx + current_stride];
        }
        __syncthreads();
        current_stride /= 2;
    }

    output[blockIdx.x] = s_data[0];
}

float reduction(const float *input, float *pivot, int N, int threadsPerBlock) {
    float ret = 0;
    const float *current_input = input;

    int current_size = N;
    int blocksPerGrid = (current_size + threadsPerBlock - 1) / threadsPerBlock;

    while (current_size > 1) {
        reduction_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(current_input, pivot, current_size);
        current_input = pivot;
        current_size = blocksPerGrid;
        blocksPerGrid = (current_size + threadsPerBlock - 1) / threadsPerBlock;
    }

    cudaMemcpy(&ret, current_input, sizeof(float), cudaMemcpyDeviceToHost);
    return ret;
}

void debug_print_array(const float* to_print, int N) {
    std::vector<float> h(N);
    cudaMemcpy(h.data(), to_print, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("[");
    for (int i = 0; i < N; ++i) printf("%f ", h[i]);
    printf("]\n");
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* pivot_buffer = nullptr;
    cudaMalloc(&pivot_buffer, N * sizeof(float));

    // printf("Input: "); debug_print_array(input, N);
    float max_val = -FLT_MAX;
    getmax(input, pivot_buffer, &max_val, N, threadsPerBlock);
    cudaDeviceSynchronize();
    // printf("Output: "); debug_print_array(pivot_buffer, N);
    // printf("Max: %f\n", max_val);

    elementwise_sub_and_exp(input, output, N, threadsPerBlock, max_val);

    float sum = reduction(output, pivot_buffer, N, threadsPerBlock);

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(output, output, N, sum);
    cudaDeviceSynchronize();

    cudaFree(pivot_buffer);
}

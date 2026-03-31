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

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
    // TODO: implement
    __shared__ float ds_A[16][16];
    __shared__ float ds_B[16][16];
    float p_val = 0;

    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    for (int i = 0; i < K; i += 16) {
        if (i + threadIdx.x < K && row < M) {
            ds_A[threadIdx.y][threadIdx.x] = A[row * K + (i + threadIdx.x)];
        } else {
            ds_A[threadIdx.y][threadIdx.x] = 0;
        }

        if (i + threadIdx.y < K && col < N) {
            ds_B[threadIdx.y][threadIdx.x] = B[(i + threadIdx.y) * N + col];
        } else {
            ds_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();
        for (int j = 0; j < 16; j++) {
            p_val += ds_A[threadIdx.y][j] * ds_B[j][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = p_val;
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x,
                         (M + threads_per_block.y - 1) / threads_per_block.y);

    matrix_multiplication_kernel<<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

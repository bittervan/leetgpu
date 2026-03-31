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

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
    // TODO: implement matrix multiplication where:
    // A is MxN, B is NxK, C is MxK.
    __shared__ float ds_A[16][16];
    __shared__ float ds_B[16][16];
    float p_val = 0;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    for (int i = 0; i < N; i += 16) {
        if (i + tx < N && row < M) {
            ds_A[ty][tx] = A[row * N + i + tx];
        } else {
            ds_A[ty][tx] = 0;
        }

        if (i + ty < N && col < K) {
            ds_B[ty][tx] = B[(i + ty) * K + col];
        } else {
            ds_B[ty][tx] = 0;
        }

        __syncthreads();
        for (int j = 0; j < 16; j++) {
            p_val += ds_A[ty][j] * ds_B[j][tx];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = p_val;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

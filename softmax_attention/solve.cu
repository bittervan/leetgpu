#include <cuda_runtime.h>

__global__ void matmul_kernel(const float *A, const float *B, float *output, int M, int N, int K) {
    __shared__ float s_A[16][16];
    __shared__ float s_B[16][16];
    __shared__ float s_C[16][16];

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    s_C[ty][tx] = 0;

    __syncthreads();

    for (int bk = 0; bk < K; bk += 16) {
        int ax = bk * 16 + tx;
        int by = bk * 16 + ty;

        if (y < M && ax < K)
            s_A[ty][tx] = A[y * K + ax];
        else
            s_A[ty][tx] = 0;

        if (x < N && by < K)
            s_B[ty][tx] = B[by * N + x];
        else
            s_B[ty][tx] = 0;

        __syncthreads();

        for (int i = 0; i < 16; i++) 
            s_C[ty][tx] += s_A[ty][i] * s_B[i][tx];

        __syncthreads();
    }

    if (y < M && x < N)
        output[y * N + x] = s_C[ty][tx];
    __syncthreads();
}

// A is M * N, B is N * K
void matmul(const float* A, const float *B, float *output, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, M, N, K);
}

__global__ void transpose_kernel(const float *input, float *output, int M, int N) {
    __shared__ float s_data[16][16];
    __shared__ float s_trans[16][17];

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (x < N && y < M) {
        s_data[ty][tx] = input[y * N + x];
    }
    __syncthreads();
    s_trans[tx][ty] = s_data[ty][tx];

    int new_x = threadIdx.x + blockDim.y * blockIdx.y;
    int new_y = threadIdx.y + blockDim.x * blockIdx.x;

    if (new_x < M && new_y < N) {
        output[new_y * M + new_x] = s_trans[ty][tx];
    }
}

void transpose(const float *input, float *output, int M, int N) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, M, N);
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {}

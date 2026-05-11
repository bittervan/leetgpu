#include <cuda_runtime.h>
#include <cfloat>

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

__global__ void reduction_kernel(const float *input, float *output, int N) {
    __shared__ float s_data[256];
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int tx = threadIdx.x;
    if (index < N)
        s_data[tx] = input[index];
    else
        s_data[tx] = 0;

    __syncthreads();

    int stride = 128;
    while (stride) {
        if (tx < stride) {
            s_data[tx] += s_data[tx + stride];
        }
        __syncthreads();
        stride /= 2;
    }
    output[blockIdx.x] = s_data[0];
}

float reduction(const float *input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int current_size = N;
    const float *current_input = input;
    float *pivot_buffer = nullptr;
    cudaMalloc(&pivot_buffer, sizeof(float) * blocksPerGrid);

    while (current_size > 1) {
        reduction_kernel<<<blocksPerGrid, threadsPerBlock>>>(current_input, pivot_buffer, current_size);
        current_size = blocksPerGrid;
        blocksPerGrid = (current_size + threadsPerBlock - 1) / threadsPerBlock;
        current_input = pivot_buffer;
    }

    float ret = 0;
    cudaMemcpy(&ret, current_input, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(pivot_buffer);
    return ret;
}

__global__ void getmax_kernel(const float *input, float *output, int N) {
    __shared__ float s_data[256];
    int tx = threadIdx.x;
    int index = tx + blockIdx.x + blockDim.x;

    if (index < N)
        s_data[tx] = input[index];
    else
        s_data[tx] = -FLT_MAX;

    __syncthreads();

    int current_stride = 128;
    while (current_stride) {
        if (tx < current_stride) {
            s_data[tx] = fmaxf(s_data[tx], s_data[tx + current_stride]);
        }
        __syncthreads();
        current_stride /= 2;
    }

    output[blockIdx.x] = s_data[0];
}

float getmax(const float *input, int N) {
    int threadsPerBlock(256);
    int blocksPerGrid((N + threadsPerBlock - 1) / threadsPerBlock);

    const float *current_input = input;
    float *pivot_buffer = nullptr;
    cudaMalloc(&pivot_buffer, blocksPerGrid * sizeof(float));

    int current_size = N;

    while (current_size > 1) {
        getmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(current_input, pivot_buffer, current_size);
        current_size = blocksPerGrid;
        blocksPerGrid =(current_size + threadsPerBlock - 1) / threadsPerBlock;
        current_input = pivot_buffer;
    }

    float ret = 0;
    cudaMemcpy(&ret, current_input, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(pivot_buffer);
    return ret;
}

__global__ void elementwise_divide(float *data, float divider, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        data[index] /= divider;
    }
}


// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    float *K_T = nullptr;
    cudaMalloc(&K_T, N * d * sizeof(float));
    transpose(K, K_T, N, d);

    float *Q_K_T = nullptr;
    cudaMalloc(&Q_K_T, M * N * sizeof(float));
    matmul(Q, K_T, Q_K_T, M, N, d);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N * M + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_divide<<<blocksPerGrid, threadsPerBlock>>>(Q_K_T, sqrtf(d), N * M);


    float *smax = nullptr;
    cudaMalloc(&smax, M * N * sizeof(float));
    softmax(Q_K_T, smax, M * N);

    matmul(smax, V, output, M, d, N);

    cudaFree(smax);
    cudaFree(Q_K_T);
    cudaFree(K_T);
}

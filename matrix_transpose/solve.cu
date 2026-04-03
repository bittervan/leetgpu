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

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    // TODO: implement matrix transpose.
    __shared__ float s_input[16][17];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;

    int ox = blockIdx.y * blockDim.y + tx;
    int oy = blockIdx.x * blockDim.x + ty;

    if (gx < cols && gy < rows) {
        s_input[ty][tx] = input[cols * gy + gx];        
    }

    __syncthreads();

    if (ox < rows && oy < cols) {
        output[rows * oy + ox] = s_input[tx][ty];
    }

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

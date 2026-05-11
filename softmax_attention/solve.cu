#include <cuda_runtime.h>

// A is M * N, B is N * K
__global__ void matmul(const float* A, const float *B, int M, int N, int K) {

}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {}

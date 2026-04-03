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

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    // TODO: invert the image in-place.
    // image is a row-major RGBA buffer with width * height * 4 values.
    // Only invert R, G, B. Keep A unchanged.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width * height) {
        for (int i = 0; i < 3; i++) {
            image[idx * 4 + i] = 255 - image[idx * 4 + i];
        }
    }
}

// image is a device pointer (i.e. a pointer to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

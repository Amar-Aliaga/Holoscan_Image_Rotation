#include "cuda_blur.cuh"
#include <cuda_runtime.h>
#include <cstdint>


__global__ void blur_kernel(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int sum = 0;
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int px = x + kx;
                int py = y + ky;
                sum += input[py * width + px];
            }
        }
        output[y * width + x] = sum / 9;
    }
}


inline void launch_blur_kernel(const unsigned char* input, unsigned char* output,
							   int width, int height,
							   cudaStream_t stream,
							   dim3 grid, dim3 block) {
	blur_kernel<<<grid, block, 0, stream>>>(input, output, width, height);
}

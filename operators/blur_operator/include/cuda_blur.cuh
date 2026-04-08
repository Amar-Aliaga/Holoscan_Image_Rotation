#pragma once

#include <cuda_runtime.h>
#include <cstdint>


__global__ void blur_kernel(const unsigned char* input, unsigned char* output, int width, int height);

inline void launch_blur_kernel(const unsigned char* input, unsigned char* output,
							   int width, int height,
							   cudaStream_t stream,
							   dim3 grid, dim3 block);
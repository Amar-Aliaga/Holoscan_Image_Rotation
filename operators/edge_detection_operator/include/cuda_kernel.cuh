#pragma once

#include <cuda_runtime.h>
#include <cstdint>

__global__ void rgba_to_laplacian(
    const uint8_t* __restrict__ rgba,   // H x W x 4
    uint8_t*      __restrict__ grey,    // H x W
    int rows, int cols, int rgba_pitch, int grey_pitch);


void launch_rgba_to_laplacian(const uint8_t* rgba, uint8_t* grey,
                              int rows, int cols,
                              int rgba_pitch, int grey_pitch,
                              cudaStream_t stream,
                              dim3 grid, dim3 block);


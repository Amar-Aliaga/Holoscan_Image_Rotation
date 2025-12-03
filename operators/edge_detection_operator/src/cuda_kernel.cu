#include "cuda_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdint>


__global__ void rgba_to_laplacian(
    const uint8_t* __restrict__ rgba,   // H x W x 4
    uint8_t*      __restrict__ grey,    // H x W
    int rows, int cols, int rgba_pitch, int grey_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    // read one RGBA pixel
    const uchar4 px = *reinterpret_cast<const uchar4*>(
        rgba + y * rgba_pitch + x * 4);

    // very cheap Laplacian (4-neighbour) on grey value
    // here we just show intensity; real kernel would use shared memory
    uint8_t v = (px.x + px.y + px.z) / 3;   // quick grey
    grey[y * grey_pitch + x] = v;           // place-holder
}


void launch_rgba_to_laplacian(const uint8_t* rgba, uint8_t* grey,
                              int rows, int cols,
                              int rgba_pitch, int grey_pitch,
                              dim3 grid, dim3 block)
{
    rgba_to_laplacian<<<grid, block>>>(rgba, grey, rows, cols, rgba_pitch, grey_pitch);
    cudaDeviceSynchronize(); 
}
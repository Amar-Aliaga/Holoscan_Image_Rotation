#include "cuda_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <npp.h>             
#include <nppcore.h>     
#include <nppi.h>  


__global__ void rgba_to_laplacian(
    const uint8_t* __restrict__ rgba,  // input RGBA H x W x 4
    uint8_t*      __restrict__ out,   // output grayscale Laplacian H x W
    int rows, int cols,
    int rgba_pitch, int out_pitch) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || y <= 0 || x >= cols - 1 || y >= rows - 1)
        return; // ignore borders

    // --- Step 1: Convert RGBA -> grayscale ---
    uchar4 px = *reinterpret_cast<const uchar4*>(rgba + y * rgba_pitch + x * 4);
    uint8_t gray = static_cast<uint8_t>(0.299f * px.x + 0.587f * px.y + 0.114f * px.z);

    //uint8_t gray = (px.x + px.y + px.z) / 3;

    // --- Step 2: Compute Laplacian on grayscale ---
    // Read neighbors from input RGBA (convert to grayscale on the fly)
    uchar4 left_px   = *reinterpret_cast<const uchar4*>(rgba + y * rgba_pitch + (x - 1) * 4);
    uchar4 right_px  = *reinterpret_cast<const uchar4*>(rgba + y * rgba_pitch + (x + 1) * 4);
    uchar4 up_px     = *reinterpret_cast<const uchar4*>(rgba + (y - 1) * rgba_pitch + x * 4);
    uchar4 down_px   = *reinterpret_cast<const uchar4*>(rgba + (y + 1) * rgba_pitch + x * 4);

    int left  = (left_px.x + left_px.y + left_px.z) / 3;
    int right = (right_px.x + right_px.y + right_px.z) / 3;
    int up    = (up_px.x + up_px.y + up_px.z) / 3;
    int down  = (down_px.x + down_px.y + down_px.z) / 3;

    int lap = 4 * gray - left - right - up - down;

    // Clamp to 0–255
    lap = max(0, min(255, lap));

    // Write result
    out[y * out_pitch + x] = static_cast<uint8_t>(lap);
}


void launch_rgba_to_laplacian(const uint8_t* rgba, uint8_t* out,
                                       int rows, int cols,
                                       int rgba_pitch, int out_pitch,
                                       dim3 grid, dim3 block) {
    rgba_to_laplacian<<<grid, block>>>(rgba, out, rows, cols, rgba_pitch, out_pitch);
    cudaDeviceSynchronize();  // or use streams for async
}

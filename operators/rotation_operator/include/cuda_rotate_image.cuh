#pragma once

#include <cuda_runtime.h>
#include <cstdint>


namespace CudaRotate {
    __global__ void rotate_90_kernel(const uint8_t* src, uint8_t* dst,
                                    int W, int H, int C);


    __global__ void rotate_180_kernel(uint8_t* image_data,
                                        int W, int H, int C) ;


    __global__ void rotate_270_kernel(const uint8_t* src, uint8_t* dst,
                                    int W, int H, int C);
};

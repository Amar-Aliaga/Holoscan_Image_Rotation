#include "cuda_rotate_image.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace CudaRotate {
  __global__ void rotate_90_kernel(const uint8_t* src, uint8_t* dst,
                                      int W, int H, int C) {
      int x = blockIdx.x * blockDim.x + threadIdx.x; // col
      int y = blockIdx.y * blockDim.y + threadIdx.y; // row
      if (x >= W || y >= H) return;

      int dst_row = x;
      int dst_col = (H - 1) - y;
      int dst_width = H;

      int src_base = (y * W + x) * C;
      int dst_base = (dst_row * dst_width + dst_col) * C;

      for (int c = 0; c < C; ++c)
        dst[dst_base + c] = src[src_base + c];
  }


  __global__ void rotate_180_kernel(uint8_t* image_data,
                                              int W, int H, int C) {
      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;

      if (x >= W || y >= H) return;

      int half = W / 2;

      if (x >= half) {
          int src_x = x;
          int src_y = y;
          int dst_x = W - 1 - (x - half) + half;
          int dst_y = H - 1 - y;

          if (dst_x > src_x || (dst_x == src_x && dst_y > src_y)) {
              for (int channels = 0; channels < C; ++channels) {
                  int src_idx = (src_y * W + src_x) * C + channels;
                  int dst_idx = (dst_y * W + dst_x) * C + channels;
                  uint8_t tmp = image_data[src_idx];
                  image_data[src_idx] = image_data[dst_idx];
                  image_data[dst_idx] = tmp;
              }
          }
      }
  }


  __global__ void rotate_270_kernel(const uint8_t* src, uint8_t* dst,
                                      int W, int H, int C) {
      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;
      if (x >= W || y >= H) return;

      int dst_row = (W - 1) - x;
      int dst_col = y;
      int dst_width = H;
      int src_base = (y * W + x) * C;
      int dst_base = (dst_row * dst_width + dst_col) * C;
      for (int c = 0; c < C; ++c) dst[dst_base + c] = src[src_base + c];
  }
} // namespace CudaRotat 


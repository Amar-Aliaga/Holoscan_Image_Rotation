#pragma once

#include <holoscan/holoscan.hpp>
#include <cuda_runtime.h>
#include <3rdparty/dlpack/dlpack.h>
#include <memory>


DLManagedTensor* make_tensor_from_cuda(uint8_t* data, int H, int W, int C); 


enum class RotationType { ROTATE_90, ROTATE_180, ROTATE_270 };


void launch_rotation_kernel(RotationType type, const uint8_t* src, uint8_t* dst, int W, int H, int C, dim3 grid, dim3 block);


class RotationOperator : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RotationOperator)

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;
  void stop() override;
               
  private:
    uint8_t* dst {nullptr};
    holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
    std::unique_ptr<nvidia::gxf::MemoryBuffer> memory_buffer_;
    size_t bytes = 480 * 640;

  int rotation_mode_ = 0;

};



// #include "rotation_operator.hpp"
// #include <cuda_runtime.h>
// #include <holoscan/holoscan.hpp>
// #include <iostream>

// using namespace holoscan;

// // --- CUDA kernel: rotates right half of image 180 degrees ---
// __global__ void rotate_right_half_180_kernel(uint8_t* image_data,
//                                              int width, int height, int channels) {
//   int x = blockIdx.x * blockDim.x + threadIdx.x;
//   int y = blockIdx.y * blockDim.y + threadIdx.y;

//   if (x >= width || y >= height) return;

//   int half = width / 2;

//   // Process right half and swap with corresponding rotated position
//   if (x >= half) {
//     int src_x = x;
//     int src_y = y;
//     int dst_x = width - 1 - (x - half) + half;  // mirror within right half
//     int dst_y = height - 1 - y;                  // mirror vertically
    
//     // Only process each pair once
//     if (dst_x > src_x || (dst_x == src_x && dst_y > src_y)) {
//       for (int c = 0; c < channels; ++c) {
//         int src_idx = (src_y * width + src_x) * channels + c;
//         int dst_idx = (dst_y * width + dst_x) * channels + c;
        
//         uint8_t temp = image_data[src_idx];
//         image_data[src_idx] = image_data[dst_idx];
//         image_data[dst_idx] = temp;
//       }
//     }
//   }
// }

// void RotationOperator::setup(OperatorSpec& spec) {
//   spec.input<holoscan::gxf::Entity>("in");
//   spec.output<holoscan::gxf::Entity>("out");
// }

// void RotationOperator::compute(InputContext& op_input,
//                                OutputContext& op_output,
//                                ExecutionContext& context) {
//   // Receive input entity
//   auto in_entity = op_input.receive<holoscan::gxf::Entity>("in");
//   if (!in_entity) {
//     HOLOSCAN_LOG_ERROR("Failed to receive input entity");
//     return;
//   }

//   // Get the tensor from the entity - FIX: std::shared_ptr doesn't have 
//   auto in_tensor = in_entity->get<holoscan::Tensor>();
//   if (!in_tensor) {
//     // Try common alternative names
//     in_tensor = in_entity->get<holoscan::Tensor>("signal");
//     if (!in_tensor) {
//       in_tensor = in_entity->get<holoscan::Tensor>("video");
//       if (!in_tensor) {
//         HOLOSCAN_LOG_ERROR("No tensor found in entity");
//         return;
//       }
//     }
//   }

//   // FIX: in_tensor is already a std::shared_ptr<Tensor>, no need for 
//   auto tensor_ptr = in_tensor;
//   auto shape = tensor_ptr->shape();
  
//   if (shape.size() < 2) {
//     HOLOSCAN_LOG_ERROR("Tensor shape too small");
//     return;
//   }

//   // Handle different shape formats: [H,W,C] or [H,W]
//   int height = static_cast<int>(shape[0]);
//   int width = static_cast<int>(shape[1]);
//   int channels = (shape.size() > 2) ? static_cast<int>(shape[2]) : 1;

//   HOLOSCAN_LOG_DEBUG("Processing tensor: {}x{}x{}", height, width, channels);

//   // FIX: Simplified device check for SDK 3.5.0
//   // Just assume it's on GPU or let the kernel handle it
//   // Remove the device type check that was causing errors

//   // Launch CUDA kernel
//   dim3 block(16, 16);
//   dim3 grid((width + block.x - 1) / block.x,
//             (height + block.y - 1) / block.y);

//   rotate_right_half_180_kernel<<<grid, block>>>(
//       static_cast<uint8_t*>(tensor_ptr->data()),
//       width, height, channels);

//   // Check for CUDA errors
//   cudaError_t cuda_status = cudaGetLastError();
//   if (cuda_status != cudaSuccess) {
//     HOLOSCAN_LOG_ERROR("CUDA kernel launch failed: {}", cudaGetErrorString(cuda_status));
//     return;
//   }

//   cuda_status = cudaDeviceSynchronize();
//   if (cuda_status != cudaSuccess) {
//     HOLOSCAN_LOG_ERROR("CUDA synchronization failed: {}", cudaGetErrorString(cuda_status));
//     return;
//   }

//   // Emit the modified entity
//   op_output.emit(*in_entity, "out");
// }
// #include "rotation_operator.hpp"
// #include "cuda_rotate_image.cuh"

// #include <functional>
// #include <cuda_runtime.h>
// #include <holoscan/holoscan.hpp>
// #include <iostream>
// #include <3rdparty/dlpack/dlpack.h>
// #include <memory>


// using namespace holoscan;


// void launch_rotation_kernel(RotationType type, const uint8_t* src, uint8_t* dst, int W, int H, int C, dim3 grid, dim3 block) {
//     switch (type) {
//         case RotationType::ROTATE_90:   
//             CudaRotate::rotate_90_kernel<<<grid, block>>>(src, dst, W, H, C);
//             break;
//         case RotationType::ROTATE_180:
//             CudaRotate::rotate_180_kernel<<<grid, block>>>(dst, W, H, C);
//             break;
//         case RotationType::ROTATE_270:
//             CudaRotate::rotate_270_kernel<<<grid, block>>>(src, dst, W, H, C);
//             break;
//     }
//     cudaDeviceSynchronize();
// }


// void RotationOperator::setup(OperatorSpec& spec) {
//     spec.input<holoscan::gxf::Entity>("in");
//     spec.output<holoscan::gxf::Entity>("out");
//     spec.param(allocator_, "allocator", "device_allocator", "Device Allocator");
// }


// void RotationOperator::initialize() {   
//     memory_buffer_ = std::make_unique<nvidia::gxf::MemoryBuffer>();
//     holoscan::Operator::initialize();
// }

// void RotationOperator::compute(InputContext& op_input,
//                                OutputContext& op_output,
//                                ExecutionContext& context) {

//     auto in_entity = op_input.receive<holoscan::gxf::Entity>("in");
//     if (!in_entity) return;

//     auto in_tensor_opt = in_entity->get<Tensor>();
//     if (!in_tensor_opt) in_tensor_opt = in_entity->get<Tensor>("video");
//     if (!in_tensor_opt) {
//         HOLOSCAN_LOG_ERROR("No tensor in input");
//         return;
//     }

//     //auto in_tensor = in_tensor_opt;

//     auto shape = in_tensor_opt->shape();
//     int H = shape[0];
//     int W = shape[1];
//     int C = (shape.size() > 2 ? shape[2] : 1);

//     size_t bytes = static_cast<size_t>(H) * W * C;

//     uint8_t* src = static_cast<uint8_t*>(in_tensor_opt->data());

//     // ------------ Rotation kernel ------------
//     dim3 block(16, 16);
//     dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

//     RotationType rotation_type = RotationType::ROTATE_180;

//     launch_rotation_kernel(rotation_type, src, dst, W, H, C, grid, block);
//     cudaError_t err = cudaDeviceSynchronize();
//     if (err != cudaSuccess) {
//         HOLOSCAN_LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
//     }

//     auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
//         context.context(), allocator_.get()->gxf_cid()).value();

//     if (memory_buffer_->size() < bytes) {
//         memory_buffer_->resize(gxf_allocator, bytes,
//                                nvidia::gxf::MemoryStorageType::kDevice);
//     }

//     uint8_t* dst = static_cast<uint8_t*>(memory_buffer_->pointer());

//     // ------------ Create output tensor ------------
//     auto out_entity = holoscan::gxf::Entity::New(&context);
//     auto& gxf_out = static_cast<nvidia::gxf::Entity&>(out_entity);
//     auto tensor_out = gxf_out.add<nvidia::gxf::Tensor>("rotated").value();

//     tensor_out->reshape<uint8_t>(
//         nvidia::gxf::Shape({H, W, C}),
//         nvidia::gxf::MemoryStorageType::kDevice,
//         gxf_allocator
//     );

//     auto *output_tensor = static_cast<uint8_t *>(tensor_out->pointer());
//     auto *output_memory_buffer = static_cast<uint8_t *>(memory_buffer_->pointer());

//     cudaMemcpy(tensor_out->pointer(), dst, bytes, cudaMemcpyDeviceToDevice);


//     op_output.emit(out_entity, "out");
// }


// // Create output entity and tensor — IMPORTANT: shape is {W, H, C}
//     // auto out_entity = holoscan::gxf::Entity::New(&context);
//     // auto dl = make_dl_tensor_from_cuda(dst, H, W, C);
//     // auto out_tensor = std::make_shared<holoscan::Tensor>(dl);
//     // out_entity.add(out_tensor, "tensor");
//     // op_output.emit(out_entity, "out");

// // void RotationOperator::compute(InputContext& op_input,
// //                                OutputContext& op_output,
// //                                ExecutionContext& context) {
// //     auto in_entity = op_input.receive<holoscan::gxf::Entity>("in");
// //     if (!in_entity) return;

// //     auto in_tensor_opt = in_entity->get<Tensor>();
// //     if (!in_tensor_opt) in_tensor_opt = in_entity->get<Tensor>("video");
// //     if (!in_tensor_opt) {
// //         HOLOSCAN_LOG_ERROR("No tensor in input");
// //         return;
// //     }

// //     auto in_tensor = in_tensor_opt;

// //     auto shape = in_tensor->shape();
// //     int H = static_cast<int>(shape[0]);
// //     int W = static_cast<int>(shape[1]);
// //     int C = (shape.size() > 2) ? static_cast<int>(shape[2]) : 1;

// //     // Source buffer
// //     uint8_t* src = static_cast<uint8_t*>(in_tensor->data());

// //     size_t bytes = static_cast<size_t>(H) * W * C;

// //     // allocate dst (same number of bytes)
// //     uint8_t* dst {nullptr};
// //     cudaMalloc(&dst, bytes);
// //     cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
    
// //     dim3 block(16, 16);
// //     dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
// //     RotationType rotation_type = RotationType::ROTATE_180;
// //     launch_rotation_kernel(rotation_type, src, dst, W, H, C, grid, block);
// //     cudaDeviceSynchronize();

// //     // Create output entity and tensor — IMPORTANT: shape is {W, H, C}
// //     auto out_entity = holoscan::gxf::Entity::New(&context);
// //     auto dl = make_dl_tensor_from_cuda(dst, H, W, C);
// //     auto out_tensor = std::make_shared<holoscan::Tensor>(dl);
// //     out_entity.add(out_tensor, "tensor");
// //     op_output.emit(out_entity, "out");
// // }


// void RotationOperator::stop() {
//     memory_buffer_.reset();
// }

#include "rotation_operator.hpp"
#include "cuda_rotate_image.cuh"

#include <cuda_runtime.h>
#include <holoscan/holoscan.hpp>
#include <holoscan/core/gxf/entity.hpp> 
#include <gxf/std/tensor.hpp>           

using namespace holoscan;

void launch_rotation_kernel(RotationType type, const uint8_t* src, uint8_t* dst, int W, int H, int C, dim3 grid, dim3 block) {
    switch (type) {
        case RotationType::ROTATE_90:   
            CudaRotate::rotate_90_kernel<<<grid, block>>>(src, dst, W, H, C);
            break;
        case RotationType::ROTATE_180:
            CudaRotate::rotate_180_kernel<<<grid, block>>>(dst, W, H, C);
            break;
        case RotationType::ROTATE_270:
            CudaRotate::rotate_270_kernel<<<grid, block>>>(src, dst, W, H, C);
            break;
    }
}

void RotationOperator::setup(OperatorSpec& spec) {
    spec.input<holoscan::gxf::Entity>("in");
    spec.output<holoscan::gxf::Entity>("out");
    spec.param(allocator_, "allocator", "device_allocator", "Device Allocator");
}

void RotationOperator::initialize() {   
    // No need to create a manual MemoryBuffer here anymore.
    holoscan::Operator::initialize();
}

void RotationOperator::compute(InputContext& op_input,
                               OutputContext& op_output,
                               ExecutionContext& context) {
    // 1. Receive Input
    auto in_entity = op_input.receive<holoscan::gxf::Entity>("in");
    if (!in_entity) return;

    auto in_tensor_opt = in_entity->get<Tensor>();
    if (!in_tensor_opt) in_tensor_opt = in_entity->get<Tensor>("video");
    if (!in_tensor_opt) {
        HOLOSCAN_LOG_ERROR("No tensor in input");
        return;
    }

    // Get input dimensions
    auto shape = in_tensor_opt->shape();
    int H = static_cast<int>(shape[0]);
    int W = static_cast<int>(shape[1]);
    int C = (shape.size() > 2) ? static_cast<int>(shape[2]) : 1;
    
    // Get source pointer
    const uint8_t* src = static_cast<const uint8_t*>(in_tensor_opt->data());

    // 2. Prepare Output Entity (The Standard GXF Way)
    // Create a new GXF entity to hold the output
    auto out_message = nvidia::gxf::Entity::New(context.context());
    if (!out_message) {
        throw std::runtime_error("Failed to allocate output entity");
    }

    // Add a new Tensor component to this entity
    auto out_tensor = out_message.value().add<nvidia::gxf::Tensor>("tensor");
    if (!out_tensor) {
        throw std::runtime_error("Failed to allocate output tensor component");
    }

    // 3. Allocate Memory using the Allocator
    // Get the allocator handle from your spec parameter
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), allocator_->gxf_cid());
    
    // Define the output shape (H, W, C)
    nvidia::gxf::Shape output_shape{H, W, C};
    
    // Reshape the tensor. This triggers the memory allocation on the GPU.
    out_tensor.value()->reshape<uint8_t>(
        output_shape, 
        nvidia::gxf::MemoryStorageType::kDevice, 
        allocator.value());

    if (!out_tensor.value()->pointer()) {
        throw std::runtime_error("Failed to allocate output tensor buffer");
    }

    // Get the destination pointer from the newly allocated tensor
    uint8_t* dst = out_tensor.value()->data<uint8_t>().value();

    // 4. Perform the Operation
    // Copy src -> dst first (required if your rotate_180 is in-place)
    size_t bytes = static_cast<size_t>(H) * W * C;
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    RotationType rotation_type = RotationType::ROTATE_180;

    launch_rotation_kernel(rotation_type, src, dst, W, H, C, grid, block);

    // 5. Emit
    // Wrap the GXF entity in a Holoscan entity and emit
    auto result = holoscan::gxf::Entity(std::move(out_message.value()));
    op_output.emit(result, "out");
}

void RotationOperator::stop() {
    // No manual cleanup needed; the Allocator handles the memory lifecycle.
}
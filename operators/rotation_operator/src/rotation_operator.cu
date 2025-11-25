#include "rotation_operator.hpp"
#include "cuda_rotate_image.cuh"

#include <functional>
#include <cuda_runtime.h>
#include <holoscan/holoscan.hpp>
#include <iostream>
#include <3rdparty/dlpack/dlpack.h>

using namespace holoscan;


// --- Helper to wrap CUDA memory into a Holoscan tensor ---
DLManagedTensor* make_dl_tensor_from_cuda(uint8_t* data, int H, int W, int C) {
    auto* managed = new DLManagedTensor();
    managed->manager_ctx = nullptr;

    managed->deleter = [](DLManagedTensor* self) {
        delete self;
    };

    managed->dl_tensor.data = data;
    managed->dl_tensor.device = {kDLCUDA, 0};
    managed->dl_tensor.ndim = 3;

    static int64_t shape[3];
    shape[0] = H; shape[1] = W; shape[2] = C;
    managed->dl_tensor.shape = shape;

    managed->dl_tensor.dtype = {kDLUInt, 8, 1}; 
    managed->dl_tensor.strides = nullptr;
    managed->dl_tensor.byte_offset = 0;

    return managed;
}


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
    cudaDeviceSynchronize();
}


void RotationOperator::setup(OperatorSpec& spec) {
    spec.input<holoscan::gxf::Entity>("in");
    spec.output<holoscan::gxf::Entity>("out");
}

void RotationOperator::compute(InputContext& op_input,
                               OutputContext& op_output,
                               ExecutionContext& context) {
    auto in_entity = op_input.receive<holoscan::gxf::Entity>("in");
    if (!in_entity) return;

    auto in_tensor_opt = in_entity->get<Tensor>();
    if (!in_tensor_opt) in_tensor_opt = in_entity->get<Tensor>("video");
    if (!in_tensor_opt) {
        HOLOSCAN_LOG_ERROR("No tensor in input");
        return;
    }

    auto in_tensor = in_tensor_opt;

    auto shape = in_tensor->shape();
    int H = static_cast<int>(shape[0]);
    int W = static_cast<int>(shape[1]);
    int C = (shape.size() > 2) ? static_cast<int>(shape[2]) : 1;

    // Source buffer
    uint8_t* src = static_cast<uint8_t*>(in_tensor->data());


    size_t bytes = static_cast<size_t>(H) * W * C;

    // allocate dst (same number of bytes)
    uint8_t* dst {nullptr};
    cudaMalloc(&dst, bytes);
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    CudaRotate::rotate_180_kernel<<<grid, block>>>(dst, W, H, C);
    cudaDeviceSynchronize();

    // Create output entity and tensor — IMPORTANT: shape is {W, H, C}
    auto out_entity = holoscan::gxf::Entity::New(&context);
    auto dl = make_dl_tensor_from_cuda(dst, H, W, C);
    auto out_tensor = std::make_shared<holoscan::Tensor>(dl);
    out_entity.add(out_tensor, "tensor");
    op_output.emit(out_entity, "out");
}

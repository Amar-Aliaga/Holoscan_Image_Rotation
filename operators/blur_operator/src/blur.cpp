#include "blur.hpp"
#include "cuda_kernel.cuh"

#include <cuda_runtime.h>
#include <holoscan/holoscan.hpp>
#include <iostream>
#include <3rdparty/dlpack/dlpack.h>
#include <cstdint>

#include <opencv2/opencv.hpp>


void Blur::setup(holoscan::OperatorSpec &spec) {
    spec.input<holoscan::gxf::Entity>("in");
    spec.output<holoscan::gxf::Entity>("out");
    spec.param(allocator_, "allocator", "device_allocator", "Allocate Resources");
}


void Blur::compute(holoscan::InputContext&  op_input,
                             holoscan::OutputContext& op_output,
                             holoscan::ExecutionContext& context)
{
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("in");
    if (!maybe_entity) return;
    
    cudaStream_t stream = op_input.receive_cuda_stream("in");

    auto in_tensor = maybe_entity->get<holoscan::Tensor>();
    if (!in_tensor) {
        HOLOSCAN_LOG_ERROR("Input Entity does not contain a Tensor");
        return;
    }

    auto shape = in_tensor->shape();
    int H = shape[0];
    int W = shape[1];
    int C = shape[2];

    const uint8_t* src = static_cast<uint8_t*>(in_tensor->data());

    // 2. Create Output
    auto out_message = nvidia::gxf::Entity::New(context.context());
    auto out_tensor = out_message.value().add<nvidia::gxf::Tensor>("tensor");
    
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), allocator_->gxf_cid());

    out_tensor.value()->reshape<uint8_t>(
        nvidia::gxf::Shape({H, W, 1}),
        nvidia::gxf::MemoryStorageType::kDevice,
        allocator.value());
    
    uint8_t* dst = out_tensor.value()->data<uint8_t>().value();

    // 3. Processing (Async)
    // Only copy if you need the background preserved
    size_t bytes = static_cast<size_t>(H) * W * 1;
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream); 

    dim3 block(32, 8);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    //launch_rgba_to_laplacian(src, dst, H, W, W*4, W*1, stream, grid, block);
    blur_kernel(src, dst, W, H);
    launch_blur_kernel(src, dst, W, H, stream, grid, block);

    // 4. Emit with Stream
    // Attach the stream to the output message so downstream ops sync correctly
    op_output.set_cuda_stream(stream, "out");
    
    auto result = holoscan::gxf::Entity(std::move(out_message.value()));
    op_output.emit(result, "out");
}

#include "edge_detection.hpp"
#include "cuda_kernel.cuh"

#include <cuda_runtime.h>
#include <holoscan/holoscan.hpp>
#include <iostream>
#include <3rdparty/dlpack/dlpack.h>
#include <cstdint>

#include <opencv2/opencv.hpp>


void EdgeDetection::setup(holoscan::OperatorSpec &spec) {
    spec.input<holoscan::gxf::Entity>("in");
    spec.output<holoscan::gxf::Entity>("out");
    spec.param(allocator_, "allocator", "device_allocator", "Allocate Resources");
    //spec.param(threshold_, "threshold", "Binary Mask", "Binary Mask to display", double(128.0));
}


void EdgeDetection::compute(holoscan::InputContext&  op_input,
                             holoscan::OutputContext& op_output,
                             holoscan::ExecutionContext& context)
{
    // 1. Receive Input & Stream
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("in");
    if (!maybe_entity) return;
    
    // Get the CUDA stream carrying the data
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

    launch_rgba_to_laplacian(src, dst, H, W, W*4, W*1, stream, grid, block);

    // 4. Emit with Stream
    // Attach the stream to the output message so downstream ops sync correctly
    op_output.set_cuda_stream(stream, "out");
    
    auto result = holoscan::gxf::Entity(std::move(out_message.value()));
    op_output.emit(result, "out");
}

// #include "edge_detection.hpp"
// #include "cuda_kernel.cuh"

// #include <npp.h>             
// #include <nppcore.h>     
// #include <nppi.h>  

// #include <cuda_runtime.h>
// #include <holoscan/holoscan.hpp>
// #include <iostream>
// #include <3rdparty/dlpack/dlpack.h>
// #include <cstdint>

// #include <opencv2/opencv.hpp>
// // #include <opencv2/cudaarithm.hpp>
// // #include <opencv2/cudaimgproc.hpp>
// // #include <opencv2/cudafeatures2d.hpp>


// void npp_rgba_to_edges(const Npp8u* srcRGBA,
//                        int width, int height, int srcPitch,
//                        Npp8u* tmpGray, int grayPitch,
//                        Npp8u* dstEdge, int edgePitch) {

//     NppiSize size = {width, height};

//     NppStatus status = nppiRGBToGray_8u_AC4C1R(
//         srcRGBA, srcPitch,  // input
//         tmpGray, grayPitch, // output
//         size
//     );
//     if (status != NPP_SUCCESS) {
//         printf("nppiRGBToGray_8u_AC4C1R failed: %d\n", status);
//         return;
//     }

//     status = nppiFilterLaplace_8u_C1R(
//         tmpGray, grayPitch, 
//         dstEdge, edgePitch, 
//         size,               
//         NPP_MASK_SIZE_3_X_3 
//     );
//     if (status != NPP_SUCCESS) {
//         printf("nppiFilterLaplace_8u_C1R failed: %d\n", status);
//         return;
//     }
// }


// void EdgeDetection::setup(holoscan::OperatorSpec &spec) {
//     spec.input<holoscan::gxf::Entity>("in");
//     spec.output<holoscan::gxf::Entity>("out");
//     spec.param(threshold_, "threshold", "Binary Mask", "Binary Mask to display", double(128.0));
// }

// void EdgeDetection::start() {
//     cudaMalloc(&tmpGray, H * W * 1); 
    
//     cudaMalloc(&dstEdge, H * W * 1);

//     size_t bytes = H * W * 1; 
//     cudaMalloc(&dst, bytes);
// }

// void EdgeDetection::compute(holoscan::InputContext&  op_input,
//                              holoscan::OutputContext& op_output,
//                              holoscan::ExecutionContext& context)
// {
//     auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("in");
//     if (!maybe_entity) return;
//     auto in_tensor = maybe_entity->get<holoscan::Tensor>();
//     if (!in_tensor) {
//         HOLOSCAN_LOG_ERROR("Need CUDA RGBA tensor");
//         return;
//     }

//     auto shape = in_tensor->shape();
//     H = shape[0];
//     W = shape[1];
//     C = shape[2]; 

//     uint8_t* src = static_cast<uint8_t*>(in_tensor->data());

//     size_t bytes = H * W * 1; 

//     cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);

//     dim3 block(32, 8);
//     dim3 grid((W + block.x - 1) / block.x,
//               (H + block.y - 1) / block.y);

//     int rgba_pitch = W * C; 
//     int grey_pitch = W * 1; 
//     int srcPitch = W * C; 
//     int edgePitch = W * 1;

    

//     //npp_rgba_to_edges(dst, W, H, rgba_pitch, tmpGray, grey_pitch, dstEdge, edgePitch);
//     cudaDeviceSynchronize();

//     launch_rgba_to_laplacian(src, dst, H, W, rgba_pitch, grey_pitch, grid, block);

//     auto out_entity = holoscan::gxf::Entity::New(&context);

//     auto out_dl = make_dl_tensor_from_cuda(dstEdge, H, W, 1);
//     auto out_tensor = std::make_shared<holoscan::Tensor>(out_dl);
//     cudaDeviceSynchronize();

// //    auto out_tensor = std::make_shared<holoscan::Tensor>(
// //     dst,   // GPU pointer
// //     H,     // height
// //     W,     // width
// //     1,     // channels
// //     holoscan::ArgElementType::kInt8,  // dtype
// //     /* device = */ 0                 // GPU device ID
// // );
    
//     //auto out_tensor = std::make_shared<holoscan::Tensor>(out_dl);
//     out_entity.add(out_tensor, "tensor");   

//     cudaDeviceSynchronize();
//     op_output.emit(out_entity, "out");

//     //cudaFree(tmpGray);
//     //cudaFree(dst);
//    // cudaFree(dstEdge);
    
// }
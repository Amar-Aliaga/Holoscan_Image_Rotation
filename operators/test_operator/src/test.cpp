#include "test.hpp"
#include <holoscan/holoscan.hpp>
#include <string>


void Test::setup(holoscan::OperatorSpec &spec) {
    spec.param(string_param_, "string_param", "Message", "String to display", std::string("Default message"));
}


void Test::compute(holoscan::InputContext &op_input, holoscan::OutputContext &op_output, holoscan::ExecutionContext &context) {
    std::cout << "Message from YAML: " 
              << string_param_.get() 
              << std::endl;
}


std::string Test::get_message() {
    return string_param_.get();
}



// #pragma once
// #include <holoscan/holoscan.hpp>
// #include <cuda_runtime.h>
// #include <memory>
// #include <string>
// namespace ks::ops
// {
//   class ImageLoaderOp : public holoscan::Operator
//   {
//   public:
//     HOLOSCAN_OPERATOR_FORWARD_ARGS(ImageLoaderOp)
//     ImageLoaderOp() = default;
//     ~ImageLoaderOp() override;
//     void setup(holoscan::OperatorSpec &spec) override;
//     void stop() override;
//     void initialize() override;
//     void compute(holoscan::InputContext &,
//                  holoscan::OutputContext &,
//                  holoscan::ExecutionContext &ctx) override;
//   private:
//     holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
//     holoscan::Parameter<std::shared_ptr<holoscan::CudaStreamPool>> cuda_stream_pool_;
//     std::unique_ptr<nvidia::gxf::MemoryBuffer> raw_dev_;
//     std::unique_ptr<nvidia::gxf::MemoryBuffer> fp16_dev_;
//     std::unique_ptr<nvidia::gxf::MemoryBuffer> fp16_corr_dev_;
//     uint8_t *raw_host_pinned = nullptr;
//     holoscan::Parameter<std::string> file_path_str_;
//     holoscan::Parameter<float> black_level_offset_;
//   };
// } // namespace ks::ops
// 5:34
// #include "image_loader_op.hpp"
// #include <cuda_fp16.h>
// #include <fstream>
// #include <vector>
// #include "gxf/std/tensor.hpp"
// #include "holoscan/core/gxf/entity.hpp"
// #include <nvtx3/nvtx3.hpp>
// #include <nvtx3/nvToolsExt.h>
// namespace nvx
// {
//     inline nvtx3::color make_color(uint32_t argb) { return nvtx3::color{argb}; }
//     struct Scope
//     {
//         nvtx3::scoped_range r;
//         Scope(const char *name, uint32_t argb)
//             : r(nvtx3::message{name}, make_color(argb)) {}
//     };
//     static constexpr uint32_t C_TOTAL = 0xFF3A86FF;
//     static constexpr uint32_t C_IO = 0xFFFFBE0B;
//     static constexpr uint32_t C_ALLOC = 0xFF8338EC;
//     static constexpr uint32_t C_COPY = 0xFFFF006E;
//     static constexpr uint32_t C_PREP = 0xFF2EC4B6;
//     static constexpr uint32_t C_BLC = 0xFF06D6A0;
//     static constexpr uint32_t C_REPACK = 0xFFEF476F;
//     static constexpr uint32_t C_EMIT = 0xFF118AB2;
// } // namespace nvx
// extern "C" void prepare_data_for_BLC(const uint8_t *raw_data, __half2 *fp16_data,
//                                      int num_pixels, cudaStream_t stream);
// extern "C" void black_level_correction(const __half2 *src, __half2 *dst,
//                                        int width_half2, int height,
//                                        float black_level, cudaStream_t stream);
// extern "C" void repack_to_8bit(const __half2 *corrected_fp16_data, uint8_t *bayer_output,
//                                int num_pixels, cudaStream_t stream);
// namespace {
//     // Image geometry
//     constexpr auto kImageWidth{3840};
//     constexpr auto kImageHeight{2160};
//     constexpr auto kNumPixels{kImageWidth * kImageHeight};
//     constexpr auto kRawByteSize{(kNumPixels * 5) / 4};
//     constexpr auto kHalf2BufferSize{((kNumPixels + 1) / 2) * sizeof(__half2)};
// }
// namespace ks::ops
// {
//     void ImageLoaderOp::setup(holoscan::OperatorSpec &spec)
//     {
//         spec.output<holoscan::gxf::Entity>("source");
//         spec.param(allocator_, "allocator", "Allocator",
//                    "Device BlockMemoryPool for tensors.");
//         spec.param(cuda_stream_pool_, "cuda_stream_pool", "CudaStreamPool",
//                    "Optional CUDA stream pool (not required).");
//         spec.param(file_path_str_, "file_path", "RawFilePath",
//                    "Packed Bayer10 file path.", std::string{});
//         spec.param(black_level_offset_, "black_level_offset", "BlackLevelOffset",
//                    "Black level in [0,1].", 1.0f);
//     }
//     void ImageLoaderOp::initialize()
//     {
//         raw_dev_ = std::make_unique<nvidia::gxf::MemoryBuffer>();
//         fp16_dev_ = std::make_unique<nvidia::gxf::MemoryBuffer>();
//         fp16_corr_dev_ = std::make_unique<nvidia::gxf::MemoryBuffer>();
//         cudaMallocHost(&raw_host_pinned, kRawByteSize); // pinned, page-locked
//         holoscan::Operator::initialize();
//     }
//     ImageLoaderOp::~ImageLoaderOp() = default;
//     void ImageLoaderOp::compute(holoscan::InputContext &,
//                                 holoscan::OutputContext &op_output,
//                                 holoscan::ExecutionContext &context)
//     {
//         nvx::Scope total("ImageLoaderOp::compute (TOTAL)", nvx::C_TOTAL);
//         auto unpack_stream_expected = context.allocate_cuda_stream("unpack_blc");
//         if (!unpack_stream_expected.has_value())
//         {
//             throw std::runtime_error("Failed to allocate stream 'unpack_blc': " +
//                                      std::string(unpack_stream_expected.error().what()));
//         }
//         cudaStream_t unpack_stream = unpack_stream_expected.value();
//         {
//             nvx::Scope s("Read raw file (host)", nvx::C_IO);
//             std::ifstream raw_file(file_path_str_.get(), std::ios::binary);
//             if (!raw_file)
//                 throw std::runtime_error("Failed to open raw file: " + file_path_str_.get());
//             raw_file.read(reinterpret_cast<char *>(raw_host_pinned), kRawByteSize);
//             if (!raw_file)
//                 throw std::runtime_error("Unexpected end of raw file read.");
//             {
//                 nvx::Scope s2("Create entity & tensor/allocs", nvx::C_ALLOC);
//                 auto output_message = holoscan::gxf::Entity::New(&context);
//                 auto &gxf_entity = static_cast<nvidia::gxf::Entity &>(output_message);
//                 auto tensor_output_bayer8 = gxf_entity.add<nvidia::gxf::Tensor>("disk_frames").value();
//                 auto gxf_allocator =
//                     nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
//                         context.context(), allocator_.get()->gxf_cid())
//                         .value();
//                 if (raw_dev_->size() < kRawByteSize)
//                     raw_dev_->resize(gxf_allocator, kRawByteSize, nvidia::gxf::MemoryStorageType::kDevice);
//                 if (fp16_dev_->size() < kHalf2BufferSize)
//                     fp16_dev_->resize(gxf_allocator, kHalf2BufferSize, nvidia::gxf::MemoryStorageType::kDevice);
//                 if (fp16_corr_dev_->size() < kHalf2BufferSize)
//                     fp16_corr_dev_->resize(gxf_allocator, kHalf2BufferSize, nvidia::gxf::MemoryStorageType::kDevice);
//                 tensor_output_bayer8->reshape<uint8_t>(
//                     nvidia::gxf::Shape({kImageHeight, kImageWidth, 1}),
//                     nvidia::gxf::MemoryStorageType::kDevice, gxf_allocator);
//                 auto *d_output_bayer8 = static_cast<uint8_t *>(tensor_output_bayer8->pointer());
//                 auto *d_raw_input = static_cast<uint8_t *>(raw_dev_->pointer());
//                 auto *d_fp16_intermediate = reinterpret_cast<__half2 *>(fp16_dev_->pointer());
//                 auto *d_fp16_corrected = reinterpret_cast<__half2 *>(fp16_corr_dev_->pointer());
//                 {
//                     nvx::Scope s3("HtoD raw copy (async)", nvx::C_COPY);
//                     cudaMemcpyAsync(d_raw_input, raw_host_pinned, kRawByteSize,
//                                     cudaMemcpyHostToDevice, unpack_stream);
//                 }
//                 {
//                     nvx::Scope s3("prepare_data_for_BLC (kernel)", nvx::C_PREP);
//                     prepare_data_for_BLC(d_raw_input, d_fp16_intermediate, kNumPixels, unpack_stream);
//                 }
//                 {
//                     nvx::Scope s3("black_level_correction (kernel)", nvx::C_BLC);
//                     black_level_correction(d_fp16_intermediate, d_fp16_corrected,
//                                            kImageWidth / 2, kImageHeight,
//                                            black_level_offset_, unpack_stream);
//                 }
//                 {
//                     nvx::Scope s3("repack_to_8bit (kernel)", nvx::C_REPACK);
//                     repack_to_8bit(d_fp16_corrected, d_output_bayer8, kNumPixels, unpack_stream);
//                 }
//                 {
//                     nvx::Scope s3("op_output.set_cuda_stream", 0xFF90CAF9);
//                     op_output.set_cuda_stream(unpack_stream, "source");
//                 }
//                 {
//                     nvx::Scope s3("op_output.emit", nvx::C_EMIT);
//                     op_output.emit(output_message, "source");
//                 }
//             }
//         }
//     }
//     void ImageLoaderOp::stop()
//     {
//         raw_dev_.reset();
//         fp16_dev_.reset();
//         fp16_corr_dev_.reset();
//     }
// } // namespace ks::ops
// 5:34
// #include <cuda_fp16.h>
// #include <cuda_runtime.h>
// #include <cstdint>
// // One thread handles 4 pixels (exactly one 5-byte packed group).
// // It writes 2 half2 words: [p0,p1] and (if present) [p2,p3].
// static __global__ void unpackAndNormalizeToFP16(const uint8_t *__restrict__ rawData,
//                                                 __half2 *__restrict__ fp16Data,
//                                                 int numPixels)
// {
//     const int group = blockIdx.x * blockDim.x + threadIdx.x; // group of 4 pixels
//     const int basePixel = group * 4;
//     if (basePixel >= numPixels)
//         return;
//     const size_t byteIdx = static_cast<size_t>(group) * 5;
//     // Decode 4×10-bit from 5 bytes (little-endian packing)
//     const uint16_t p0 = (rawData[byteIdx] | ((rawData[byteIdx + 1] & 0x03) << 8));
//     const uint16_t p1 = (((rawData[byteIdx + 1] & 0xFC) >> 2) |
//                          ((rawData[byteIdx + 2] & 0x0F) << 6));
//     const uint16_t p2 = (((rawData[byteIdx + 2] & 0xF0) >> 4) |
//                          ((rawData[byteIdx + 3] & 0x3F) << 4));
//     const uint16_t p3 = (((rawData[byteIdx + 3] & 0xC0) >> 6) |
//                          (rawData[byteIdx + 4] << 2));
//     const float s = 1.0f / 1023.0f;
//     const __half2 h01 = __floats2half2_rn(p0 * s, p1 * s);
//     fp16Data[2 * group] = h01;
//     if (basePixel + 2 < numPixels)
//     {
//         const __half2 h23 = __floats2half2_rn(p2 * s, p3 * s);
//         fp16Data[2 * group + 1] = h23;
//     }
// }
// extern "C" void prepare_data_for_BLC(const uint8_t *rawData,
//                                      __half2 *fp16Data,
//                                      int numPixels,
//                                      cudaStream_t stream)
// {
//     const int threadsPerBlock = 256;
//     const int groups = (numPixels + 3) / 4; // 4 pixels per thread
//     const int blocks = (groups + threadsPerBlock - 1) / threadsPerBlock;
//     unpackAndNormalizeToFP16<<<blocks, threadsPerBlock, 0, stream>>>(
//         rawData, fp16Data, numPixels);
//     // no device-wide sync here
// }
// 5:34
// #include <cuda_fp16.h>
// #include <cuda_runtime.h>
// #include <cstdint>
// // Repack normalized FP16 [0..1] (two pixels per half2) to uint8_t.
// // One thread processes 4 pixels -> reads 2 half2 words and writes 4 bytes.
// static __global__ void repackTo8BitKernel(const __half2 *__restrict__ src,
//                                           uint8_t *__restrict__ dst,
//                                           int numPixels)
// {
//     const int group = blockIdx.x * blockDim.x + threadIdx.x; // 4 pixels
//     const int basePixel = group * 4;
//     if (basePixel >= numPixels)
//         return;
//     const int hIdx = 2 * group;
//     const __half2 h01 = src[hIdx];
//     float p0 = __low2float(h01) * 255.0f;
//     float p1 = __high2float(h01) * 255.0f;
//     dst[basePixel] = static_cast<uint8_t>(p0);
//     if (basePixel + 1 < numPixels)
//     {
//         dst[basePixel + 1] = static_cast<uint8_t>(p1);
//     }
//     if (basePixel + 2 < numPixels)
//     {
//         const __half2 h23 = src[hIdx + 1];
//         float p2 = __low2float(h23) * 255.0f;
//         float p3 = __high2float(h23) * 255.0f;
//         dst[basePixel + 2] = static_cast<uint8_t>(p2);
//         if (basePixel + 3 < numPixels)
//         {
//             dst[basePixel + 3] = static_cast<uint8_t>(p3);
//         }
//     }
// }
// extern "C" void repack_to_8bit(const __half2 *correctedFP16Data,
//                                uint8_t *bayerData,
//                                int numPixels,
//                                cudaStream_t stream)
// {
//     const int threadsPerBlock = 256;
//     const int groups = (numPixels + 3) / 4; // 4 pixels per thread
//     const int blocks = (groups + threadsPerBlock - 1) / threadsPerBlock;
//     repackTo8BitKernel<<<blocks, threadsPerBlock, 0, stream>>>(
//         correctedFP16Data, bayerData, numPixels);
//     // no device-wide sync here
// }










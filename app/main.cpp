#include "rotation_operator.hpp"
#include "pingMxOp.hpp"
#include "PingTxOp.hpp"
#include "PingRxOp.hpp"
#include "test.hpp"
#include "edge_detection.hpp"

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/v4l2_video_capture/v4l2_video_capture.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/core/resources/gxf/unbounded_allocator.hpp>
#include <holoscan/core/resources/gxf/cuda_stream_pool.hpp>

#include <tl/expected.hpp>


class CameraGPUDisplayApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto host_alloc   = make_resource<UnboundedAllocator>("host_allocator");
    auto device_alloc = make_resource<UnboundedAllocator>("device_allocator");
    auto stream_pool  = make_resource<CudaStreamPool>("cuda_stream_pool", Arg("dev_id") = 0);

    auto camera = make_operator<ops::V4L2VideoCaptureOp>("camera",
        from_config("camera"),
        Arg("allocator") = host_alloc);

    auto format_converter = make_operator<ops::FormatConverterOp>("format_converter",
        from_config("format_converter"),
        Arg("pool") = device_alloc,
        Arg("cuda_stream_pool") = stream_pool);

    auto viz = make_operator<ops::HolovizOp>("holoviz",
        from_config("holoviz"),
        Arg("allocator") = device_alloc,
        Arg("cuda_stream_pool") = stream_pool);

    auto viz2 = make_operator<ops::HolovizOp>("holoviz2",
        from_config("holoviz2"),
        Arg("allocator") = device_alloc,
        Arg("cuda_stream_pool") = stream_pool);    

    auto multithread_sched = make_resource<MultiThreadScheduler>("m_scheduler",
        from_config("m_scheduler"));
    
    this->scheduler(multithread_sched);

    auto rotate_op = make_operator<RotationOperator>("rotate",
        Arg("allocator") = device_alloc);

    auto edge = make_operator<EdgeDetection>("edge", from_config("binary_mask_value"));

    add_flow(camera, format_converter,    {{"signal", "source_video"}});
    add_flow(format_converter, viz,       {{"tensor", "receivers"}});
    add_flow(format_converter, edge, {{"tensor", "in"}});
    add_flow(edge, viz2,             {{"out",    "receivers"}});
  }
};



class Ping : public holoscan::Application {
    public:
        void compose() override {
            auto pingTx = make_operator<holoscan::ops::PingTxOp>("pingTx", make_condition<holoscan::CountCondition>(10));
            auto pingMx = make_operator<PingMxOp>("pingMx", holoscan::Arg("multiplier", 3));
            auto pingRx = make_operator<holoscan::ops::PingRxOp>("pingRx");

            add_flow(pingTx, pingMx, {{"out", "in"}});
            add_flow(pingMx, pingRx, {{"out", "in"}});
        }
};


class MyOp1 : public holoscan::Application {
    public:
        void compose() override {
            auto c1 = make_condition<holoscan::CountCondition>("my_count_condition", 10);
            auto c2 = make_condition<holoscan::PeriodicCondition>("my_periodic_condition", 2000);
            auto test = make_operator<Test>("test", from_config("simple_string_param"), c1, c2);
            add_operator(test);
        }
};


class Edge : public holoscan::Application {
    public:
        void compose() override {
            auto edge = make_operator<EdgeDetection>("edge", from_config("binary_mask_value"));
            add_operator(edge);
        }
};


int main(int argc, char** argv) {
  auto app = holoscan::make_application<CameraGPUDisplayApp>();
  app->config("config/app_config.yaml");  
  app->run();

//SECOND OPERATOR
    // auto app = holoscan::make_application<Ping>();
    // app->run();


//THIRD OPERATOR
    // auto app = holoscan::make_application<MyOp1>();
    // app->config("operators/test_operator/config.yaml");
    // app->run();

//FOURTH OPERATOR
    // auto app = holoscan::make_application<Edge>();
    // app->config("operators/edge_detection_operator/config.yaml");
    // app->run();

  return 0;
}
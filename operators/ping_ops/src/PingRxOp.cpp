#include "PingRxOp.hpp"
#include <holoscan/holoscan.hpp>
#include <iostream>

namespace holoscan::ops {

    void PingRxOp::setup(OperatorSpec& spec) {
        spec.input<int>("in");
    }

    void PingRxOp::compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
                        [[maybe_unused]] ExecutionContext& context) {
    auto maybe_value = op_input.receive<int>("in");
    if (!maybe_value) {
        auto error_msg = fmt::format("Operator '{}' failed to receive message from port 'in': {}",
                                    name_,
                                    maybe_value.error().what());
        HOLOSCAN_LOG_ERROR(error_msg);
        throw std::runtime_error(error_msg);
    }
    int value = maybe_value.value();
    HOLOSCAN_LOG_INFO("Rx message value: {}", value);
    }

} 

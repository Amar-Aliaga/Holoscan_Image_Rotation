#include "PingTxOp.hpp"
#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

    void PingTxOp::setup(OperatorSpec& spec) {
    spec.output<int>("out");
    }

    void PingTxOp::compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
                        [[maybe_unused]] ExecutionContext& context) {
    int value = index_++;
    op_output.emit(value, "out");
    }
}  

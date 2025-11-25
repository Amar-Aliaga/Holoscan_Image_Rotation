#pragma once
#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

    class PingRxOp : public Operator {
    public:
        HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxOp)

        PingRxOp() = default;

        void setup(OperatorSpec& spec) override;

        void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
                    [[maybe_unused]] ExecutionContext& context) override;
        };

} 

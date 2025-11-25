#pragma once

#include <holoscan/holoscan.hpp>


class PingMxOp : public holoscan::Operator {
    public:
        HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMxOp)
        PingMxOp() = default;

        void setup(holoscan::OperatorSpec &spec) override;
        void compute(holoscan::InputContext &op_input, holoscan::OutputContext &op_output, [[maybe_unused]]holoscan::ExecutionContext &context) override;
    private:
        holoscan::Parameter<int> multiplier_;
};
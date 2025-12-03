#pragma once

#include <holoscan/holoscan.hpp>


class EdgeDetection : public holoscan::Operator {
    public:
        HOLOSCAN_OPERATOR_FORWARD_ARGS(EdgeDetection)
        EdgeDetection() = default;

        void setup(holoscan::OperatorSpec &spec) override;
        void compute([[maybe_unused]]holoscan::InputContext &op_input, [[maybe_unused]]holoscan::OutputContext &op_output, [[maybe_unused]]holoscan::ExecutionContext &context) override;
    
    private:
        holoscan::Parameter<double> threshold_;
};
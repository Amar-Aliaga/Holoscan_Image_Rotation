#pragma once
#include <holoscan/holoscan.hpp>
#include <string>


class Test : public holoscan::Operator {
    public:
        HOLOSCAN_OPERATOR_FORWARD_ARGS(Test)
        Test() = default;
        void setup(holoscan::OperatorSpec &spec) override;
        void compute([[maybe_unused]]holoscan::InputContext &op_input, [[maybe_unused]]holoscan::OutputContext &op_output, [[maybe_unused]]holoscan::ExecutionContext &context) override;
        std::string get_message();
    private:
        holoscan::Parameter<std::string> string_param_;
};
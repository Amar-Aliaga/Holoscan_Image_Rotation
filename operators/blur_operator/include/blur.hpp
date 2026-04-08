#pragma once

#include <holoscan/holoscan.hpp>
#include <memory>


class Blur : public holoscan::Operator {
    public:
        HOLOSCAN_OPERATOR_FORWARD_ARGS(Blur)
        Blur() = default;

        [[maybe_unused]] void setup(holoscan::OperatorSpec &spec) override;
        [[maybe_unused]] void compute([[maybe_unused]]holoscan::InputContext &op_input, [[maybe_unused]]holoscan::OutputContext &op_output, [[maybe_unused]]holoscan::ExecutionContext &context) override;
        //[[maybe_unused]]void initialize() override;
    private:
       // holoscan::Parameter<double> threshold_;
        holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
        std::unique_ptr<nvidia::gxf::MemoryBuffer> mem_buffer_;
};
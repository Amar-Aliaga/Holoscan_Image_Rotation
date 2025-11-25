#include "pingMxOp.hpp"
#include <holoscan/holoscan.hpp>
#include <tl/expected.hpp>

void PingMxOp::setup(holoscan::OperatorSpec &spec) {
    spec.input<int>("in");
    spec.output<int>("out");
    spec.param(multiplier_, "multiplier", "Multiplier", "Multiply the input by this value: ", 2);
}


void PingMxOp::compute(holoscan::InputContext &op_input, holoscan::OutputContext &op_output, holoscan::ExecutionContext &context) {
    auto value = op_input.receive<int>("in");

    if (value)
    op_output.emit(*value * multiplier_.get());

}
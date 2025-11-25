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
#pragma once
#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

    class PingTxOp : public Operator {
        public:
            HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

            PingTxOp() = default;

            void setup(OperatorSpec& spec) override;

            void compute(InputContext& op_input, OutputContext& op_output,
                        ExecutionContext& context) override;

            int index() const { return index_; }

        private:
            int index_ = 1;
    };

}
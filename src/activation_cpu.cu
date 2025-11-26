#include "activation_cpu.h"

Tensor ReLUCPU::forward(const Tensor &input) {
    Tensor out(input.N, input.C, input.H, input.W);
    for (int i = 0; i < input.size(); ++i) {
        out.data[i] = input.data[i] > 0.0f ? input.data[i] : 0.0f;
    }
    return out;
}

Tensor ReLUCPU::backward(const Tensor &input, const Tensor &grad_output) {
    Tensor grad_input(input.N, input.C, input.H, input.W);
    for (int i = 0; i < input.size(); ++i) {
        grad_input.data[i] = (input.data[i] > 0.0f) ? grad_output.data[i] : 0.0f;
    }
    return grad_input;
}

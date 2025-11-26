#pragma once
#include "tensor.h"

struct ReLUCPU {
    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &input, const Tensor &grad_output);
};

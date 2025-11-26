#pragma once
#include "tensor.h"

struct MSELossCPU {
    // trả về loss scalar
    float forward(const Tensor &pred, const Tensor &target);
    // grad w.r.t pred
    Tensor backward(const Tensor &pred, const Tensor &target);
};

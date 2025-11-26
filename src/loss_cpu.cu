#include "loss_cpu.h"
#include <cmath>

float MSELossCPU::forward(const Tensor &pred, const Tensor &target) {
    int n = pred.size();
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = pred.data[i] - target.data[i];
        sum += diff * diff;
    }
    return sum / n;
}

Tensor MSELossCPU::backward(const Tensor &pred, const Tensor &target) {
    Tensor grad(pred.N, pred.C, pred.H, pred.W);
    int n = pred.size();
    for (int i = 0; i < n; ++i) {
        grad.data[i] = 2.0f * (pred.data[i] - target.data[i]) / n;
    }
    return grad;
}

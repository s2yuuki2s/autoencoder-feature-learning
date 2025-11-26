#pragma once
#include "tensor.h"
#include <vector>

struct Conv2DCPU {
    int in_channels;
    int out_channels;
    int kernel_h;
    int kernel_w;
    int stride;
    int padding;

    // weight: (out_channels, in_channels, kernel_h, kernel_w)
    Tensor weight;
    // bias: (out_channels)
    std::vector<float> bias;

    // gradients
    Tensor grad_weight;
    std::vector<float> grad_bias;

    Conv2DCPU(int in_c, int out_c, int k_h, int k_w,
              int stride_=1, int padding_=1);

    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &input, const Tensor &grad_output);

    void zero_grad();
    void sgd_update(float lr);
};

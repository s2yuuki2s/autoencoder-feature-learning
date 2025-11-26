#pragma once
#include "tensor.h"
#include <vector>

// MaxPool2D 2x2 stride 2
struct MaxPool2DCPU {
    int kernel;
    int stride;

    // Lưu vị trí max để backward (N,C,H_out,W_out)
    std::vector<int> max_indices;

    MaxPool2DCPU(int kernel_=2, int stride_=2)
        : kernel(kernel_), stride(stride_) {}

    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &input, const Tensor &grad_output);
};

// Upsample 2x2 (nearest neighbor)
struct UpSample2DCPU {
    int scale;

    UpSample2DCPU(int scale_=2) : scale(scale_) {}

    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &input, const Tensor &grad_output);
};

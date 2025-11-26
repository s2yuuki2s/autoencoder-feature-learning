#include <iostream>
#include "tensor.h"
#include "conv2d_cpu.h"
#include "pooling_cpu.h"
#include "activation_cpu.h"
#include "loss_cpu.h"

std::string data_path = "/content/proj/data/";

int main(int argc, char** argv) {
    std::string data_path = (argc > 1 ? argv[1] : "./data/");

    // ==== Test Conv2D 1 filter 3x3 trên input 1x1x3x3 ====
    Tensor x(1, 1, 3, 3);
    int v = 1;
    for (int h = 0; h < 3; ++h)
        for (int w = 0; w < 3; ++w)
            x(0, 0, h, w) = (float)v++;

    Conv2DCPU conv(1, 1, 3, 3, 1, 1);
    // set weight = 1, bias = 0 để dễ check
    for (int kh = 0; kh < 3; ++kh)
        for (int kw = 0; kw < 3; ++kw)
            conv.weight(0, 0, kh, kw) = 1.0f;
    conv.bias[0] = 0.0f;

    Tensor y = conv.forward(x);
    std::cout << "Conv2D forward output (1x1x3x3 with all-ones kernel):\n";
    for (int h = 0; h < y.H; ++h) {
        for (int w = 0; w < y.W; ++w) {
            std::cout << y(0, 0, h, w) << " ";
        }
        std::cout << "\n";
    }
    // expected: mỗi vị trí là tổng 3x3 = sum(1..9) = 45, nhưng biên dùng padding nên khác,
    // miễn là symmetric, không NaN là ok.

    // ==== Test MaxPool 2x2 trên 1x1x4x4 ====
    Tensor x2(1, 1, 4, 4);
    for (int h = 0; h < 4; ++h)
        for (int w = 0; w < 4; ++w)
            x2(0, 0, h, w) = (float)(h * 4 + w);

    MaxPool2DCPU pool;
    Tensor p = pool.forward(x2);
    std::cout << "\nMaxPool2D forward output (1x1x4x4 -> 1x1x2x2):\n";
    for (int h = 0; h < p.H; ++h) {
        for (int w = 0; w < p.W; ++w)
            std::cout << p(0, 0, h, w) << " ";
        std::cout << "\n";
    }
    // expected: [[5,7],[13,15]]

    // ==== Test UpSample 2x2 ====
    UpSample2DCPU up;
    Tensor up_out = up.forward(p);
    std::cout << "\nUpSample2D forward output shape: "
              << up_out.N << "x" << up_out.C << "x"
              << up_out.H << "x" << up_out.W << "\n";

    // ==== Test ReLU + MSE ====
    ReLUCPU relu;
    Tensor x3(1, 1, 2, 2);
    x3(0,0,0,0) = -1.0f; x3(0,0,0,1) = 2.0f;
    x3(0,0,1,0) = -3.0f; x3(0,0,1,1) = 4.0f;
    Tensor r = relu.forward(x3);
    std::cout << "\nReLU forward:\n";
    for (int h = 0; h < 2; ++h) {
        for (int w = 0; w < 2; ++w)
            std::cout << r(0,0,h,w) << " ";
        std::cout << "\n";
    }

    Tensor target(1,1,2,2);
    target.fill(1.0f);
    MSELossCPU mse;
    float loss = mse.forward(r, target);
    std::cout << "\nMSE loss (vs all ones): " << loss << "\n";

    Tensor grad = mse.backward(r, target);
    std::cout << "MSE grad first element: " << grad(0,0,0,0) << "\n";

    std::cout << "\nAll layer tests finished.\n";
    return 0;
}

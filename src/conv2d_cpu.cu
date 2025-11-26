#include "conv2d_cpu.h"
#include <cmath>
#include <cstdlib>

static float rand_uniform(float a = -0.1f, float b = 0.1f) {
    return a + (b - a) * (float)rand() / (float)RAND_MAX;
}

Conv2DCPU::Conv2DCPU(int in_c, int out_c, int k_h, int k_w,
                     int stride_, int padding_)
    : in_channels(in_c), out_channels(out_c),
      kernel_h(k_h), kernel_w(k_w),
      stride(stride_), padding(padding_),
      weight(out_c, in_c, k_h, k_w),
      grad_weight(out_c, in_c, k_h, k_w),
      bias(out_c, 0.0f),
      grad_bias(out_c, 0.0f) {

    // init He-like
    float scale = std::sqrt(2.0f / (in_c * k_h * k_w));
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    weight(oc, ic, kh, kw) = rand_uniform(-scale, scale);
                }
            }
        }
        bias[oc] = 0.0f;
    }
}

void Conv2DCPU::zero_grad() {
    std::fill(grad_weight.data.begin(), grad_weight.data.end(), 0.0f);
    std::fill(grad_bias.begin(), grad_bias.end(), 0.0f);
}

void Conv2DCPU::sgd_update(float lr) {
    int total = grad_weight.size();
    for (int i = 0; i < total; ++i) {
        weight.data[i] -= lr * grad_weight.data[i];
    }
    for (int oc = 0; oc < out_channels; ++oc) {
        bias[oc] -= lr * grad_bias[oc];
    }
}

Tensor Conv2DCPU::forward(const Tensor &input) {
    int N = input.N;
    int H_in = input.H;
    int W_in = input.W;
    int H_out = (H_in + 2 * padding - kernel_h) / stride + 1;
    int W_out = (W_in + 2 * padding - kernel_w) / stride + 1;

    Tensor output(N, out_channels, H_out, W_out);
    output.fill(0.0f);

    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float sum = bias[oc];
                    int ih0 = oh * stride - padding;
                    int iw0 = ow * stride - padding;

                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            int ih = ih0 + kh;
                            if (ih < 0 || ih >= H_in) continue;

                            for (int kw = 0; kw < kernel_w; ++kw) {
                                int iw = iw0 + kw;
                                if (iw < 0 || iw >= W_in) continue;

                                sum += input(n, ic, ih, iw) *
                                       weight(oc, ic, kh, kw);
                            }
                        }
                    }
                    output(n, oc, oh, ow) = sum;
                }
            }
        }
    }
    return output;
}

// grad_output: same shape as output
Tensor Conv2DCPU::backward(const Tensor &input, const Tensor &grad_output) {
    zero_grad();

    int N = input.N;
    int H_in = input.H;
    int W_in = input.W;

    int H_out = grad_output.H;
    int W_out = grad_output.W;

    Tensor grad_input(N, in_channels, H_in, W_in);
    grad_input.fill(0.0f);

    // grad w.r.t weight + bias
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float go = grad_output(n, oc, oh, ow);
                    grad_bias[oc] += go;

                    int ih0 = oh * stride - padding;
                    int iw0 = ow * stride - padding;

                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            int ih = ih0 + kh;
                            if (ih < 0 || ih >= H_in) continue;

                            for (int kw = 0; kw < kernel_w; ++kw) {
                                int iw = iw0 + kw;
                                if (iw < 0 || iw >= W_in) continue;

                                float x = input(n, ic, ih, iw);
                                grad_weight(oc, ic, kh, kw) += go * x;

                                // grad_input
                                grad_input(n, ic, ih, iw) +=
                                    go * weight(oc, ic, kh, kw);
                            }
                        }
                    }
                }
            }
        }
    }
    return grad_input;
}

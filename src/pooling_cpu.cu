#include "pooling_cpu.h"
#include <algorithm>

Tensor MaxPool2DCPU::forward(const Tensor &input) {
    int N = input.N, C = input.C, H = input.H, W = input.W;
    int H_out = H / stride;
    int W_out = W / stride;

    Tensor output(N, C, H_out, W_out);
    output.fill(0.0f);
    max_indices.assign(N * C * H_out * W_out, -1);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    int h0 = oh * stride;
                    int w0 = ow * stride;
                    float max_val = -1e30f;
                    int max_idx = -1;

                    for (int kh = 0; kh < kernel; ++kh) {
                        for (int kw = 0; kw < kernel; ++kw) {
                            int ih = h0 + kh;
                            int iw = w0 + kw;
                            if (ih >= H || iw >= W) continue;
                            float v = input(n, c, ih, iw);
                            int flat_idx = ((n * C + c) * H + ih) * W + iw;
                            if (v > max_val) {
                                max_val = v;
                                max_idx = flat_idx;
                            }
                        }
                    }
                    output(n, c, oh, ow) = max_val;
                    int out_flat = ((n * C + c) * H_out + oh) * W_out + ow;
                    max_indices[out_flat] = max_idx;
                }
            }
        }
    }
    return output;
}

Tensor MaxPool2DCPU::backward(const Tensor &input, const Tensor &grad_output) {
    int N = input.N, C = input.C, H = input.H, W = input.W;
    int H_out = grad_output.H;
    int W_out = grad_output.W;

    Tensor grad_input(N, C, H, W);
    grad_input.fill(0.0f);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    int out_flat = ((n * C + c) * H_out + oh) * W_out + ow;
                    int idx = max_indices[out_flat];
                    if (idx < 0) continue;
                    float go = grad_output(n, c, oh, ow);

                    int h = (idx / W) % H;
                    int w = idx % W;
                    int cc = (idx / (H * W)) % C;
                    int nn = idx / (C * H * W);

                    if (nn == n && cc == c) {
                        grad_input(n, c, h, w) += go;
                    }
                }
            }
        }
    }
    return grad_input;
}

Tensor UpSample2DCPU::forward(const Tensor &input) {
    int N = input.N, C = input.C, H = input.H, W = input.W;
    int H_out = H * scale;
    int W_out = W * scale;

    Tensor output(N, C, H_out, W_out);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H_out; ++h)
                for (int w = 0; w < W_out; ++w) {
                    int ih = h / scale;
                    int iw = w / scale;
                    output(n, c, h, w) = input(n, c, ih, iw);
                }
    return output;
}

Tensor UpSample2DCPU::backward(const Tensor &input, const Tensor &grad_output) {
    int N = input.N, C = input.C, H = input.H, W = input.W;
    int H_out = grad_output.H;
    int W_out = grad_output.W;

    Tensor grad_input(N, C, H, W);
    grad_input.fill(0.0f);

    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H_out; ++h)
                for (int w = 0; w < W_out; ++w) {
                    int ih = h / scale;
                    int iw = w / scale;
                    grad_input(n, c, ih, iw) += grad_output(n, c, h, w);
                }
    return grad_input;
}

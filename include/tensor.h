#pragma once
#include <vector>
#include <cassert>
#include <cstring>

// Tensor 4D đơn giản theo format NCHW
struct Tensor {
    int N, C, H, W;
    std::vector<float> data;

    Tensor() : N(0), C(0), H(0), W(0) {}
    Tensor(int n, int c, int h, int w)
        : N(n), C(c), H(h), W(w), data(n * c * h * w, 0.0f) {}

    inline int size() const { return N * C * H * W; }

    inline float &operator()(int n, int c, int h, int w) {
        int idx = ((n * C + c) * H + h) * W + w;
        return data[idx];
    }

    inline const float &operator()(int n, int c, int h, int w) const {
        int idx = ((n * C + c) * H + h) * W + w;
        return data[idx];
    }

    void fill(float v) {
        std::fill(data.begin(), data.end(), v);
    }
};

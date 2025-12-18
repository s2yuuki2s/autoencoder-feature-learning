#ifndef GPU_LAYERS_OPT_H
#define GPU_LAYERS_OPT_H

// =========================================================
// FORWARD LAYERS (FUSED & OPTIMIZED)
// =========================================================

// Fused Conv2D + AddBias + ReLU (Optional)
// Input: in
// Output: out = ReLU(Conv(in, W) + bias) (nếu use_relu = true)
// Output: out = Conv(in, W) + bias       (nếu use_relu = false)
void gpu_conv2d_fused_forward(const float *in, const float *W, const float *bias, float *out,
                              int n, int width, int height, int in_c, int out_c,
                              bool use_relu);

// Max Pooling 2x2
void gpu_max_pooling(const float *in, float *out, int n, int width, int height, int depth);

// Upsampling 2x
void gpu_upsampling(const float *in, float *out, int n, int width, int height, int depth);

// =========================================================
// BACKWARD LAYERS (SPLIT & OPTIMIZED)
// =========================================================

// Tính Gradient Input (dIn) sử dụng Shared Memory Tiling
// Không dùng AtomicAdd -> Nhanh hơn
void gpu_conv2d_backward_input(const float *grad_out, const float *W, float *grad_in,
                               int n, int width, int height, int in_c, int out_c);

// Tính Gradient Weights (dW) & Bias (db)
// Dùng AtomicAdd (Chấp nhận được vì size Weights nhỏ)
void gpu_conv2d_backward_params(const float *in, const float *grad_out,
                                float *grad_W, float *grad_b,
                                int n, int width, int height, int in_c, int out_c);

// Các backward khác
void gpu_relu_backward(const float *feat_out, const float *grad_out, float *grad_in, int total);
void gpu_maxpool_backward(const float *in, const float *out, const float *grad_out, float *grad_in,
                          int n, int width, int height, int depth);
void gpu_upsample_backward(const float *grad_out, float *grad_in, int n, int width, int height, int depth);

#endif
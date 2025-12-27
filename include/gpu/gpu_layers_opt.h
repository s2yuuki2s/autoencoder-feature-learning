#ifndef GPU_LAYERS_OPT_H
#define GPU_LAYERS_OPT_H

// Fused convolutional layer with optional ReLU activation
void gpu_conv2d_fused_forward(const float *in, const float *W, const float *bias, float *out,
                              int n, int width, int height, int in_c, int out_c,
                              bool use_relu);

// Optimized backward convolution to compute input gradients
void gpu_conv2d_backward_input(const float *grad_out, const float *W, float *grad_in,
                               int n, int width, int height, int in_c, int out_c);

void gpu_conv2d_backward_params(const float *in, const float *grad_out,
                                float *grad_W, float *grad_b,
                                int n, int width, int height, int in_c, int out_c);

// Helper pooling and activation functions
void gpu_max_pooling(const float *in, float *out, int n, int width, int height, int depth);

void gpu_upsampling(const float *in, float *out, int n, int width, int height, int depth);

void gpu_relu_backward(const float *feat_out, const float *grad_out, float *grad_in, int total);
void gpu_maxpool_backward(const float *in, const float *out, const float *grad_out, float *grad_in,
                          int n, int width, int height, int depth);
void gpu_upsample_backward(const float *grad_out, float *grad_in, int n, int width, int height, int depth);

#endif
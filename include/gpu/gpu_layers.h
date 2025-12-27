#ifndef GPU_LAYERS_H
#define GPU_LAYERS_H

void gpu_conv2D(const float *in, const float *filter, float *out,
                int n, int width, int height, int depth, int n_filter);

void gpu_add_bias(const float *in, const float *bias, float *out,
                  int n, int width, int height, int depth);

void gpu_relu(const float *in, float *out,
              int n, int width, int height, int depth);

void gpu_max_pooling(const float *in, float *out,
                     int n, int width, int height, int depth);

void gpu_upsampling(const float *in, float *out,
                    int n, int width, int height, int depth);

void gpu_conv2d_backward_native(const float *in, const float *grad_out, const float *W, int n, int width, int height, int in_c, int out_c, float *grad_in, float *grad_W, float *grad_b);
void gpu_maxpool_backward_native(const float *in, const float *out, const float *grad_out, int n, int width, int height, int depth, float *grad_in);
void gpu_relu_backward_native(const float *in, const float *grad_out, float *grad_in, int total);
void gpu_upsample_backward_native(const float *grad_out, int n, int width, int height, int depth, float *grad_in);
#endif

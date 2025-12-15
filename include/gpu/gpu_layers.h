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

#endif

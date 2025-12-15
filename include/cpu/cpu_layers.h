#ifndef CPU_LAYERS_H
#define CPU_LAYERS_H

void cpu_conv2D(float *in, float *filter, float *out,
                int n, int width, int height, int depth, int n_filter);

void cpu_add_bias(float *in, float *bias, float *out,
                  int n, int width, int height, int depth);

void cpu_relu(float *in, float *out,
              int n, int width, int height, int depth);

void cpu_max_pooling(float *in, float *out,
                     int n, int width, int height, int depth);

void cpu_upsampling(float *in, float *out,
                    int n, int width, int height, int depth);

float cpu_mse_loss(float *expected, float *actual,
                   int n, int width, int height, int depth);

#endif
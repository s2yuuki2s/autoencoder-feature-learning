#include <cstdio>
#include "include/gpu/gpu_layers.h"
#include <cuda_runtime.h>

#define CHECK(call)                                        \
  do                                                       \
  {                                                        \
    cudaError_t err = call;                                \
    if (err != cudaSuccess)                                \
    {                                                      \
      printf("CUDA error %s:%d: %s\n",                     \
             __FILE__, __LINE__, cudaGetErrorString(err)); \
    }                                                      \
  } while (0)

// ================= Conv2D 3x3 =================
__global__ void conv2d_kernel(const float *__restrict__ in,
                              const float *__restrict__ W,
                              float *__restrict__ out,
                              int n, int width, int height,
                              int in_c, int out_c)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * width * height * out_c;
  if (idx >= total)
    return;

  const int KH = 3, KW = 3;

  int f = idx % out_c;
  int tmp = idx / out_c;
  int j = tmp % width;
  tmp /= width;
  int i = tmp % height;
  int img = tmp / height;

  float sum = 0.0f;

  for (int fi = 0; fi < KH; ++fi)
  {
    int row = i + fi - KH / 2;
    if (row < 0 || row >= height)
      continue;

    for (int fj = 0; fj < KW; ++fj)
    {
      int col = j + fj - KW / 2;
      if (col < 0 || col >= width)
        continue;

      const float *in_pix = in + ((img * height + row) * width + col) * in_c;
      const float *w_ptr = W + ((f * KH + fi) * KW + fj) * in_c;

      for (int c = 0; c < in_c; ++c)
      {
        sum += in_pix[c] * w_ptr[c];
      }
    }
  }

  out[idx] = sum;
}

void gpu_conv2D(const float *in, const float *filter, float *out,
                int n, int width, int height, int depth, int n_filter)
{
  int total = n * width * height * n_filter;
  int block = 256;
  int grid = (total + block - 1) / block;
  conv2d_kernel<<<grid, block>>>(in, filter, out,
                                 n, width, height,
                                 depth, n_filter);
  CHECK(cudaDeviceSynchronize());
}

// ================= Add bias =================
__global__ void add_bias_kernel(const float *__restrict__ in,
                                const float *__restrict__ bias,
                                float *__restrict__ out,
                                int n, int width, int height, int depth)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * width * height * depth;
  if (idx >= total)
    return;

  int c = idx % depth;
  out[idx] = in[idx] + bias[c];
}

void gpu_add_bias(const float *in, const float *bias, float *out,
                  int n, int width, int height, int depth)
{
  int total = n * width * height * depth;
  int block = 256;
  int grid = (total + block - 1) / block;
  add_bias_kernel<<<grid, block>>>(in, bias, out,
                                   n, width, height, depth);
  CHECK(cudaDeviceSynchronize());
}

// ================= ReLU =================
__global__ void relu_kernel(const float *__restrict__ in,
                            float *__restrict__ out,
                            int total)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;
  float v = in[idx];
  out[idx] = v > 0.0f ? v : 0.0f;
}

void gpu_relu(const float *in, float *out,
              int n, int width, int height, int depth)
{
  int total = n * width * height * depth;
  int block = 256;
  int grid = (total + block - 1) / block;
  relu_kernel<<<grid, block>>>(in, out, total);
  CHECK(cudaDeviceSynchronize());
}

// ================= MaxPool 2x2 =================
__global__ void maxpool2x2_kernel(const float *__restrict__ in,
                                  float *__restrict__ out,
                                  int n, int width, int height, int depth)
{
  int out_w = width / 2;
  int out_h = height / 2;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * out_w * out_h * depth;
  if (idx >= total)
    return;

  int c = idx % depth;
  int tmp = idx / depth;
  int j = tmp % out_w;
  tmp /= out_w;
  int i = tmp % out_h;
  int img = tmp / out_h;

  int in_i0 = 2 * i;
  int in_j0 = 2 * j;

  const float *p00 = in + ((img * height + in_i0) * width + in_j0) * depth;
  const float *p01 = in + ((img * height + in_i0) * width + (in_j0 + 1)) * depth;
  const float *p10 = in + ((img * height + in_i0 + 1) * width + in_j0) * depth;
  const float *p11 = in + ((img * height + in_i0 + 1) * width + (in_j0 + 1)) * depth;

  float m0 = fmaxf(p00[c], p01[c]);
  float m1 = fmaxf(p10[c], p11[c]);
  out[idx] = fmaxf(m0, m1);
}

void gpu_max_pooling(const float *in, float *out,
                     int n, int width, int height, int depth)
{
  int out_w = width / 2;
  int out_h = height / 2;
  int total = n * out_w * out_h * depth;
  int block = 256;
  int grid = (total + block - 1) / block;
  maxpool2x2_kernel<<<grid, block>>>(in, out,
                                     n, width, height, depth);
  CHECK(cudaDeviceSynchronize());
}

// ================= Upsampling 2x =================
__global__ void upsampling2x_kernel(const float *__restrict__ in,
                                    float *__restrict__ out,
                                    int n, int width, int height, int depth)
{
  int out_w = width * 2;
  int out_h = height * 2;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * out_w * out_h * depth;
  if (idx >= total)
    return;

  int c = idx % depth;
  int tmp = idx / depth;
  int j = tmp % out_w;
  tmp /= out_w;
  int i = tmp % out_h;
  int img = tmp / out_h;

  int src_i = i / 2;
  int src_j = j / 2;

  const float *in_pix = in + ((img * height + src_i) * width + src_j) * depth;
  out[idx] = in_pix[c];
}

void gpu_upsampling(const float *in, float *out,
                    int n, int width, int height, int depth)
{
  int out_w = width * 2;
  int out_h = height * 2;
  int total = n * out_w * out_h * depth;
  int block = 256;
  int grid = (total + block - 1) / block;
  upsampling2x_kernel<<<grid, block>>>(in, out,
                                       n, width, height, depth);
  CHECK(cudaDeviceSynchronize());
}

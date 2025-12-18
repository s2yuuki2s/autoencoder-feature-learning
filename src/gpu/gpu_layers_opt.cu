#include <cuda_runtime.h>
#include <cstdio>
#include "include/gpu/gpu_layers_opt.h"

#define TILE_W 16
#define TILE_H 16
#define HALO 1
#define SMEM_DIM (TILE_W + 2 * HALO) // 18x18

// ======================================================================
// 1. FUSED FORWARD KERNEL (Shared Memory + Bias + ReLU)
// ======================================================================
__global__ void conv2d_fused_kernel(const float *__restrict__ in,
                                    const float *__restrict__ W,
                                    const float *__restrict__ bias,
                                    float *__restrict__ out,
                                    int n, int width, int height,
                                    int in_c, int out_c,
                                    bool use_relu)
{
  __shared__ float s_in[SMEM_DIM][SMEM_DIM];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row_out = blockIdx.y * TILE_H + ty;
  int col_out = blockIdx.x * TILE_W + tx;

  int img_idx = blockIdx.z / out_c;
  int f_idx = blockIdx.z % out_c; // Filter index

  int tid = ty * TILE_W + tx;
  int tile_r = blockIdx.y * TILE_H - HALO;
  int tile_c = blockIdx.x * TILE_W - HALO;

  float sum = 0.0f;

  // Loop qua từng channel input
  for (int c = 0; c < in_c; ++c)
  {
    // 1. Load Tiled Input vào Shared Memory
    for (int i = tid; i < SMEM_DIM * SMEM_DIM; i += TILE_W * TILE_H)
    {
      int r_s = i / SMEM_DIM;
      int c_s = i % SMEM_DIM;
      int r_g = tile_r + r_s;
      int c_g = tile_c + c_s;
      float val = 0.0f;
      if (r_g >= 0 && r_g < height && c_g >= 0 && c_g < width)
      {
        val = in[((img_idx * height + r_g) * width + c_g) * in_c + c];
      }
      s_in[r_s][c_s] = val;
    }
    __syncthreads();

    // 2. Tính Convolution
    if (row_out < height && col_out < width)
    {
      for (int ki = 0; ki < 3; ++ki)
      {
        for (int kj = 0; kj < 3; ++kj)
        {
          // W layout: [out_c][3][3][in_c]
          int w_idx = ((f_idx * 3 + ki) * 3 + kj) * in_c + c;
          sum += s_in[ty + ki][tx + kj] * W[w_idx];
        }
      }
    }
    __syncthreads();
  }

  // 3. Add Bias & ReLU & Write Output
  if (row_out < height && col_out < width)
  {
    // Add Bias
    sum += bias[f_idx];

    // ReLU Fusion
    if (use_relu)
    {
      sum = fmaxf(0.0f, sum);
    }

    int out_idx = ((img_idx * height + row_out) * width + col_out) * out_c + f_idx;
    out[out_idx] = sum;
  }
}

void gpu_conv2d_fused_forward(const float *in, const float *W, const float *bias, float *out,
                              int n, int width, int height, int in_c, int out_c, bool use_relu)
{
  dim3 block(TILE_W, TILE_H);
  dim3 grid((width + TILE_W - 1) / TILE_W, (height + TILE_H - 1) / TILE_H, n * out_c);
  conv2d_fused_kernel<<<grid, block>>>(in, W, bias, out, n, width, height, in_c, out_c, use_relu);
  cudaDeviceSynchronize();
}

// ======================================================================
// 2. BACKWARD KERNELS (Split Strategy)
// ======================================================================

// Kernel 2.1: Tính dIn (Tiled Shared Memory, No Atomic)
__global__ void conv2d_bw_input_tiled_kernel(const float *__restrict__ grad_out,
                                             const float *__restrict__ W,
                                             float *__restrict__ grad_in,
                                             int n, int width, int height, int in_c, int out_c)
{
  __shared__ float s_g_out[SMEM_DIM][SMEM_DIM];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row_in = blockIdx.y * TILE_H + ty; // Output coord (grad_in)
  int col_in = blockIdx.x * TILE_W + tx;

  int img_idx = blockIdx.z / in_c;
  int c_in = blockIdx.z % in_c; // Input Channel we are computing

  int tid = ty * TILE_W + tx;
  int tile_r = blockIdx.y * TILE_H - HALO;
  int tile_c = blockIdx.x * TILE_W - HALO;

  float sum = 0.0f;

  // Loop over OUT channels
  for (int f = 0; f < out_c; ++f)
  {
    // Load grad_out tile
    for (int i = tid; i < SMEM_DIM * SMEM_DIM; i += TILE_W * TILE_H)
    {
      int r_s = i / SMEM_DIM;
      int c_s = i % SMEM_DIM;
      int r_g = tile_r + r_s;
      int c_g = tile_c + c_s;
      float val = 0.0f;
      if (r_g >= 0 && r_g < height && c_g >= 0 && c_g < width)
      {
        val = grad_out[((img_idx * height + r_g) * width + c_g) * out_c + f];
      }
      s_g_out[r_s][c_s] = val;
    }
    __syncthreads();

    if (row_in < height && col_in < width)
    {
#pragma unroll
      for (int ki = 0; ki < 3; ++ki)
      {
#pragma unroll
        for (int kj = 0; kj < 3; ++kj)
        {
          // Correlation with flipped kernel
          // s_g_out corresponds to padded grad_out
          float g = s_g_out[ty + ki][tx + kj];

          // Flipped weights: W[f, 2-ki, 2-kj, c_in]
          int w_idx = ((f * 3 + (2 - ki)) * 3 + (2 - kj)) * in_c + c_in;
          sum += g * W[w_idx];
        }
      }
    }
    __syncthreads();
  }

  if (row_in < height && col_in < width)
  {
    grad_in[((img_idx * height + row_in) * width + col_in) * in_c + c_in] = sum;
  }
}

void gpu_conv2d_backward_input(const float *grad_out, const float *W, float *grad_in,
                               int n, int width, int height, int in_c, int out_c)
{
  dim3 block(TILE_W, TILE_H);
  dim3 grid((width + TILE_W - 1) / TILE_W, (height + TILE_H - 1) / TILE_H, n * in_c);
  conv2d_bw_input_tiled_kernel<<<grid, block>>>(grad_out, W, grad_in, n, width, height, in_c, out_c);
  cudaDeviceSynchronize();
}

// Kernel 2.2: Tính dW, db (Atomic)
__global__ void conv2d_bw_params_kernel(const float *__restrict__ in,
                                        const float *__restrict__ grad_out,
                                        float *__restrict__ grad_W,
                                        float *__restrict__ grad_b,
                                        int total, int width, int height, int in_c, int out_c)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return; // total = n * h * w * out_c

  int f = idx % out_c;
  int tmp = idx / out_c;
  int j = tmp % width;
  tmp /= width;
  int i = tmp % height;
  int img = tmp / height;

  float go = grad_out[idx];

  // 1. Grad Bias
  atomicAdd(&grad_b[f], go);

  // 2. Grad Weights
  for (int ki = 0; ki < 3; ++ki)
  {
    int r = i + ki - 1; // padding=1 implies input offset
    if (r < 0 || r >= height)
      continue;
    for (int kj = 0; kj < 3; ++kj)
    {
      int c = j + kj - 1;
      if (c < 0 || c >= width)
        continue;

      const float *in_pix = in + ((img * height + r) * width + c) * in_c;
      float *dw_base = grad_W + ((f * 3 + ki) * 3 + kj) * in_c;

      for (int ci = 0; ci < in_c; ++ci)
      {
        atomicAdd(&dw_base[ci], go * in_pix[ci]);
      }
    }
  }
}

void gpu_conv2d_backward_params(const float *in, const float *grad_out,
                                float *grad_W, float *grad_b,
                                int n, int width, int height, int in_c, int out_c)
{
  int total = n * width * height * out_c;
  int block = 256;
  int grid = (total + block - 1) / block;
  conv2d_bw_params_kernel<<<grid, block>>>(in, grad_out, grad_W, grad_b, total, width, height, in_c, out_c);
  cudaDeviceSynchronize();
}

// ======================================================================
// 3. OTHER LAYERS (Helper Wrappers)
// ======================================================================

// MaxPool Forward
__global__ void maxpool2x2_k(const float *in, float *out, int n, int w, int h, int d)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int out_w = w / 2;
  int out_h = h / 2;
  if (idx >= n * out_w * out_h * d)
    return;
  int c = idx % d;
  int tmp = idx / d;
  int j = tmp % out_w;
  tmp /= out_w;
  int i = tmp % out_h;
  int img = tmp / out_h;
  int i0 = 2 * i;
  int j0 = 2 * j;
  const float *base = in + (img * h * w + i0 * w + j0) * d + c;
  float m = fmaxf(fmaxf(base[0], base[d]), fmaxf(base[w * d], base[(w + 1) * d]));
  out[idx] = m;
}
void gpu_max_pooling(const float *in, float *out, int n, int width, int height, int depth)
{
  int total = n * (width / 2) * (height / 2) * depth;
  maxpool2x2_k<<<(total + 255) / 256, 256>>>(in, out, n, width, height, depth);
  cudaDeviceSynchronize();
}

// Upsample Forward
__global__ void upsample2x_k(const float *in, float *out, int n, int w, int h, int d)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int out_w = w * 2;
  int out_h = h * 2;
  if (idx >= n * out_w * out_h * d)
    return;
  int c = idx % d;
  int tmp = idx / d;
  int j = tmp % out_w;
  tmp /= out_w;
  int i = tmp % out_h;
  int img = tmp / out_h;
  out[idx] = in[((img * h + i / 2) * w + j / 2) * d + c];
}
void gpu_upsampling(const float *in, float *out, int n, int width, int height, int depth)
{
  int total = n * (width * 2) * (height * 2) * depth;
  upsample2x_k<<<(total + 255) / 256, 256>>>(in, out, n, width, height, depth);
  cudaDeviceSynchronize();
}

// ReLU Backward (Dựa trên output đã activated)
// Nếu y = ReLU(x) > 0 thì grad truyền qua, ngược lại 0.
__global__ void relu_bw_k(const float *out_val, const float *grad_out, float *grad_in, int total)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total)
    grad_in[idx] = (out_val[idx] > 0.0f) ? grad_out[idx] : 0.0f;
}
void gpu_relu_backward(const float *feat_out, const float *grad_out, float *grad_in, int total)
{
  relu_bw_k<<<(total + 255) / 256, 256>>>(feat_out, grad_out, grad_in, total);
  cudaDeviceSynchronize();
}

// Upsample Backward
__global__ void ups_bw_k(const float *g_out, float *g_in, int n, int w, int h, int d)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // index of Input
  if (idx >= n * w * h * d)
    return;
  int c = idx % d;
  int tmp = idx / d;
  int j = tmp % w;
  tmp /= w;
  int i = tmp % h;
  int img = tmp / h;
  int out_w = w * 2;
  const float *base = g_out + (img * h * 2 * out_w + i * 2 * out_w + j * 2) * d + c;
  g_in[idx] += base[0] + base[d] + base[out_w * d] + base[(out_w + 1) * d];
}
void gpu_upsample_backward(const float *grad_out, float *grad_in, int n, int width, int height, int depth)
{
  ups_bw_k<<<(n * width * height * depth + 255) / 256, 256>>>(grad_out, grad_in, n, width, height, depth);
  cudaDeviceSynchronize();
}

// MaxPool Backward
__global__ void pool_bw_k(const float *in, const float *out, const float *g_out, float *g_in, int n, int w, int h, int d)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // idx of Output
  int out_w = w / 2;
  int out_h = h / 2;
  if (idx >= n * out_w * out_h * d)
    return;
  int c = idx % d;
  int tmp = idx / d;
  int j = tmp % out_w;
  tmp /= out_w;
  int i = tmp % out_h;
  int img = tmp / out_h;

  float val = out[idx];
  float go = g_out[idx];
  int i0 = 2 * i;
  int j0 = 2 * j;
  const float *base_in = in + (img * h * w + i0 * w + j0) * d + c;
  float *base_gin = g_in + (img * h * w + i0 * w + j0) * d + c;

  if (base_in[0] == val)
    atomicAdd(&base_gin[0], go);
  else if (base_in[d] == val)
    atomicAdd(&base_gin[d], go);
  else if (base_in[w * d] == val)
    atomicAdd(&base_gin[w * d], go);
  else
    atomicAdd(&base_gin[(w + 1) * d], go);
}
void gpu_maxpool_backward(const float *in, const float *out, const float *grad_out, float *grad_in, int n, int width, int height, int depth)
{
  int total = n * (width / 2) * (height / 2) * depth;
  pool_bw_k<<<(total + 255) / 256, 256>>>(in, out, grad_out, grad_in, n, width, height, depth);
  cudaDeviceSynchronize();
}
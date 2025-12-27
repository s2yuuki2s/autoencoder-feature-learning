#include <cstdio>
#include <random>
#include <cstring>
#include <cstdlib>

#include "gpu/gpu_autoencoder.h"
#include "gpu/gpu_layers.h"
#include "cpu/cpu_layers.h"
#include "constants.h"
#include "data_loader.h"

#include <cuda_runtime.h>

#define CHECK(call)                                         \
  do                                                        \
  {                                                         \
    cudaError_t err = call;                                 \
    if (err != cudaSuccess)                                 \
    {                                                       \
      fprintf(stderr, "CUDA error %s:%d: %s\n",             \
              __FILE__, __LINE__, cudaGetErrorString(err)); \
      std::abort();                                         \
    }                                                       \
  } while (0)

static float rand_uniform(std::mt19937 &rng, float a, float b)
{
  std::uniform_real_distribution<float> dist(a, b);
  return dist(rng);
}

// ================== KERNELS BACKWARD & UPDATE ==================

// dL/d(recon) cho MSE: loss = 1/N * sum (recon - target)^2
__global__ void mse_grad_kernel(const float *__restrict__ recon,
                                const float *__restrict__ target,
                                float *__restrict__ grad_recon,
                                int total, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;
  float diff = recon[idx] - target[idx];
  // float scale = 2.0f / (float)total;
  float scale = 2.0f / (float)total; // scale theo batch size
  grad_recon[idx] = scale * diff;
}

// conv2d_backward giống conv2d_backward CPU nhưng chạy song song
// in:       (n, H, W, in_c)
// grad_out: (n, H, W, out_c)
// W:        (out_c, 3, 3, in_c)
// grad_in:  (n, H, W, in_c)  (đã memset 0)
// grad_W:   (out_c,3,3,in_c) (đã memset 0 trước mỗi batch)
// grad_b:   (out_c)          (đã memset 0 trước mỗi batch)
__global__ void conv2d_backward_kernel(
    const float *__restrict__ in,
    const float *__restrict__ grad_out,
    const float *__restrict__ W,
    int n, int width, int height, int in_c, int out_c,
    float *__restrict__ grad_in,
    float *__restrict__ grad_W,
    float *__restrict__ grad_b)
{
  const int KH = 3, KW = 3;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * height * width * out_c;
  if (idx >= total)
    return;

  int f = idx % out_c;
  int tmp = idx / out_c;
  int j = tmp % width;
  tmp /= width;
  int i = tmp % height;
  int img = tmp / height;

  const float *grad_out_pix = grad_out + img * width * height * out_c + (i * width + j) * out_c;

  float go = grad_out_pix[f];

  // db
  atomicAdd(&grad_b[f], go);

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

      const float *in_pix = in + img * width * height * in_c + (row * width + col) * in_c;
      const float *w_ptr = W + f * KH * KW * in_c + (fi * KW + fj) * in_c;

      float *grad_in_pix = grad_in + img * width * height * in_c + (row * width + col) * in_c;
      float *grad_w_ptr = grad_W + f * KH * KW * in_c + (fi * KW + fj) * in_c;

      for (int c = 0; c < in_c; ++c)
      {
        atomicAdd(&grad_w_ptr[c], go * in_pix[c]);
        atomicAdd(&grad_in_pix[c], go * w_ptr[c]);
      }
    }
  }
}

// ReLU backward: grad_in = (in > 0 ? grad_out : 0)
__global__ void relu_backward_kernel(
    const float *__restrict__ in,
    const float *__restrict__ grad_out,
    float *__restrict__ grad_in,
    int total)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;
  grad_in[idx] = (in[idx] > 0.0f) ? grad_out[idx] : 0.0f;
}

// MaxPool 2x2 backward, stride 2
// in:       (n,H,W,C)
// out:      (n,H/2,W/2,C)
// grad_out: (n,H/2,W/2,C)
// grad_in:  (n,H,W,C) (đã memset 0)
__global__ void maxpool2x2_backward_kernel(
    const float *__restrict__ in,
    const float *__restrict__ out,
    const float *__restrict__ grad_out,
    int n, int width, int height, int depth,
    float *__restrict__ grad_in)
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

  const float *out_pix = out + img * out_w * out_h * depth + (i * out_w + j) * depth;
  const float *grad_out_pix = grad_out + img * out_w * out_h * depth + (i * out_w + j) * depth;

  int in_i0 = 2 * i;
  int in_j0 = 2 * j;

  const float *p00 = in + img * width * height * depth + (in_i0 * width + in_j0) * depth;
  const float *p01 = in + img * width * height * depth + (in_i0 * width + (in_j0 + 1)) * depth;
  const float *p10 = in + img * width * height * depth + ((in_i0 + 1) * width + in_j0) * depth;

  float *g00 = grad_in + img * width * height * depth + (in_i0 * width + in_j0) * depth;
  float *g01 = grad_in + img * width * height * depth + (in_i0 * width + (in_j0 + 1)) * depth;
  float *g10 = grad_in + img * width * height * depth + ((in_i0 + 1) * width + in_j0) * depth;
  float *g11 = grad_in + img * width * height * depth + ((in_i0 + 1) * width + (in_j0 + 1)) * depth;

  float out_val = out_pix[c];
  float go = grad_out_pix[c];

  // route gradient to the max location
  // if (p00[c] == out_val)
  //   g00[c] += go;
  // else if (p01[c] == out_val)
  //   g01[c] += go;
  // else if (p10[c] == out_val)
  //   g10[c] += go;
  // else
  //   g11[c] += go;
  if (p00[c] == out_val)
    atomicAdd(&g00[c], go);
  else if (p01[c] == out_val)
    atomicAdd(&g01[c], go);
  else if (p10[c] == out_val)
    atomicAdd(&g10[c], go);
  else
    atomicAdd(&g11[c], go);
}

// Upsample 2x backward:
// grad_out: (n,2H,2W,C)
// grad_in:  (n,H,W,C) (đã memset 0)
// mỗi pixel in nhận tổng 4 pixel out
__global__ void upsample2x_backward_kernel(
    const float *__restrict__ grad_out,
    int n, int width, int height, int depth,
    float *__restrict__ grad_in)
{
  int out_w = width * 2;
  int out_h = height * 2;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * height * width * depth;
  if (idx >= total)
    return;

  int c = idx % depth;
  int tmp = idx / depth;
  int j = tmp % width;
  tmp /= width;
  int i = tmp % height;
  int img = tmp / height;

  float *gin = grad_in + img * width * height * depth + (i * width + j) * depth;

  const float *g00 = grad_out + img * out_w * out_h * depth + ((2 * i) * out_w + (2 * j)) * depth;
  const float *g01 = grad_out + img * out_w * out_h * depth + ((2 * i) * out_w + (2 * j + 1)) * depth;
  const float *g10 = grad_out + img * out_w * out_h * depth + ((2 * i + 1) * out_w + (2 * j)) * depth;
  const float *g11 = grad_out + img * out_w * out_h * depth + ((2 * i + 1) * out_w + (2 * j + 1)) * depth;

  gin[c] += g00[c] + g01[c] + g10[c] + g11[c];
}

// SGD update: w[i] -= lr * dw[i]
__global__ void sgd_update_kernel(float *w, const float *dw, float lr, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  w[idx] -= lr * dw[idx];
}

// ================== Gpu_Autoencoder IMPLEMENTATION ==================

Gpu_Autoencoder::Gpu_Autoencoder(int batch_size_)
    : d_W1(nullptr), d_b1(nullptr),
      d_W2(nullptr), d_b2(nullptr),
      d_W3(nullptr), d_b3(nullptr),
      d_W4(nullptr), d_b4(nullptr),
      d_W5(nullptr), d_b5(nullptr),
      d_h1(nullptr), d_p1(nullptr), d_h2(nullptr), d_encoded(nullptr),
      d_h3(nullptr), d_u1(nullptr), d_h4(nullptr), d_u2(nullptr), d_recon(nullptr),
      d_input(nullptr), d_target(nullptr),
      d_dW1(nullptr), d_db1(nullptr),
      d_dW2(nullptr), d_db2(nullptr),
      d_dW3(nullptr), d_db3(nullptr),
      d_dW4(nullptr), d_db4(nullptr),
      d_dW5(nullptr), d_db5(nullptr),
      d_g_recon(nullptr), d_g_u2(nullptr), d_g_h4(nullptr), d_g_u1(nullptr),
      d_g_h3(nullptr), d_g_encoded(nullptr), d_g_h2(nullptr), d_g_p1(nullptr),
      d_g_h1(nullptr), d_g_input(nullptr),
      recon_host(nullptr),
      batch_size(batch_size_)
{
  int B = batch_size;

  int W1_size = 256 * 3 * 3 * 3;
  int W2_size = 128 * 3 * 3 * 256;
  int W3_size = 128 * 3 * 3 * 128;
  int W4_size = 256 * 3 * 3 * 128;
  int W5_size = 3 * 3 * 3 * 256;

  CHECK(cudaMalloc(&d_W1, W1_size * sizeof(float)));
  CHECK(cudaMalloc(&d_W2, W2_size * sizeof(float)));
  CHECK(cudaMalloc(&d_W3, W3_size * sizeof(float)));
  CHECK(cudaMalloc(&d_W4, W4_size * sizeof(float)));
  CHECK(cudaMalloc(&d_W5, W5_size * sizeof(float)));

  CHECK(cudaMalloc(&d_b1, 256 * sizeof(float)));
  CHECK(cudaMalloc(&d_b2, 128 * sizeof(float)));
  CHECK(cudaMalloc(&d_b3, 128 * sizeof(float)));
  CHECK(cudaMalloc(&d_b4, 256 * sizeof(float)));
  CHECK(cudaMalloc(&d_b5, 3 * sizeof(float)));

  // Activations
  CHECK(cudaMalloc(&d_h1, B * 32 * 32 * 256 * sizeof(float)));
  CHECK(cudaMalloc(&d_p1, B * 16 * 16 * 256 * sizeof(float)));
  CHECK(cudaMalloc(&d_h2, B * 16 * 16 * 128 * sizeof(float)));
  CHECK(cudaMalloc(&d_encoded, B * 8 * 8 * 128 * sizeof(float)));
  CHECK(cudaMalloc(&d_h3, B * 8 * 8 * 128 * sizeof(float)));
  CHECK(cudaMalloc(&d_u1, B * 16 * 16 * 128 * sizeof(float)));
  CHECK(cudaMalloc(&d_h4, B * 16 * 16 * 256 * sizeof(float)));
  CHECK(cudaMalloc(&d_u2, B * 32 * 32 * 256 * sizeof(float)));
  CHECK(cudaMalloc(&d_recon, B * 32 * 32 * 3 * sizeof(float)));

  CHECK(cudaMalloc(&d_input, B * 32 * 32 * 3 * sizeof(float)));
  CHECK(cudaMalloc(&d_target, B * 32 * 32 * 3 * sizeof(float)));

  // Grad weights/biases
  CHECK(cudaMalloc(&d_dW1, W1_size * sizeof(float)));
  CHECK(cudaMalloc(&d_dW2, W2_size * sizeof(float)));
  CHECK(cudaMalloc(&d_dW3, W3_size * sizeof(float)));
  CHECK(cudaMalloc(&d_dW4, W4_size * sizeof(float)));
  CHECK(cudaMalloc(&d_dW5, W5_size * sizeof(float)));

  CHECK(cudaMalloc(&d_db1, 256 * sizeof(float)));
  CHECK(cudaMalloc(&d_db2, 128 * sizeof(float)));
  CHECK(cudaMalloc(&d_db3, 128 * sizeof(float)));
  CHECK(cudaMalloc(&d_db4, 256 * sizeof(float)));
  CHECK(cudaMalloc(&d_db5, 3 * sizeof(float)));

  // Grad activations
  CHECK(cudaMalloc(&d_g_recon, B * 32 * 32 * 3 * sizeof(float)));
  CHECK(cudaMalloc(&d_g_u2, B * 32 * 32 * 256 * sizeof(float)));
  CHECK(cudaMalloc(&d_g_h4, B * 16 * 16 * 256 * sizeof(float)));
  CHECK(cudaMalloc(&d_g_u1, B * 16 * 16 * 128 * sizeof(float)));
  CHECK(cudaMalloc(&d_g_h3, B * 8 * 8 * 128 * sizeof(float)));
  CHECK(cudaMalloc(&d_g_encoded, B * 8 * 8 * 128 * sizeof(float)));
  CHECK(cudaMalloc(&d_g_h2, B * 16 * 16 * 128 * sizeof(float)));
  CHECK(cudaMalloc(&d_g_p1, B * 16 * 16 * 256 * sizeof(float)));
  CHECK(cudaMalloc(&d_g_h1, B * 32 * 32 * 256 * sizeof(float)));
  CHECK(cudaMalloc(&d_g_input, B * 32 * 32 * 3 * sizeof(float)));

  recon_host = new float[B * 32 * 32 * 3];
}

Gpu_Autoencoder::~Gpu_Autoencoder()
{
  CHECK(cudaFree(d_W1));
  CHECK(cudaFree(d_b1));
  CHECK(cudaFree(d_W2));
  CHECK(cudaFree(d_b2));
  CHECK(cudaFree(d_W3));
  CHECK(cudaFree(d_b3));
  CHECK(cudaFree(d_W4));
  CHECK(cudaFree(d_b4));
  CHECK(cudaFree(d_W5));
  CHECK(cudaFree(d_b5));

  CHECK(cudaFree(d_h1));
  CHECK(cudaFree(d_p1));
  CHECK(cudaFree(d_h2));
  CHECK(cudaFree(d_encoded));
  CHECK(cudaFree(d_h3));
  CHECK(cudaFree(d_u1));
  CHECK(cudaFree(d_h4));
  CHECK(cudaFree(d_u2));
  CHECK(cudaFree(d_recon));
  CHECK(cudaFree(d_input));
  CHECK(cudaFree(d_target));

  CHECK(cudaFree(d_dW1));
  CHECK(cudaFree(d_db1));
  CHECK(cudaFree(d_dW2));
  CHECK(cudaFree(d_db2));
  CHECK(cudaFree(d_dW3));
  CHECK(cudaFree(d_db3));
  CHECK(cudaFree(d_dW4));
  CHECK(cudaFree(d_db4));
  CHECK(cudaFree(d_dW5));
  CHECK(cudaFree(d_db5));

  CHECK(cudaFree(d_g_recon));
  CHECK(cudaFree(d_g_u2));
  CHECK(cudaFree(d_g_h4));
  CHECK(cudaFree(d_g_u1));
  CHECK(cudaFree(d_g_h3));
  CHECK(cudaFree(d_g_encoded));
  CHECK(cudaFree(d_g_h2));
  CHECK(cudaFree(d_g_p1));
  CHECK(cudaFree(d_g_h1));
  CHECK(cudaFree(d_g_input));

  delete[] recon_host;
}

void Gpu_Autoencoder::init_weights(int seed)
{
  std::mt19937 rng(seed);

  int W1_size = 256 * 3 * 3 * 3;
  int W2_size = 128 * 3 * 3 * 256;
  int W3_size = 128 * 3 * 3 * 128;
  int W4_size = 256 * 3 * 3 * 128;
  int W5_size = 3 * 3 * 3 * 256;

  std::vector<float> h_W1(W1_size), h_W2(W2_size),
      h_W3(W3_size), h_W4(W4_size),
      h_W5(W5_size);
  std::vector<float> h_b1(256), h_b2(128), h_b3(128),
      h_b4(256), h_b5(3);

  auto init_array = [&](std::vector<float> &w, float scale)
  {
    for (auto &v : w)
      v = rand_uniform(rng, -scale, scale);
  };

  init_array(h_W1, 0.05f);
  init_array(h_W2, 0.05f);
  init_array(h_W3, 0.05f);
  init_array(h_W4, 0.05f);
  init_array(h_W5, 0.05f);

  std::fill(h_b1.begin(), h_b1.end(), 0.0f);
  std::fill(h_b2.begin(), h_b2.end(), 0.0f);
  std::fill(h_b3.begin(), h_b3.end(), 0.0f);
  std::fill(h_b4.begin(), h_b4.end(), 0.0f);
  std::fill(h_b5.begin(), h_b5.end(), 0.0f);

  CHECK(cudaMemcpy(d_W1, h_W1.data(), W1_size * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_W2, h_W2.data(), W2_size * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_W3, h_W3.data(), W3_size * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_W4, h_W4.data(), W4_size * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_W5, h_W5.data(), W5_size * sizeof(float), cudaMemcpyHostToDevice));

  CHECK(cudaMemcpy(d_b1, h_b1.data(), 256 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b2, h_b2.data(), 128 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b3, h_b3.data(), 128 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b4, h_b4.data(), 256 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b5, h_b5.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));
}

void Gpu_Autoencoder::forward(const float *input,
                              int n, int width, int height, int depth)
{
  (void)width;
  (void)height;
  (void)depth; // assume 32x32x3

  int input_size = n * 32 * 32 * 3;
  CHECK(cudaMemcpy(d_input, input,
                   input_size * sizeof(float),
                   cudaMemcpyHostToDevice));

  // Encoder
  gpu_conv2D(d_input, d_W1, d_h1,
             n, 32, 32, 3, 256);
  gpu_add_bias(d_h1, d_b1, d_h1,
               n, 32, 32, 256);
  gpu_relu(d_h1, d_h1,
           n, 32, 32, 256);

  gpu_max_pooling(d_h1, d_p1,
                  n, 32, 32, 256);

  gpu_conv2D(d_p1, d_W2, d_h2,
             n, 16, 16, 256, 128);
  gpu_add_bias(d_h2, d_b2, d_h2,
               n, 16, 16, 128);
  gpu_relu(d_h2, d_h2,
           n, 16, 16, 128);

  gpu_max_pooling(d_h2, d_encoded,
                  n, 16, 16, 128);

  gpu_conv2D(d_encoded, d_W3, d_h3,
             n, 8, 8, 128, 128);
  gpu_add_bias(d_h3, d_b3, d_h3,
               n, 8, 8, 128);
  gpu_relu(d_h3, d_h3,
           n, 8, 8, 128);

  gpu_upsampling(d_h3, d_u1,
                 n, 8, 8, 128);

  gpu_conv2D(d_u1, d_W4, d_h4,
             n, 16, 16, 128, 256);
  gpu_add_bias(d_h4, d_b4, d_h4,
               n, 16, 16, 256);
  gpu_relu(d_h4, d_h4,
           n, 16, 16, 256);

  gpu_upsampling(d_h4, d_u2,
                 n, 16, 16, 256);

  gpu_conv2D(d_u2, d_W5, d_recon,
             n, 32, 32, 256, 3);
  gpu_add_bias(d_recon, d_b5, d_recon,
               n, 32, 32, 3);

  // copy về host cho eval/visualize
  int recon_size = n * 32 * 32 * 3;
  CHECK(cudaMemcpy(recon_host, d_recon,
                   recon_size * sizeof(float),
                   cudaMemcpyDeviceToHost));
}

void Gpu_Autoencoder::backward(const float *input,
                               const float *target,
                               int n, int width, int height, int depth)
{
  (void)width;
  (void)height;
  (void)depth;

  int B = n;

  int W1_size = 256 * 3 * 3 * 3;
  int W2_size = 128 * 3 * 3 * 256;
  int W3_size = 128 * 3 * 3 * 128;
  int W4_size = 256 * 3 * 3 * 128;
  int W5_size = 3 * 3 * 3 * 256;

  // zero grad weights/bias
  CHECK(cudaMemset(d_dW1, 0, W1_size * sizeof(float)));
  CHECK(cudaMemset(d_dW2, 0, W2_size * sizeof(float)));
  CHECK(cudaMemset(d_dW3, 0, W3_size * sizeof(float)));
  CHECK(cudaMemset(d_dW4, 0, W4_size * sizeof(float)));
  CHECK(cudaMemset(d_dW5, 0, W5_size * sizeof(float)));

  CHECK(cudaMemset(d_db1, 0, 256 * sizeof(float)));
  CHECK(cudaMemset(d_db2, 0, 128 * sizeof(float)));
  CHECK(cudaMemset(d_db3, 0, 128 * sizeof(float)));
  CHECK(cudaMemset(d_db4, 0, 256 * sizeof(float)));
  CHECK(cudaMemset(d_db5, 0, 3 * sizeof(float)));

  // zero grad activations
  CHECK(cudaMemset(d_g_recon, 0, B * 32 * 32 * 3 * sizeof(float)));
  CHECK(cudaMemset(d_g_u2, 0, B * 32 * 32 * 256 * sizeof(float)));
  CHECK(cudaMemset(d_g_h4, 0, B * 16 * 16 * 256 * sizeof(float)));
  CHECK(cudaMemset(d_g_u1, 0, B * 16 * 16 * 128 * sizeof(float)));
  CHECK(cudaMemset(d_g_h3, 0, B * 8 * 8 * 128 * sizeof(float)));
  CHECK(cudaMemset(d_g_encoded, 0, B * 8 * 8 * 128 * sizeof(float)));
  CHECK(cudaMemset(d_g_h2, 0, B * 16 * 16 * 128 * sizeof(float)));
  CHECK(cudaMemset(d_g_p1, 0, B * 16 * 16 * 256 * sizeof(float)));
  CHECK(cudaMemset(d_g_h1, 0, B * 32 * 32 * 256 * sizeof(float)));
  CHECK(cudaMemset(d_g_input, 0, B * 32 * 32 * 3 * sizeof(float)));

  // copy input & target lên device (target = input cho autoencoder)
  int img_size = B * 32 * 32 * 3;
  CHECK(cudaMemcpy(d_input, input,
                   img_size * sizeof(float),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_target, target,
                   img_size * sizeof(float),
                   cudaMemcpyHostToDevice));

  // dL/d(recon)
  int total = B * 32 * 32 * 3;
  {
    int block = 256;
    int grid = (total + block - 1) / block;
    mse_grad_kernel<<<grid, block>>>(
        d_recon, d_target, d_g_recon, total, B);
    CHECK(cudaDeviceSynchronize());
  }

  // conv5 backward: u2 -> recon
  {
    int block = 256;
    int grid = (B * 32 * 32 * 3 + block - 1) / block;
    conv2d_backward_kernel<<<grid, block>>>(
        d_u2, d_g_recon, d_W5,
        B, 32, 32, 256, 3,
        d_g_u2, d_dW5, d_db5);
    CHECK(cudaDeviceSynchronize());
  }

  // upsample2 backward: h4 -> u2
  {
    int total_h4 = B * 16 * 16 * 256;
    int block = 256;
    int grid = (total_h4 + block - 1) / block;
    upsample2x_backward_kernel<<<grid, block>>>(
        d_g_u2,
        B, 16, 16, 256,
        d_g_h4);
    CHECK(cudaDeviceSynchronize());
  }

  // ReLU4 backward: h4
  {
    int total_h4 = B * 16 * 16 * 256;
    int block = 256;
    int grid = (total_h4 + block - 1) / block;
    relu_backward_kernel<<<grid, block>>>(
        d_h4, d_g_h4, d_g_h4, total_h4);
    CHECK(cudaDeviceSynchronize());
  }

  // conv4 backward: u1 -> h4
  {
    int block = 256;
    int grid = (B * 16 * 16 * 256 + block - 1) / block;
    conv2d_backward_kernel<<<grid, block>>>(
        d_u1, d_g_h4, d_W4,
        B, 16, 16, 128, 256,
        d_g_u1, d_dW4, d_db4);
    CHECK(cudaDeviceSynchronize());
  }

  // upsample1 backward: h3 -> u1
  {
    int total_h3 = B * 8 * 8 * 128;
    int block = 256;
    int grid = (total_h3 + block - 1) / block;
    upsample2x_backward_kernel<<<grid, block>>>(
        d_g_u1,
        B, 8, 8, 128,
        d_g_h3);
    CHECK(cudaDeviceSynchronize());
  }

  // ReLU3 backward: h3
  {
    int total_h3 = B * 8 * 8 * 128;
    int block = 256;
    int grid = (total_h3 + block - 1) / block;
    relu_backward_kernel<<<grid, block>>>(
        d_h3, d_g_h3, d_g_h3, total_h3);
    CHECK(cudaDeviceSynchronize());
  }

  // conv3 backward: encoded -> h3
  {
    int block = 256;
    int grid = (B * 8 * 8 * 128 + block - 1) / block;
    conv2d_backward_kernel<<<grid, block>>>(
        d_encoded, d_g_h3, d_W3,
        B, 8, 8, 128, 128,
        d_g_encoded, d_dW3, d_db3);
    CHECK(cudaDeviceSynchronize());
  }

  // pool2 backward: h2 -> encoded
  {
    int total_out = B * 8 * 8 * 128;
    int block = 256;
    int grid = (total_out + block - 1) / block;
    maxpool2x2_backward_kernel<<<grid, block>>>(
        d_h2, d_encoded, d_g_encoded,
        B, 16, 16, 128,
        d_g_h2);
    CHECK(cudaDeviceSynchronize());
  }

  // ReLU2 backward: h2
  {
    int total_h2 = B * 16 * 16 * 128;
    int block = 256;
    int grid = (total_h2 + block - 1) / block;
    relu_backward_kernel<<<grid, block>>>(
        d_h2, d_g_h2, d_g_h2, total_h2);
    CHECK(cudaDeviceSynchronize());
  }

  // conv2 backward: p1 -> h2
  {
    int block = 256;
    int grid = (B * 16 * 16 * 128 + block - 1) / block;
    conv2d_backward_kernel<<<grid, block>>>(
        d_p1, d_g_h2, d_W2,
        B, 16, 16, 256, 128,
        d_g_p1, d_dW2, d_db2);
    CHECK(cudaDeviceSynchronize());
  }

  // pool1 backward: h1 -> p1
  {
    int total_out = B * 16 * 16 * 256;
    int block = 256;
    int grid = (total_out + block - 1) / block;
    maxpool2x2_backward_kernel<<<grid, block>>>(
        d_h1, d_p1, d_g_p1,
        B, 32, 32, 256,
        d_g_h1);
    CHECK(cudaDeviceSynchronize());
  }

  // ReLU1 backward: h1
  {
    int total_h1 = B * 32 * 32 * 256;
    int block = 256;
    int grid = (total_h1 + block - 1) / block;
    relu_backward_kernel<<<grid, block>>>(
        d_h1, d_g_h1, d_g_h1, total_h1);
    CHECK(cudaDeviceSynchronize());
  }

  // conv1 backward: input -> h1
  {
    int block = 256;
    int grid = (B * 32 * 32 * 256 + block - 1) / block;
    conv2d_backward_kernel<<<grid, block>>>(
        d_input, d_g_h1, d_W1,
        B, 32, 32, 3, 256,
        d_g_input, d_dW1, d_db1);
    CHECK(cudaDeviceSynchronize());
  }
}

void Gpu_Autoencoder::update_weights(float lr)
{
  int W1_size = 256 * 3 * 3 * 3;
  int W2_size = 128 * 3 * 3 * 256;
  int W3_size = 128 * 3 * 3 * 128;
  int W4_size = 256 * 3 * 3 * 128;
  int W5_size = 3 * 3 * 3 * 256;

  auto launch_sgd = [&](float *w, float *dw, int size)
  {
    int block = 256;
    int grid = (size + block - 1) / block;
    sgd_update_kernel<<<grid, block>>>(w, dw, lr, size);
    CHECK(cudaDeviceSynchronize());
  };

  launch_sgd(d_W1, d_dW1, W1_size);
  launch_sgd(d_W2, d_dW2, W2_size);
  launch_sgd(d_W3, d_dW3, W3_size);
  launch_sgd(d_W4, d_dW4, W4_size);
  launch_sgd(d_W5, d_dW5, W5_size);

  launch_sgd(d_b1, d_db1, 256);
  launch_sgd(d_b2, d_db2, 128);
  launch_sgd(d_b3, d_db3, 128);
  launch_sgd(d_b4, d_db4, 256);
  launch_sgd(d_b5, d_db5, 3);
}

// Train trên full dataset
void Gpu_Autoencoder::fit(const Dataset &dataset,
                          int n_epoch, int batch_size_,
                          float learning_rate,
                          int seed,
                          int checkpoint,
                          const char *output_dir)
{
  (void)seed;
  (void)checkpoint;
  (void)output_dir;

  for (int epoch = 1; epoch <= n_epoch; ++epoch)
  {
    Dataset shuffled = dataset;
    shuffle_dataset(shuffled);
    std::vector<Dataset> batches =
        create_minibatches(shuffled, batch_size_);

    double epoch_loss = 0.0;
    int num_batches = (int)batches.size();

    for (int b = 0; b < num_batches; ++b)
    {
      Dataset &batch = batches[b];
      int bn = batch.n;

      const float *x = batch.get_data();

      // forward trên GPU
      this->forward(x, bn, 32, 32, 3);
      // loss trên CPU (recon_host đã cập nhật trong forward)
      float loss = cpu_mse_loss(const_cast<float *>(x),
                                recon_host,
                                bn, 32, 32, 3);
      epoch_loss += loss;

      // backward + update trên GPU
      this->backward(x, x, bn, 32, 32, 3);
      this->update_weights(learning_rate);

      float progress = 100.0f * (float)(b + 1) / (float)num_batches;
      std::printf("\r[GPU] Epoch %d/%d - batch %d/%d (%.1f%%)",
                  epoch, n_epoch, b + 1, num_batches, progress);
      std::fflush(stdout);
    }
    std::printf("\n");

    epoch_loss /= (double)num_batches;
    std::printf("[GPU][Epoch %d] loss = %.6f\n", epoch, epoch_loss);
  }
}

// Evaluate reconstruction MSE trên dataset (forward GPU)
float Gpu_Autoencoder::eval(const Dataset &dataset)
{
  int N = dataset.n;
  int image_size = IMAGE_SIZE;

  int B = this->batch_size;
  int num_batches = (N + B - 1) / B;

  double total_loss = 0.0;
  int count = 0;

  for (int b = 0; b < num_batches; ++b)
  {
    int start = b * B;
    int end = (start + B <= N) ? (start + B) : N;
    int bn = end - start;

    const float *x = dataset.get_data() + start * image_size;

    this->forward(x, bn, 32, 32, 3);
    float loss = cpu_mse_loss(const_cast<float *>(x),
                              recon_host,
                              bn, 32, 32, 3);
    total_loss += (double)loss;
    ++count;
  }
  return (float)(total_loss / (double)count);
}

// Encode: hiện cho zeros giống CPU stub
Dataset Gpu_Autoencoder::encode(const Dataset &dataset) const
{
  int N = dataset.n;
  int feat_w = 8, feat_h = 8, feat_c = 128;
  Dataset features(N, feat_w, feat_h, feat_c);
  std::memset(features.get_data(), 0,
              sizeof(float) * N * feat_w * feat_h * feat_c);
  return features;
}

void Gpu_Autoencoder::save(const char *path) const
{
  std::printf("[Gpu_Autoencoder::save] Not implemented, path=%s\n", path);
}

void Gpu_Autoencoder::load(const char *path)
{
  std::printf("[Gpu_Autoencoder::load] Not implemented, path=%s\n", path);
}

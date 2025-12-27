#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cstring>

#include "gpu/gpu_autoencoder_fused.h"
#include "gpu/gpu_layers_opt.h" // Gọi các kernel tối ưu
#include "cpu/cpu_layers.h"     // Để dùng cpu_mse_loss khi eval
#include "data_loader.h"

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

// ======================================================================
// LOCAL HELPER KERNELS (SGD, LOSS)
// ======================================================================

__global__ void sgd_k(float *w, const float *dw, float lr, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < size; i += blockDim.x * gridDim.x)
    w[i] -= lr * dw[i];
}

__global__ void mse_loss_k(const float *r, const float *t, float *l, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float s = 0;
  for (int i = idx; i < n; i += blockDim.x * gridDim.x)
  {
    float d = r[i] - t[i];
    s += d * d;
  }
  if (s > 0)
    atomicAdd(l, s);
}

__global__ void mse_grad_k(const float *r, const float *t, float *g, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float scale = 2.0f / (float)n;
  for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    g[i] = scale * (r[i] - t[i]);
}

// ======================================================================
// CLASS IMPLEMENTATION
// ======================================================================

Gpu_Autoencoder_Fused::Gpu_Autoencoder_Fused(int batch_size_) : batch_size(batch_size_)
{
  int B = batch_size;

  // --- 1. Cấp phát Weights & Biases (Naive Allocation) ---
  CHECK(cudaMalloc(&d_W1, 256 * 3 * 3 * 3 * sizeof(float)));
  CHECK(cudaMalloc(&d_b1, 256 * sizeof(float)));
  CHECK(cudaMalloc(&d_W2, 128 * 3 * 3 * 256 * sizeof(float)));
  CHECK(cudaMalloc(&d_b2, 128 * sizeof(float)));
  CHECK(cudaMalloc(&d_W3, 128 * 3 * 3 * 128 * sizeof(float)));
  CHECK(cudaMalloc(&d_b3, 128 * sizeof(float)));
  CHECK(cudaMalloc(&d_W4, 256 * 3 * 3 * 128 * sizeof(float)));
  CHECK(cudaMalloc(&d_b4, 256 * sizeof(float)));
  CHECK(cudaMalloc(&d_W5, 3 * 3 * 3 * 256 * sizeof(float)));
  CHECK(cudaMalloc(&d_b5, 3 * sizeof(float)));

  // --- 2. Cấp phát Activations ---
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

  // --- 3. Cấp phát Gradients (Weights) ---
  CHECK(cudaMalloc(&d_dW1, 256 * 3 * 3 * 3 * sizeof(float)));
  CHECK(cudaMalloc(&d_db1, 256 * sizeof(float)));
  CHECK(cudaMalloc(&d_dW2, 128 * 3 * 3 * 256 * sizeof(float)));
  CHECK(cudaMalloc(&d_db2, 128 * sizeof(float)));
  CHECK(cudaMalloc(&d_dW3, 128 * 3 * 3 * 128 * sizeof(float)));
  CHECK(cudaMalloc(&d_db3, 128 * sizeof(float)));
  CHECK(cudaMalloc(&d_dW4, 256 * 3 * 3 * 128 * sizeof(float)));
  CHECK(cudaMalloc(&d_db4, 256 * sizeof(float)));
  CHECK(cudaMalloc(&d_dW5, 3 * 3 * 3 * 256 * sizeof(float)));
  CHECK(cudaMalloc(&d_db5, 3 * sizeof(float)));

  // --- 4. Cấp phát Gradients (Activations) ---
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

  // --- 5. Misc ---
  CHECK(cudaMalloc(&d_loss_val, sizeof(float)));
  recon_host = new float[B * 32 * 32 * 3];
}

Gpu_Autoencoder_Fused::~Gpu_Autoencoder_Fused()
{
  // Giải phóng lẻ tẻ (Naive Free)
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

  CHECK(cudaFree(d_loss_val));
  delete[] recon_host;
}

void Gpu_Autoencoder_Fused::init_weights(int seed)
{
  std::mt19937 rng(seed);
  auto init = [&](float *d_w, int size)
  {
    std::vector<float> h_w(size);
    for (auto &v : h_w)
    {
      std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
      v = dist(rng);
    }
    CHECK(cudaMemcpy(d_w, h_w.data(), size * sizeof(float), cudaMemcpyHostToDevice));
  };
  init(d_W1, 256 * 3 * 3 * 3);
  init(d_W2, 128 * 3 * 3 * 256);
  init(d_W3, 128 * 3 * 3 * 128);
  init(d_W4, 256 * 3 * 3 * 128);
  init(d_W5, 3 * 3 * 3 * 256);
}

// === FORWARD: SỬ DỤNG FUSED KERNEL ===
void Gpu_Autoencoder_Fused::forward(const float *input, int n, int width, int height, int depth)
{
  CHECK(cudaMemcpy(d_input, input, n * 32 * 32 * 3 * sizeof(float), cudaMemcpyHostToDevice));

  // Encoder
  // Conv + Bias + ReLU (Fused)
  gpu_conv2d_fused_forward(d_input, d_W1, d_b1, d_h1, n, 32, 32, 3, 256, true);
  gpu_max_pooling(d_h1, d_p1, n, 32, 32, 256);

  gpu_conv2d_fused_forward(d_p1, d_W2, d_b2, d_h2, n, 16, 16, 256, 128, true);
  gpu_max_pooling(d_h2, d_encoded, n, 16, 16, 128);

  // Decoder
  gpu_conv2d_fused_forward(d_encoded, d_W3, d_b3, d_h3, n, 8, 8, 128, 128, true);
  gpu_upsampling(d_h3, d_u1, n, 8, 8, 128);

  gpu_conv2d_fused_forward(d_u1, d_W4, d_b4, d_h4, n, 16, 16, 128, 256, true);
  gpu_upsampling(d_h4, d_u2, n, 16, 16, 256);

  // Output: No ReLU (false)
  gpu_conv2d_fused_forward(d_u2, d_W5, d_b5, d_recon, n, 32, 32, 256, 3, false);
}

// === BACKWARD: SỬ DỤNG FUSED KERNEL ===
void Gpu_Autoencoder_Fused::backward(const float *input, const float *target, int n, int width, int height, int depth)
{
  // Reset Grads
  CHECK(cudaMemset(d_dW1, 0, 256 * 3 * 3 * 3 * sizeof(float)));
  CHECK(cudaMemset(d_db1, 0, 256 * sizeof(float)));
  CHECK(cudaMemset(d_dW2, 0, 128 * 3 * 3 * 256 * sizeof(float)));
  CHECK(cudaMemset(d_db2, 0, 128 * sizeof(float)));
  CHECK(cudaMemset(d_dW3, 0, 128 * 3 * 3 * 128 * sizeof(float)));
  CHECK(cudaMemset(d_db3, 0, 128 * sizeof(float)));
  CHECK(cudaMemset(d_dW4, 0, 256 * 3 * 3 * 128 * sizeof(float)));
  CHECK(cudaMemset(d_db4, 0, 256 * sizeof(float)));
  CHECK(cudaMemset(d_dW5, 0, 3 * 3 * 3 * 256 * sizeof(float)));
  CHECK(cudaMemset(d_db5, 0, 3 * sizeof(float)));

  CHECK(cudaMemset(d_g_recon, 0, n * 32 * 32 * 3 * sizeof(float)));
  CHECK(cudaMemset(d_g_u2, 0, n * 32 * 32 * 256 * sizeof(float)));
  CHECK(cudaMemset(d_g_h4, 0, n * 16 * 16 * 256 * sizeof(float)));
  CHECK(cudaMemset(d_g_u1, 0, n * 16 * 16 * 128 * sizeof(float)));
  CHECK(cudaMemset(d_g_h3, 0, n * 8 * 8 * 128 * sizeof(float)));
  CHECK(cudaMemset(d_g_encoded, 0, n * 8 * 8 * 128 * sizeof(float)));
  CHECK(cudaMemset(d_g_h2, 0, n * 16 * 16 * 128 * sizeof(float)));
  CHECK(cudaMemset(d_g_p1, 0, n * 16 * 16 * 256 * sizeof(float)));
  CHECK(cudaMemset(d_g_h1, 0, n * 32 * 32 * 256 * sizeof(float)));
  CHECK(cudaMemset(d_g_input, 0, n * 32 * 32 * 3 * sizeof(float)));

  // MSE Grad
  int total = n * 32 * 32 * 3;
  mse_grad_k<<<(total + 255) / 256, 256>>>(d_recon, d_target, d_g_recon, total);

  // Backward Flow (Using Fused Kernels)
  // Layer 5
  gpu_conv2d_backward_input(d_g_recon, d_W5, d_g_u2, n, 32, 32, 256, 3);
  gpu_conv2d_backward_params(d_u2, d_g_recon, d_dW5, d_db5, n, 32, 32, 256, 3);

  // Layer 4
  gpu_upsample_backward(d_g_u2, d_g_h4, n, 16, 16, 256);
  gpu_relu_backward(d_h4, d_g_h4, d_g_h4, n * 16 * 16 * 256);
  gpu_conv2d_backward_input(d_g_h4, d_W4, d_g_u1, n, 16, 16, 128, 256);
  gpu_conv2d_backward_params(d_u1, d_g_h4, d_dW4, d_db4, n, 16, 16, 128, 256);

  // Layer 3
  gpu_upsample_backward(d_g_u1, d_g_h3, n, 8, 8, 128);
  gpu_relu_backward(d_h3, d_g_h3, d_g_h3, n * 8 * 8 * 128);
  gpu_conv2d_backward_input(d_g_h3, d_W3, d_g_encoded, n, 8, 8, 128, 128);
  gpu_conv2d_backward_params(d_encoded, d_g_h3, d_dW3, d_db3, n, 8, 8, 128, 128);

  // Layer 2
  gpu_maxpool_backward(d_h2, d_encoded, d_g_encoded, d_g_h2, n, 16, 16, 128);
  gpu_relu_backward(d_h2, d_g_h2, d_g_h2, n * 16 * 16 * 128);
  gpu_conv2d_backward_input(d_g_h2, d_W2, d_g_p1, n, 16, 16, 256, 128);
  gpu_conv2d_backward_params(d_p1, d_g_h2, d_dW2, d_db2, n, 16, 16, 256, 128);

  // Layer 1
  gpu_maxpool_backward(d_h1, d_p1, d_g_p1, d_g_h1, n, 32, 32, 256);
  gpu_relu_backward(d_h1, d_g_h1, d_g_h1, n * 32 * 32 * 256);
  gpu_conv2d_backward_input(d_g_h1, d_W1, d_g_input, n, 32, 32, 3, 256);
  gpu_conv2d_backward_params(d_input, d_g_h1, d_dW1, d_db1, n, 32, 32, 3, 256);

  CHECK(cudaDeviceSynchronize());
}

void Gpu_Autoencoder_Fused::update_weights(float lr)
{
  auto up = [&](float *w, float *dw, int s)
  { sgd_k<<<(s + 255) / 256, 256>>>(w, dw, lr, s); };
  up(d_W1, d_dW1, 256 * 3 * 3 * 3);
  up(d_b1, d_db1, 256);
  up(d_W2, d_dW2, 128 * 3 * 3 * 256);
  up(d_b2, d_db2, 128);
  up(d_W3, d_dW3, 128 * 3 * 3 * 128);
  up(d_b3, d_db3, 128);
  up(d_W4, d_dW4, 256 * 3 * 3 * 128);
  up(d_b4, d_db4, 256);
  up(d_W5, d_dW5, 3 * 3 * 3 * 256);
  up(d_b5, d_db5, 3);
}

void Gpu_Autoencoder_Fused::fit(const Dataset &dataset, int n_epoch, int batch_size_, float learning_rate, int seed, int checkpoint, const char *output_dir)
{
  float h_loss = 0;
  for (int epoch = 1; epoch <= n_epoch; ++epoch)
  {
    Dataset shuffled = dataset;
    shuffle_dataset(shuffled);
    std::vector<Dataset> batches = create_minibatches(shuffled, batch_size_);
    double ep_loss = 0;
    int nb = batches.size();
    for (int b = 0; b < nb; ++b)
    {
      Dataset &batch = batches[b];
      int n = batch.n;
      int sz = n * 32 * 32 * 3;

      forward(batch.get_data(), n, 32, 32, 3);
      CHECK(cudaMemcpy(d_target, batch.get_data(), sz * 4, cudaMemcpyHostToDevice));
      CHECK(cudaMemset(d_loss_val, 0, 4));

      mse_loss_k<<<(sz + 255) / 256, 256>>>(d_recon, d_target, d_loss_val, sz);

      backward(nullptr, nullptr, n, 32, 32, 3);
      update_weights(learning_rate);

      CHECK(cudaMemcpy(&h_loss, d_loss_val, 4, cudaMemcpyDeviceToHost));
      ep_loss += h_loss / sz;
      printf("\r[Fused - No Pool] Epoch %d/%d - Batch %d/%d", epoch, n_epoch, b + 1, nb);
    }
    printf("\nEpoch %d Loss: %.6f\n", epoch, ep_loss / nb);
  }
}

float Gpu_Autoencoder_Fused::eval(const Dataset &dataset)
{
  int N = dataset.n;
  int B = this->batch_size;
  double total_loss = 0;
  int count = 0;
  for (int b = 0; b < (N + B - 1) / B; ++b)
  {
    int s = b * B;
    int e = (s + B <= N) ? s + B : N;
    int n = e - s;
    const float *x = dataset.get_data() + s * 32 * 32 * 3;
    forward(x, n, 32, 32, 3);
    CHECK(cudaMemcpy(recon_host, d_recon, n * 32 * 32 * 3 * 4, cudaMemcpyDeviceToHost));
    total_loss += cpu_mse_loss(const_cast<float *>(x), recon_host, n, 32, 32, 3);
    count++;
  }
  return (float)(total_loss / count);
}

Dataset Gpu_Autoencoder_Fused::encode(const Dataset &dataset) const
{
  int N = dataset.n;
  int B = batch_size;
  Dataset features(N, 8, 8, 128);
  std::memcpy(features.get_labels(), dataset.get_labels(), N * sizeof(int));
  Gpu_Autoencoder_Fused *self = const_cast<Gpu_Autoencoder_Fused *>(this);
  for (int b = 0; b < (N + B - 1) / B; ++b)
  {
    int s = b * B;
    int n = ((s + B) <= N) ? B : (N - s);
    self->forward(dataset.get_data() + s * 3072, n, 32, 32, 3);
    CHECK(cudaMemcpy(features.get_data() + s * 8192, d_encoded, n * 8192 * 4, cudaMemcpyDeviceToHost));
  }
  return features;
}

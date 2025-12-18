#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "gpu/gpu_autoencoder_opt_v2.h"
#include "gpu/gpu_layers_opt.h"
#include "cpu/cpu_layers.h"

#define CHECK(call)                                        \
  {                                                        \
    cudaError_t err = call;                                \
    if (err != cudaSuccess)                                \
    {                                                      \
      printf("CUDA error: %s\n", cudaGetErrorString(err)); \
      abort();                                             \
    }                                                      \
  }

static float *allocate_from_pool(float *pool, size_t &offset, int count)
{
  float *ptr = pool + offset;
  offset += count;
  if (offset % 32 != 0)
    offset += (32 - (offset % 32));
  return ptr;
}

// Update Kernel (SGD)
__global__ void sgd_k(float *w, const float *dw, float lr, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < size; i += blockDim.x * gridDim.x)
    w[i] -= lr * dw[i];
}

// MSE Loss
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

// MSE Grad [ĐÃ SỬA LỖI INF]
// n: total elements (B * H * W * C)
__global__ void mse_grad_k(const float *r, const float *t, float *g, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // SỬA: Chia cho tổng số phần tử (n) thay vì batch_size
  // Để gradient không bị quá lớn (Exploding Gradient)
  float scale = 2.0f / (float)n;

  for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    g[i] = scale * (r[i] - t[i]);
}

// ======================================================================
// CLASS METHODS
// ======================================================================

Gpu_Autoencoder_Opt::Gpu_Autoencoder_Opt(int batch_size_) : batch_size(batch_size_), d_memory_pool(nullptr)
{
  int B = batch_size;
  // Tăng size pool lên 150M để an toàn cho cả Batch 32 và 64
  size_t total_floats = 150000000;
  this->total_memory_size = total_floats * sizeof(float);
  CHECK(cudaMalloc(&d_memory_pool, this->total_memory_size));
  CHECK(cudaMemset(d_memory_pool, 0, this->total_memory_size));

  size_t off = 0;
  // Weights
  d_W1 = allocate_from_pool(d_memory_pool, off, 256 * 3 * 3 * 3);
  d_b1 = allocate_from_pool(d_memory_pool, off, 256);
  d_W2 = allocate_from_pool(d_memory_pool, off, 128 * 3 * 3 * 256);
  d_b2 = allocate_from_pool(d_memory_pool, off, 128);
  d_W3 = allocate_from_pool(d_memory_pool, off, 128 * 3 * 3 * 128);
  d_b3 = allocate_from_pool(d_memory_pool, off, 128);
  d_W4 = allocate_from_pool(d_memory_pool, off, 256 * 3 * 3 * 128);
  d_b4 = allocate_from_pool(d_memory_pool, off, 256);
  d_W5 = allocate_from_pool(d_memory_pool, off, 3 * 3 * 3 * 256);
  d_b5 = allocate_from_pool(d_memory_pool, off, 3);

  // Activations
  d_h1 = allocate_from_pool(d_memory_pool, off, B * 32 * 32 * 256);
  d_p1 = allocate_from_pool(d_memory_pool, off, B * 16 * 16 * 256);
  d_h2 = allocate_from_pool(d_memory_pool, off, B * 16 * 16 * 128);
  d_encoded = allocate_from_pool(d_memory_pool, off, B * 8 * 8 * 128);
  d_h3 = allocate_from_pool(d_memory_pool, off, B * 8 * 8 * 128);
  d_u1 = allocate_from_pool(d_memory_pool, off, B * 16 * 16 * 128);
  d_h4 = allocate_from_pool(d_memory_pool, off, B * 16 * 16 * 256);
  d_u2 = allocate_from_pool(d_memory_pool, off, B * 32 * 32 * 256);
  d_recon = allocate_from_pool(d_memory_pool, off, B * 32 * 32 * 3);
  d_input = allocate_from_pool(d_memory_pool, off, B * 32 * 32 * 3);
  d_target = allocate_from_pool(d_memory_pool, off, B * 32 * 32 * 3);

  // Grads
  d_dW1 = allocate_from_pool(d_memory_pool, off, 256 * 3 * 3 * 3);
  d_db1 = allocate_from_pool(d_memory_pool, off, 256);
  d_dW2 = allocate_from_pool(d_memory_pool, off, 128 * 3 * 3 * 256);
  d_db2 = allocate_from_pool(d_memory_pool, off, 128);
  d_dW3 = allocate_from_pool(d_memory_pool, off, 128 * 3 * 3 * 128);
  d_db3 = allocate_from_pool(d_memory_pool, off, 128);
  d_dW4 = allocate_from_pool(d_memory_pool, off, 256 * 3 * 3 * 128);
  d_db4 = allocate_from_pool(d_memory_pool, off, 256);
  d_dW5 = allocate_from_pool(d_memory_pool, off, 3 * 3 * 3 * 256);
  d_db5 = allocate_from_pool(d_memory_pool, off, 3);

  d_g_recon = allocate_from_pool(d_memory_pool, off, B * 32 * 32 * 3);
  d_g_u2 = allocate_from_pool(d_memory_pool, off, B * 32 * 32 * 256);
  d_g_h4 = allocate_from_pool(d_memory_pool, off, B * 16 * 16 * 256);
  d_g_u1 = allocate_from_pool(d_memory_pool, off, B * 16 * 16 * 128);
  d_g_h3 = allocate_from_pool(d_memory_pool, off, B * 8 * 8 * 128);
  d_g_encoded = allocate_from_pool(d_memory_pool, off, B * 8 * 8 * 128);
  d_g_h2 = allocate_from_pool(d_memory_pool, off, B * 16 * 16 * 128);
  d_g_p1 = allocate_from_pool(d_memory_pool, off, B * 16 * 16 * 256);
  d_g_h1 = allocate_from_pool(d_memory_pool, off, B * 32 * 32 * 256);
  d_g_input = allocate_from_pool(d_memory_pool, off, B * 32 * 32 * 3);

  d_loss_val = allocate_from_pool(d_memory_pool, off, 1);
  recon_host = new float[B * 32 * 32 * 3];

  // Safety check
  if (off * sizeof(float) > this->total_memory_size)
  {
    printf("ERROR: Memory Pool Overflow! Needed %zu, allocated %zu\n", off * sizeof(float), this->total_memory_size);
    abort();
  }
}

Gpu_Autoencoder_Opt::~Gpu_Autoencoder_Opt()
{
  CHECK(cudaFree(d_memory_pool));
  delete[] recon_host;
}

void Gpu_Autoencoder_Opt::init_weights(int seed)
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

// FORWARD: Sử dụng Fused Kernels
void Gpu_Autoencoder_Opt::forward(const float *input, int n, int width, int height, int depth)
{
  CHECK(cudaMemcpy(d_input, input, n * 32 * 32 * 3 * sizeof(float), cudaMemcpyHostToDevice));

  // Encoder
  gpu_conv2d_fused_forward(d_input, d_W1, d_b1, d_h1, n, 32, 32, 3, 256, true); // true = Use ReLU
  gpu_max_pooling(d_h1, d_p1, n, 32, 32, 256);

  gpu_conv2d_fused_forward(d_p1, d_W2, d_b2, d_h2, n, 16, 16, 256, 128, true);
  gpu_max_pooling(d_h2, d_encoded, n, 16, 16, 128);

  // Decoder
  gpu_conv2d_fused_forward(d_encoded, d_W3, d_b3, d_h3, n, 8, 8, 128, 128, true);
  gpu_upsampling(d_h3, d_u1, n, 8, 8, 128);

  gpu_conv2d_fused_forward(d_u1, d_W4, d_b4, d_h4, n, 16, 16, 128, 256, true);
  gpu_upsampling(d_h4, d_u2, n, 16, 16, 256);

  // Output Layer: No ReLU
  gpu_conv2d_fused_forward(d_u2, d_W5, d_b5, d_recon, n, 32, 32, 256, 3, false); // false = No ReLU
}

// BACKWARD: Sử dụng Split Kernels
void Gpu_Autoencoder_Opt::backward(const float *input, const float *target, int n, int width, int height, int depth)
{
  // Reset Grads
  CHECK(cudaMemset(d_dW1, 0, 256 * 27 * 4));
  CHECK(cudaMemset(d_db1, 0, 256 * 4));
  CHECK(cudaMemset(d_dW2, 0, 128 * 3 * 3 * 256 * 4));
  CHECK(cudaMemset(d_db2, 0, 128 * 4));
  CHECK(cudaMemset(d_dW3, 0, 128 * 3 * 3 * 128 * 4));
  CHECK(cudaMemset(d_db3, 0, 128 * 4));
  CHECK(cudaMemset(d_dW4, 0, 256 * 3 * 3 * 128 * 4));
  CHECK(cudaMemset(d_db4, 0, 256 * 4));
  CHECK(cudaMemset(d_dW5, 0, 3 * 3 * 3 * 256 * 4));
  CHECK(cudaMemset(d_db5, 0, 3 * 4));

  // Reset Act Grads (Để cộng dồn an toàn)
  CHECK(cudaMemset(d_g_recon, 0, n * 32 * 32 * 3 * 4));
  CHECK(cudaMemset(d_g_u2, 0, n * 32 * 32 * 256 * 4));
  CHECK(cudaMemset(d_g_h4, 0, n * 16 * 16 * 256 * 4));
  CHECK(cudaMemset(d_g_u1, 0, n * 16 * 16 * 128 * 4));
  CHECK(cudaMemset(d_g_h3, 0, n * 8 * 8 * 128 * 4));
  CHECK(cudaMemset(d_g_encoded, 0, n * 8 * 8 * 128 * 4));
  CHECK(cudaMemset(d_g_h2, 0, n * 16 * 16 * 128 * 4));
  CHECK(cudaMemset(d_g_p1, 0, n * 16 * 16 * 256 * 4));
  CHECK(cudaMemset(d_g_h1, 0, n * 32 * 32 * 256 * 4));
  CHECK(cudaMemset(d_g_input, 0, n * 32 * 32 * 3 * 4));

  // [SỬA] Không copy d_input, d_target ở đây nữa vì đã có sẵn trên GPU từ hàm forward/fit
  // Tránh lỗi crash nếu truyền nullptr vào

  // MSE Grad
  int total = n * 32 * 32 * 3;
  // SỬA: Chỉ truyền total (n), bỏ tham số bs vì không cần thiết nữa
  mse_grad_k<<<(total + 255) / 256, 256>>>(d_recon, d_target, d_g_recon, total);

  // Backward Flow
  // Layer 5 (Recon -> u2). Output layer no ReLU.
  gpu_conv2d_backward_input(d_g_recon, d_W5, d_g_u2, n, 32, 32, 256, 3);
  gpu_conv2d_backward_params(d_u2, d_g_recon, d_dW5, d_db5, n, 32, 32, 256, 3);

  // Layer 4
  gpu_upsample_backward(d_g_u2, d_g_h4, n, 16, 16, 256);
  gpu_relu_backward(d_h4, d_g_h4, d_g_h4, n * 16 * 16 * 256); // d_h4 chứa activated output
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

void Gpu_Autoencoder_Opt::update_weights(float lr)
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

void Gpu_Autoencoder_Opt::fit(const Dataset &dataset, int n_epoch, int batch_size_, float learning_rate, int seed, int checkpoint, const char *output_dir)
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

      // Tính loss
      mse_loss_k<<<(sz + 255) / 256, 256>>>(d_recon, d_target, d_loss_val, sz);

      // Backward: Truyền nullptr vì d_input và d_target đã có sẵn
      backward(nullptr, nullptr, n, 32, 32, 3);

      update_weights(learning_rate);
      CHECK(cudaMemcpy(&h_loss, d_loss_val, 4, cudaMemcpyDeviceToHost));
      ep_loss += h_loss / sz;
      printf("\r[Opt V3 - Fused] Epoch %d/%d - Batch %d/%d (%.1f%%)", epoch, n_epoch, b + 1, nb, 100.0f * (b + 1) / nb);
    }
    printf("\nEpoch %d Loss: %.6f\n", epoch, ep_loss / nb);
  }
}

// Eval, Encode, SaveFeatures: Copy from previous version (No changes needed)
float Gpu_Autoencoder_Opt::eval(const Dataset &dataset)
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
Dataset Gpu_Autoencoder_Opt::encode(const Dataset &dataset) const
{
  int N = dataset.n;
  int B = batch_size;
  Dataset features(N, 8, 8, 128);
  Gpu_Autoencoder_Opt *self = const_cast<Gpu_Autoencoder_Opt *>(this);
  for (int b = 0; b < (N + B - 1) / B; ++b)
  {
    int s = b * B;
    int n = ((s + B) <= N) ? B : (N - s);
    self->forward(dataset.get_data() + s * 3072, n, 32, 32, 3);
    CHECK(cudaMemcpy(features.get_data() + s * 8192, d_encoded, n * 8192 * 4, cudaMemcpyDeviceToHost));
  }
  return features;
}
void Gpu_Autoencoder_Opt::save_features(const Dataset &f, const char *fn) const
{
  FILE *file = fopen(fn, "wb");
  int n = f.n;
  int d = f.width * f.height * f.depth;
  fwrite(&n, 4, 1, file);
  fwrite(&d, 4, 1, file);
  fwrite(f.get_data(), 4, n * d, file);
  fwrite(f.get_labels(), 4, n, file);
  fclose(file);
}
void Gpu_Autoencoder_Opt::save(const char *path) const {}
void Gpu_Autoencoder_Opt::load(const char *path) {}
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cstring>

#include "../../include/gpu/gpu_autoencoder_mempool.h"
#include "../../include/gpu/gpu_layers.h" // Dùng Native Kernels
#include "../../include/cpu/cpu_layers.h" // Dùng CPU Loss để tính toán hiển thị
#include "../../include/data_loader.h"

#define CHECK(call)                                     \
 {                                                      \
  cudaError_t err = call;                               \
  if (err != cudaSuccess)                               \
  {                                                     \
   printf("CUDA error: %s\n", cudaGetErrorString(err)); \
   abort();                                             \
  }                                                     \
 }

// Helper: Cấp phát từ Pool
static float *allocate_from_pool(float *pool, size_t &offset, int count)
{
 float *ptr = pool + offset;
 offset += count;
 // Căn chỉnh bộ nhớ (Alignment) cho an toàn (32 floats = 128 bytes)
 if (offset % 32 != 0)
  offset += (32 - (offset % 32));
 return ptr;
}

// Kernel SGD đơn giản
__global__ void sgd_k_mp(float *w, const float *dw, float lr, int size)
{
 int idx = blockIdx.x * blockDim.x + threadIdx.x;
 for (int i = idx; i < size; i += blockDim.x * gridDim.x)
  w[i] -= lr * dw[i];
}

// Kernel MSE Loss
__global__ void mse_loss_k_mp(const float *r, const float *t, float *l, int n)
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

// Kernel MSE Grad
__global__ void mse_grad_k_mp(const float *r, const float *t, float *g, int n)
{
 int idx = blockIdx.x * blockDim.x + threadIdx.x;
 float scale = 2.0f / (float)n;
 for (int i = idx; i < n; i += blockDim.x * gridDim.x)
  g[i] = scale * (r[i] - t[i]);
}

// ======================================================================

Gpu_Autoencoder_MemPool::Gpu_Autoencoder_MemPool(int batch_size_) : batch_size(batch_size_), d_memory_pool(nullptr)
{
 int B = batch_size;
 // Ước lượng khoảng 150 triệu phần tử float (~600MB) là đủ cho mạng này
 size_t total_floats = 150000000;
 this->total_memory_size = total_floats * sizeof(float);

 // 1. Cấp phát 1 lần duy nhất
 CHECK(cudaMalloc(&d_memory_pool, this->total_memory_size));
 CHECK(cudaMemset(d_memory_pool, 0, this->total_memory_size));

 size_t off = 0;

 // 2. Chia nhỏ pointer từ Pool
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

 if (off * sizeof(float) > this->total_memory_size)
 {
  printf("ERROR: Memory Pool Overflow!\n");
  abort();
 }
}

Gpu_Autoencoder_MemPool::~Gpu_Autoencoder_MemPool()
{
 // Free 1 lần duy nhất
 CHECK(cudaFree(d_memory_pool));
 delete[] recon_host;
}

void Gpu_Autoencoder_MemPool::init_weights(int seed)
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

// === FORWARD: SỬ DỤNG NATIVE KERNEL (gpu_layers.h) ===
void Gpu_Autoencoder_MemPool::forward(const float *input, int n, int width, int height, int depth)
{
 CHECK(cudaMemcpy(d_input, input, n * 32 * 32 * 3 * sizeof(float), cudaMemcpyHostToDevice));

 // Encoder
 gpu_conv2D(d_input, d_W1, d_h1, n, 32, 32, 3, 256);
 gpu_add_bias(d_h1, d_b1, d_h1, n, 32, 32, 256);
 gpu_relu(d_h1, d_h1, n, 32, 32, 256);
 gpu_max_pooling(d_h1, d_p1, n, 32, 32, 256);

 gpu_conv2D(d_p1, d_W2, d_h2, n, 16, 16, 256, 128);
 gpu_add_bias(d_h2, d_b2, d_h2, n, 16, 16, 128);
 gpu_relu(d_h2, d_h2, n, 16, 16, 128);
 gpu_max_pooling(d_h2, d_encoded, n, 16, 16, 128);

 // Decoder
 gpu_conv2D(d_encoded, d_W3, d_h3, n, 8, 8, 128, 128);
 gpu_add_bias(d_h3, d_b3, d_h3, n, 8, 8, 128);
 gpu_relu(d_h3, d_h3, n, 8, 8, 128);
 gpu_upsampling(d_h3, d_u1, n, 8, 8, 128);

 gpu_conv2D(d_u1, d_W4, d_h4, n, 16, 16, 128, 256);
 gpu_add_bias(d_h4, d_b4, d_h4, n, 16, 16, 256);
 gpu_relu(d_h4, d_h4, n, 16, 16, 256);
 gpu_upsampling(d_h4, d_u2, n, 16, 16, 256);

 gpu_conv2D(d_u2, d_W5, d_recon, n, 32, 32, 256, 3);
 gpu_add_bias(d_recon, d_b5, d_recon, n, 32, 32, 3);
}

// === BACKWARD: SỬ DỤNG NATIVE LOGIC (Tính toán Gradient thủ công) ===
// Lưu ý: Bản MemPool này chưa dùng kernel backward tối ưu
void Gpu_Autoencoder_MemPool::backward(const float *input, const float *target, int n, int width, int height, int depth)
{
 // 1. Reset Grads
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

 // 2. Tính MSE Gradient
 int total_out = n * 32 * 32 * 3;
 mse_grad_k_mp<<<(total_out + 255) / 256, 256>>>(d_recon, d_target, d_g_recon, total_out);

 // 3. Backward Flow (Gọi các hàm wrapper Native mới viết)

 // Layer 5: Conv (u2 -> recon)
 gpu_conv2d_backward_native(d_u2, d_g_recon, d_W5, n, 32, 32, 256, 3, d_g_u2, d_dW5, d_db5);

 // Layer 4: Upsample -> ReLU -> Conv
 gpu_upsample_backward_native(d_g_u2, n, 16, 16, 256, d_g_h4);      // g_u2 -> g_h4
 gpu_relu_backward_native(d_h4, d_g_h4, d_g_h4, n * 16 * 16 * 256); // ReLU in-place
 gpu_conv2d_backward_native(d_u1, d_g_h4, d_W4, n, 16, 16, 128, 256, d_g_u1, d_dW4, d_db4);

 // Layer 3: Upsample -> ReLU -> Conv
 gpu_upsample_backward_native(d_g_u1, n, 8, 8, 128, d_g_h3);
 gpu_relu_backward_native(d_h3, d_g_h3, d_g_h3, n * 8 * 8 * 128);
 gpu_conv2d_backward_native(d_encoded, d_g_h3, d_W3, n, 8, 8, 128, 128, d_g_encoded, d_dW3, d_db3);

 // Layer 2: MaxPool -> ReLU -> Conv
 gpu_maxpool_backward_native(d_h2, d_encoded, d_g_encoded, n, 16, 16, 128, d_g_h2);
 gpu_relu_backward_native(d_h2, d_g_h2, d_g_h2, n * 16 * 16 * 128);
 gpu_conv2d_backward_native(d_p1, d_g_h2, d_W2, n, 16, 16, 256, 128, d_g_p1, d_dW2, d_db2);

 // Layer 1: MaxPool -> ReLU -> Conv
 gpu_maxpool_backward_native(d_h1, d_p1, d_g_p1, n, 32, 32, 256, d_g_h1);
 gpu_relu_backward_native(d_h1, d_g_h1, d_g_h1, n * 32 * 32 * 256);
 gpu_conv2d_backward_native(d_input, d_g_h1, d_W1, n, 32, 32, 3, 256, d_g_input, d_dW1, d_db1);
}

void Gpu_Autoencoder_MemPool::update_weights(float lr)
{
 auto up = [&](float *w, float *dw, int s)
 { sgd_k_mp<<<(s + 255) / 256, 256>>>(w, dw, lr, s); };
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

void Gpu_Autoencoder_MemPool::fit(const Dataset &dataset, int n_epoch, int batch_size_, float learning_rate, int seed, int checkpoint, const char *output_dir)
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

   mse_loss_k_mp<<<(sz + 255) / 256, 256>>>(d_recon, d_target, d_loss_val, sz);

   // Backward Native (Lưu ý: bạn cần implement full native backward logic ở trên)
   backward(d_input, d_target, n, 32, 32, 3);

   update_weights(learning_rate);

   CHECK(cudaMemcpy(&h_loss, d_loss_val, 4, cudaMemcpyDeviceToHost));
   ep_loss += h_loss / sz;
   printf("\r[MemPool Only] Epoch %d/%d - Batch %d/%d", epoch, n_epoch, b + 1, nb);
  }
  printf("\nEpoch %d Loss: %.6f\n", epoch, ep_loss / nb);
 }
}

float Gpu_Autoencoder_MemPool::eval(const Dataset &dataset)
{
 int N = dataset.n;
 int B = this->batch_size;
 double total_loss = 0;
 int count = 0;
 for (int b = 0; b < (N + B - 1) / B; ++b)
 {
  int s = b * B;
  int n = ((s + B <= N) ? B : (N - s));
  const float *x = dataset.get_data() + s * 32 * 32 * 3;
  forward(x, n, 32, 32, 3);
  CHECK(cudaMemcpy(recon_host, d_recon, n * 32 * 32 * 3 * 4, cudaMemcpyDeviceToHost));
  total_loss += cpu_mse_loss(const_cast<float *>(x), recon_host, n, 32, 32, 3);
  count++;
 }
 return (float)(total_loss / count);
}

Dataset Gpu_Autoencoder_MemPool::encode(const Dataset &dataset) const
{
 int N = dataset.n;
 int B = batch_size;
 Dataset features(N, 8, 8, 128);
 std::memcpy(features.get_labels(), dataset.get_labels(), N * sizeof(int));
 Gpu_Autoencoder_MemPool *self = const_cast<Gpu_Autoencoder_MemPool *>(this);
 for (int b = 0; b < (N + B - 1) / B; ++b)
 {
  int s = b * B;
  int n = ((s + B) <= N) ? B : (N - s);
  self->forward(dataset.get_data() + s * 3072, n, 32, 32, 3);
  CHECK(cudaMemcpy(features.get_data() + s * 8192, d_encoded, n * 8192 * 4, cudaMemcpyDeviceToHost));
 }
 return features;
}

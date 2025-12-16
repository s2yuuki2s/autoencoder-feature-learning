#ifndef GPU_AUTOENCODER_OPT_H
#define GPU_AUTOENCODER_OPT_H

#include "../data_loader.h"

struct Gpu_Autoencoder_Opt
{
  // === MEMORY POOL ===
  float *d_memory_pool;     // Con trỏ quản lý vùng nhớ tổng
  size_t total_memory_size; // Tổng kích thước byte

  // Các con trỏ dữ liệu (sẽ trỏ vào trong pool, không cần cudaMalloc riêng)
  // Weights & biases
  float *d_W1, *d_b1;
  float *d_W2, *d_b2;
  float *d_W3, *d_b3;
  float *d_W4, *d_b4;
  float *d_W5, *d_b5;

  // Activations
  float *d_h1, *d_p1, *d_h2, *d_encoded;
  float *d_h3, *d_u1, *d_h4, *d_u2, *d_recon;

  // Input & target
  float *d_input;
  float *d_target;

  // Gradients
  float *d_dW1, *d_db1;
  float *d_dW2, *d_db2;
  float *d_dW3, *d_db3;
  float *d_dW4, *d_db4;
  float *d_dW5, *d_db5;

  // Gradients Activations
  float *d_g_recon;
  float *d_g_u2;
  float *d_g_h4;
  float *d_g_u1;
  float *d_g_h3;
  float *d_g_encoded;
  float *d_g_h2;
  float *d_g_p1;
  float *d_g_h1;
  float *d_g_input;

  // Output host
  float *recon_host;
  float *d_loss_val;

  int batch_size;

  Gpu_Autoencoder_Opt(int batch_size_);
  ~Gpu_Autoencoder_Opt();

  void init_weights(int seed);
  void forward(const float *input, int n, int width, int height, int depth);
  void backward(const float *input, const float *target, int n, int width, int height, int depth);
  void update_weights(float lr);
  void fit(const Dataset &dataset, int n_epoch, int batch_size_, float learning_rate, int seed, int checkpoint, const char *output_dir);
  float eval(const Dataset &dataset);

  // Đã sửa lại hàm encode để trả về dữ liệu thật (cho Phase 4 sau này)
  Dataset encode(const Dataset &dataset) const;

  void save_features(const Dataset &features, const char *filename) const;

  void save(const char *path) const;
  void load(const char *path);
};

#endif

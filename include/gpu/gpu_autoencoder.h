#ifndef GPU_AUTOENCODER_H
#define GPU_AUTOENCODER_

#include "data_loader.h"

struct Gpu_Autoencoder
{
  // Weights & biases (device pointers)
  float *d_W1, *d_b1, *d_W2, *d_b2, *d_W3, *d_b3, *d_W4, *d_b4, *d_W5, *d_b5;

  // Activations (device)
  float *d_h1, *d_p1, *d_h2, *d_encoded, *d_h3, *d_u1, *d_h4, *d_u2, *d_recon;

  // Input & target trên device
  float *d_input, *d_target;

  // Gradients cho weights/biases (device)
  float *d_dW1, *d_db1, *d_dW2, *d_db2, *d_dW3, *d_db3, *d_dW4, *d_db4, *d_dW5, *d_db5;

  // Gradients cho activations (device)
  float *d_g_recon, *d_g_u2, *d_g_h4, *d_g_u1, *d_g_h3, *d_g_encoded, *d_g_h2, *d_g_p1, *d_g_h1, *d_g_input;

  // Output recon trên HOST để tính loss / visualize
  float *recon_host;

  int batch_size;

  Gpu_Autoencoder(int batch_size_);
  ~Gpu_Autoencoder();

  // Khởi tạo trọng số (random nhỏ) trên device
  void init_weights(int seed);

  // Forward 1 batch trên GPU
  // input: host pointer, n <= batch_size, ảnh 32x32x3
  void forward(const float *input,
               int n, int width, int height, int depth);

  // Backward toàn bộ mạng trên GPU
  // input, target: host pointer (sẽ được copy lên device)
  void backward(const float *input,
                const float *target,
                int n, int width, int height, int depth);

  // SGD update trên GPU
  void update_weights(float lr);

  // Train full dataset trên GPU
  void fit(const Dataset &dataset,
           int n_epoch, int batch_size_,
           float learning_rate,
           int seed,
           int checkpoint,
           const char *output_dir);

  // Evaluate MSE trên dataset (forward GPU, loss CPU)
  float eval(const Dataset &dataset);

  // Encode: tạm thời trả zeros (giống CPU bản stub)
  Dataset encode(const Dataset &dataset) const;

  // Save/load: stub
  void save(const char *path) const;
  void load(const char *path);
};

#endif // GPU_AUTOENCODER_H

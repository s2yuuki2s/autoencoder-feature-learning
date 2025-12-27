#ifndef GPU_AUTOENCODER_MEMPOOL_H
#define GPU_AUTOENCODER_MEMPOOL_H

#include "data_loader.h"

class Gpu_Autoencoder_MemPool
{
public:
 Gpu_Autoencoder_MemPool(int batch_size);
 ~Gpu_Autoencoder_MemPool();

 void init_weights(int seed);
 void fit(const Dataset &dataset, int n_epoch, int batch_size, float learning_rate, int seed, int checkpoint, const char *output_dir);
 float eval(const Dataset &dataset);
 Dataset encode(const Dataset &dataset) const;

private:
 void forward(const float *input, int n, int width, int height, int depth);
 void backward(const float *input, const float *target, int n, int width, int height, int depth);
 void update_weights(float lr);

 // === MEMORY POOL ===
 float *d_memory_pool;     // Con trỏ quản lý vùng nhớ lớn
 size_t total_memory_size; // Tổng kích thước

 // Các con trỏ bên dưới sẽ trỏ vào d_memory_pool (không cudaMalloc riêng lẻ)
 float *d_W1, *d_b1, *d_W2, *d_b2, *d_W3, *d_b3, *d_W4, *d_b4, *d_W5, *d_b5;
 float *d_h1, *d_p1, *d_h2, *d_encoded, *d_h3, *d_u1, *d_h4, *d_u2, *d_recon;
 float *d_input, *d_target;
 float *d_dW1, *d_db1, *d_dW2, *d_db2, *d_dW3, *d_db3, *d_dW4, *d_db4, *d_dW5, *d_db5;
 float *d_g_recon, *d_g_u2, *d_g_h4, *d_g_u1, *d_g_h3, *d_g_encoded, *d_g_h2, *d_g_p1, *d_g_h1, *d_g_input;

 float *d_loss_val;
 float *recon_host;
 int batch_size;
};

#endif
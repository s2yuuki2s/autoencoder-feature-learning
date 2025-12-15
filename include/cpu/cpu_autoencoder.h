#ifndef CPU_AUTOENCODER_H
#define CPU_AUTOENCODER_H

#include "../data_loader.h"

struct Cpu_Autoencoder
{
  // Weights & biases
  float *W1, *b1;
  float *W2, *b2;
  float *W3, *b3;
  float *W4, *b4;
  float *W5, *b5;

  // Activations buffers (per batch)
  float *h1, *p1, *h2, *encoded;
  float *h3, *u1, *h4, *u2, *recon;

  // Gradients
  float *dW1, *db1;
  float *dW2, *db2;
  float *dW3, *db3;
  float *dW4, *db4;
  float *dW5, *db5;

  int batch_size;

  Cpu_Autoencoder(int batch_size);
  ~Cpu_Autoencoder();

  void init_weights(int seed);

  void forward(const float *input,
               int n, int width, int height, int depth);

  void backward(const float *input,
                const float *target,
                int n, int width, int height, int depth);

  void update_weights(float lr);

  void fit(const Dataset &dataset,
           int n_epoch, int batch_size,
           float learning_rate,
           int seed,
           int checkpoint,
           const char *output_dir);

  float eval(const Dataset &dataset) const;

  Dataset encode(const Dataset &dataset) const;

  void save(const char *path) const;
  void load(const char *path);
};

#endif

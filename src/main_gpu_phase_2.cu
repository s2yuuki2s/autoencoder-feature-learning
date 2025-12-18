#include <cstdio>
#include <chrono>

#include "../../include/data_loader.h"
#include "../../include/gpu/gpu_autoencoder.h"

int main(int argc, char **argv)
{
  const char *dataset_dir = (argc > 1 ? argv[1] : "./data/cifar-10-batches-bin");

  int n_epoch = 20;
  int batch_size = 32;
  float lr = 1e-3f;
  int seed = 42;
  int checkpoint = 0;
  const char *out_dir = "./checkpoints_gpu";

  std::printf("=== Phase 2: GPU Baseline on CIFAR-10 ===\n");
  std::printf("Dataset dir   : %s\n", dataset_dir);
  std::printf("Epochs        : %d\n", n_epoch);
  std::printf("Batch size    : %d\n", batch_size);
  std::printf("Learning rate : %.1e\n", lr);
  std::printf("Seed          : %d\n", seed);

  // Load & shuffle dataset
  Dataset train_ds = load_dataset(dataset_dir, /*is_train=*/true);
  shuffle_dataset(train_ds);

  Gpu_Autoencoder ae(batch_size);
  ae.init_weights(seed);

  auto t0 = std::chrono::high_resolution_clock::now();
  ae.fit(train_ds, n_epoch, batch_size, lr, seed, checkpoint, out_dir);
  auto t1 = std::chrono::high_resolution_clock::now();

  double total_sec = std::chrono::duration<double>(t1 - t0).count();
  double time_per_epoch = total_sec / n_epoch;

  std::printf("GPU (Base) total training time: %.2f s\n", total_sec);
  std::printf("GPU (Base) time per epoch     : %.2f s\n", time_per_epoch);

  float mse = ae.eval(train_ds);
  std::printf("Final reconstruction MSE: %.6f\n", mse);

  ae.save("./checkpoints_gpu/autoencoder_gpu.bin");
  return 0;
}

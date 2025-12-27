#include <cstdio>
#include <chrono>

#include "data_loader.h"
#include "gpu/gpu_autoencoder.h"

int main(int argc, char **argv)
{
  const char *dataset_dir = (argc > 1 ? argv[1] : "./data/cifar-10-batches-bin");
  printf("=== GPU Baseline Training on CIFAR-10 ===\n");
  printf("Dataset dir: %s\n", dataset_dir);

  int n_take = 2000;
  int n_epoch = 5;
  int batch_size = 32;
  float lr = 1e-3f;
  int seed = 42;

  std::printf("=== Phase 2: GPU Baseline on CIFAR-10 ===\n");
  std::printf("Dataset dir   : %s\n", dataset_dir);
  std::printf("Take samples  : %d\n", n_take);
  std::printf("Epochs        : %d\n", n_epoch);
  std::printf("Batch size    : %d\n", batch_size);
  std::printf("Learning rate : %.1e\n", lr);
  std::printf("Seed          : %d\n", seed);

  // Load & shuffle dataset
  Dataset train_ds = load_dataset(dataset_dir, /*is_train=*/true);
  shuffle_dataset(train_ds);

  // Take first n_take samples
  Dataset train_2k = take_first_n(train_ds, n_take);
  printf("[INFO] subset: %d samples\n", train_2k.n);

  Gpu_Autoencoder ae(batch_size);
  ae.init_weights(seed);

  auto t0 = std::chrono::high_resolution_clock::now();
  ae.fit(train_2k, n_epoch, batch_size, lr, seed, 0, nullptr);
  auto t1 = std::chrono::high_resolution_clock::now();

  double total_sec = std::chrono::duration<double>(t1 - t0).count();
  double time_per_epoch = total_sec / n_epoch;

  std::printf("\n=== GPU Baseline Training Completed ===\n");
  std::printf("GPU (Base) total training time: %.2f s\n", total_sec);
  std::printf("GPU (Base) time per epoch     : %.2f s\n", time_per_epoch);

  float mse = ae.eval(train_2k);
  std::printf("Final reconstruction MSE: %.6f\n", mse);
  return 0;
}

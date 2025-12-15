#include <cstdio>
#include <chrono>
#include "constants.h"
#include "data_loader.h"
#include "gpu/gpu_autoencoder_opt.h"

int main(int argc, char **argv)
{
  // 1. Dataset setup
  const char *dataset_dir = (argc > 1 ? argv[1] : "./data/cifar-10-batches-bin");

  // 2. Hyperparameters (Đồng bộ với Phase 1 & 2)
  int n_epoch = 20; // Đã đổi từ 5 -> 20 để so sánh công bằng
  int batch_size = 32;
  float lr = 1e-3f;
  int seed = 42;
  int checkpoint = 0;                        // Thêm biến này
  const char *out_dir = "./checkpoints_opt"; // Folder riêng cho Opt

  std::printf("=== Phase 3: GPU Optimized (Memory Pool) ===\n");
  std::printf("Dataset dir   : %s\n", dataset_dir);
  std::printf("Epochs        : %d\n", n_epoch);
  std::printf("Batch size    : %d\n", batch_size);
  std::printf("Learning rate : %.1e\n", lr);
  std::printf("Seed          : %d\n", seed);

  // 3. Load & shuffle dataset
  Dataset train_ds = load_dataset(dataset_dir, true);
  shuffle_dataset(train_ds);

  // 4. Init Optimized Autoencoder
  Gpu_Autoencoder_Opt ae(batch_size);
  ae.init_weights(seed);

  // 5. Training Loop & Timing
  auto t0 = std::chrono::high_resolution_clock::now();
  // Lưu ý: Đảm bảo hàm fit của Gpu_Autoencoder_Opt nhận đúng tham số này
  ae.fit(train_ds, n_epoch, batch_size, lr, seed, checkpoint, out_dir);
  auto t1 = std::chrono::high_resolution_clock::now();

  double total_sec = std::chrono::duration<double>(t1 - t0).count();
  double time_per_epoch = total_sec / n_epoch;

  std::printf("GPU (Opt) total training time: %.2f s\n", total_sec);
  std::printf("GPU (Opt) time per epoch     : %.2f s\n", time_per_epoch);

  // 6. Evaluate MSE (Thêm vào để kiểm tra độ hội tụ)
  float mse = ae.eval(train_ds);
  std::printf("Final reconstruction MSE: %.6f\n", mse);

  // 7. Save Model
  ae.save("./checkpoints_opt/autoencoder_opt.bin");

  // --- FEATURE EXTRACTION (Giữ lại phần này cho Phase 4) ---
  // Phần này không ảnh hưởng đến việc đo thời gian train ở trên
  std::printf("\n--- Extracting Features for Phase 4 ---\n");

  std::printf("Extracting features from TRAIN set...\n");
  Dataset train_features = ae.encode(train_ds);
  ae.save_features(train_features, "train_features.bin");

  std::printf("Loading & Extracting features from TEST set...\n");
  Dataset test_ds = load_dataset(dataset_dir, false); // is_train = false
  Dataset test_features = ae.encode(test_ds);
  ae.save_features(test_features, "test_features.bin");

  return 0;
}
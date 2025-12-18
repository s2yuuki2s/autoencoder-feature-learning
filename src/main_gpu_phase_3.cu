#include <cstdio>
#include <chrono>

#include "data_loader.h"
#include "gpu/gpu_autoencoder_opt.h"

int main(int argc, char **argv)
{

 const char *dataset_dir = (argc > 1 ? argv[1] : "./data/cifar-10-batches-bin");

 // Hyperparameters
 // Tăng Batch Size lên 64 để tận dụng Shared Memory Tiling tốt hơn
 // (Giúp lấp đầy các SMs trên GPU và che giấu độ trễ truy cập Global Memory)
 int batch_size = 64;
 int n_epoch = 20;
 float lr = 1e-3f; // Có thể thử tăng lên 0.005f hoặc 0.01f nếu loss giảm chậm
 int seed = 42;
 int checkpoint = 0;
 const char *out_dir = "./checkpoints_opt_fused";

 std::printf("=== Phase 3: GPU Advanced Optimization (Shared Mem + Fusion) ===\n");
 std::printf("Dataset dir   : %s\n", dataset_dir);
 std::printf("Epochs        : %d\n", n_epoch);
 std::printf("Batch size    : %d (Optimized for Occupancy)\n", batch_size);
 std::printf("Learning rate : %.1e\n", lr);
 std::printf("Seed          : %d\n", seed);

 // Load & shuffle dataset
 Dataset train_ds = load_dataset(dataset_dir, true);
 shuffle_dataset(train_ds);

 // Init Optimized Autoencoder
 // Class này sử dụng Kernels trong gpu_layers.cu (bản Tiling + Fusion)
 Gpu_Autoencoder_Opt ae(batch_size);
 ae.init_weights(seed);

 // Training Loop & Timing
 auto t0 = std::chrono::high_resolution_clock::now();
 ae.fit(train_ds, n_epoch, batch_size, lr, seed, checkpoint, out_dir);
 auto t1 = std::chrono::high_resolution_clock::now();

 double total_sec = std::chrono::duration<double>(t1 - t0).count();
 double time_per_epoch = total_sec / n_epoch;

 std::printf("GPU (Fused) Total Time    : %.2f s\n", total_sec);
 std::printf("GPU (Fused) Time per Epoch: %.2f s\n", time_per_epoch);

 // 6. Evaluate MSE
 float mse = ae.eval(train_ds);
 std::printf("Final reconstruction MSE: %.6f\n", mse);

 // 7. Save Model Weights
 ae.save("./checkpoints_opt_fused/autoencoder_fused.bin");

 // --- 8. FEATURE EXTRACTION FOR PHASE 4 (SVM) ---
 std::printf("\n--- Extracting Features for Phase 4 (SVM) ---\n");

 // A. Extract Train Features
 std::printf("-> Encoding TRAIN set (50,000 images)...\n");
 // Lưu ý: encode() sẽ dùng GPU forward pass tối ưu để lấy vector (8,8,128) -> flatten
 Dataset train_features = ae.encode(train_ds);
 ae.save_features(train_features, "train_features_opt.bin");

 // B. Extract Test Features
 std::printf("-> Loading & Encoding TEST set (10,000 images)...\n");
 Dataset test_ds = load_dataset(dataset_dir, false); // is_train = false
 Dataset test_features = ae.encode(test_ds);
 ae.save_features(test_features, "test_features_opt.bin");

 std::printf("\n=== Phase 3 Completed Successfully ===\n");
 return 0;
}
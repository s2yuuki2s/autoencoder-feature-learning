#include <cstdio>
#include <chrono>

#include "../include/data_loader.h"
#include "../include/gpu/gpu_autoencoder_opt.h"

int main(int argc, char **argv)
{
 const char *data_dir = (argc > 1) ? argv[1] : "./data/cifar-10-batches-bin";
 printf("=== [PHASE 3 FINAL] FULL TRAIN & EXPORT FOR SVM ===\n");

 // 1. FULL DATASET (Không cắt)
 Dataset train = load_dataset(data_dir, true);
 shuffle_dataset(train); // Shuffle để train tốt hơn
 printf("[INFO] Train Samples: %d (Full)\n", train.n);

 // 2. CONFIG MẠNH
 int batch_size = 64;
 int epochs = 20;
 float lr = 1e-3f;
 int seed = 42;

 Gpu_Autoencoder_Opt model(batch_size);
 model.init_weights(seed);

 // 3. TRAINING
 printf("-> Training Start...\n");
 auto t0 = std::chrono::high_resolution_clock::now();
 model.fit(train, epochs, batch_size, lr, seed, 0, nullptr);
 auto t1 = std::chrono::high_resolution_clock::now();
 printf("[DONE] Training Time: %.2f s\n", std::chrono::duration<double>(t1 - t0).count());

 // 4. EXPORT
 printf("\n--- EXTRACTING FEATURES FOR SVM ---\n");

 printf("-> Encoding TRAIN set...\n");
 Dataset train_features = model.encode(train);
 model.save_features(train_features, "train_features_opt.bin");

 printf("-> Encoding TEST set...\n");
 Dataset test = load_dataset(data_dir, false);
 Dataset test_features = model.encode(test);
 model.save_features(test_features, "test_features_opt.bin");

 printf("\n=== EXPORT COMPLETE ===\n");
 return 0;
}